# [설명] 2단계: 한 마리 중복 계산과 미탐지를 줄이려고 박스 병합·YOLO11s를 더했습니다.
# 이전 실제 커밋을 바탕으로 보존한 2단계 버전
"""기본 제공 05번 main.py 흐름에 품종 분류와 정답 비교를 추가했습니다."""
from pathlib import Path
import argparse
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from legacy_v2.common import (yolo, image_files, read_image, save_image, save_csv,
                           breed_model, yolo_rescue, classify_pets, draw_pets, result_path,
                           read_ground_truth, COUNT_FIELDS, BREED_FIELDS)
from legacy_v2.pet_model import prepare_models

TARGETS = ("cat", "dog")
SAMPLE_INPUT = ROOT / "data" / "images"
OUTPUT = ROOT / "outputs"


# [설명] 0~1 사이의 신뢰도 기준값만 받도록 입력을 검사합니다.
def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("0~1 사이의 값을 입력하세요.")
    return value


# [설명] 입력 선택→모델 추론→결과 저장→수기 정답 비교까지 전체 실행 흐름을 관리합니다.
def main(argv=None):
    ap = argparse.ArgumentParser(description="개·고양이 마릿수와 품종 분석")
    default_input = SAMPLE_INPUT
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--input", type=Path, default=default_input)
    mode.add_argument("--samples", action="store_true", help="포함 사진 20장 사용")
    ap.add_argument("--output", type=Path, default=OUTPUT)
    ap.add_argument("--conf", type=probability, default=0.25)
    ap.add_argument("--breed-conf", type=probability, default=0.5)
    ap.add_argument("--no-rescue", action="store_true", help="미탐지 시 YOLO11s 재검사 끄기")
    ap.add_argument("--rescue-conf", type=probability, default=0.18,
                    help="미탐지 재검사의 탐지 신뢰도 기준 (기본 0.18)")
    ap.add_argument("--limit", type=int, default=20, help="처음 20장; 0이면 전체")
    ap.add_argument("--truth", type=Path, default=ROOT / "data" / "ground_truth.csv")
    ap.add_argument("--prepare-models", action="store_true")
    args = ap.parse_args(argv)
    if args.limit < 0:
        ap.error("--limit는 0 이상이어야 합니다.")
    if args.prepare_models:
        prepare_models()
        print("shared/models의 모델 준비 완료")
        return 0
    source = SAMPLE_INPUT if args.samples else args.input
    source = (ROOT / source).resolve()
    output = (ROOT / args.output).resolve()
    files = image_files(source)
    files = files[:args.limit] if args.limit else files
    input_root = source.parent if source.is_file() else source
    if output == input_root or input_root in output.parents:
        raise ValueError("outputs는 입력 폴더 바깥에 두세요.")
    # [설명] 수기 정답은 AI 판단에 쓰지 않고 추론 후 평가할 때만 사용합니다.
    truth = read_ground_truth((ROOT / args.truth).resolve())
    print(f"입력: {source} / 처리: {len(files)}장", flush=True)
    # 필수 기능(개·고양이 탐지 + 37품종 분류)을 먼저 준비합니다.
    # 모델이 없거나 손상되어 실행에 실패하면 기존 CSV를 지우지 않습니다.
    # [설명] 모델 준비가 실패해도 이전 CSV를 지우지 않게 모델을 먼저 로드합니다.
    model = yolo()
    classifier = breed_model(args.breed_conf)
    ids = [i for i, name in model.names.items() if name in TARGETS]
    if len(ids) != 2:
        raise ValueError("YOLO 모델에 cat/dog 탐지 클래스가 모두 필요합니다.")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output / "errors.log", filemode="w", level=logging.ERROR, encoding="utf-8", force=True)
    rows, breeds, used = [], [], set()
    save_csv(output / "pet_counts.csv", COUNT_FIELDS, rows)
    save_csv(output / "pet_breeds.csv", BREED_FIELDS, breeds)
    rescue_model = None  # Load the larger detector only if a photo needs rescue.
    # [설명] 기본 탐지에 실패한 사진만 추가 검사하며 파일별 결과를 기록합니다.
    for index, path in enumerate(files, 1):
        filename = path.relative_to(input_root).as_posix()
        row = dict.fromkeys(COUNT_FIELDS, "")
        row.update(filename=filename, status="error")
        try:
            image = read_image(path)
            result = model.predict(image, classes=ids, conf=args.conf, iou=0.45,
                                   imgsz=640, device="cpu", verbose=False)[0]
            pets = classify_pets(result, image, classifier)
            detector_name = "yolo11n"
            # [설명] 기본 YOLO가 0마리일 때만 큰 YOLO11s를 호출해 CPU 부담을 줄입니다.
            if not pets and not args.no_rescue:
                # Do not fabricate a pet; attempt another actual object detection.
                if rescue_model is None:
                    rescue_model = yolo_rescue()
                rescue_ids = [i for i, name in rescue_model.names.items() if name in TARGETS]
                if len(rescue_ids) != 2:
                    raise ValueError("YOLO11s에 cat/dog 탐지 클래스가 모두 필요합니다.")
                rescue_result = rescue_model.predict(
                    image, classes=rescue_ids, conf=min(args.conf, args.rescue_conf),
                    iou=0.45, imgsz=960, device="cpu", verbose=False,
                )[0]
                pets = classify_pets(rescue_result, image, classifier)
                detector_name = "yolo11s_rescue"
            counts = {name: sum(p["species"] == name for p in pets) for name in TARGETS}
            # [설명] 중복 제거된 개체 목록으로 결과 이미지·마릿수를 기록합니다.
            target = result_path(filename, used)
            save_image(output / target, draw_pets(image, pets))
            row.update(**counts, status="ok" if pets else "no_detection", detector=detector_name,
                       result_image=target.as_posix())
            for number, pet in enumerate(pets, 1):
                record = {"filename": filename, "object_id": number, **pet}
                breeds.append({key: record[key] for key in BREED_FIELDS})
            print(f'[{index}/{len(files)}] {filename}: 개 {counts["dog"]}마리 / 고양이 {counts["cat"]}마리 '
                  f'({detector_name})', flush=True)
            for pet in pets:
                label = pet["breed"] if pet["status"] == "classified" else f'{pet["candidate_breed"]} (보류)'
                print(f'  {pet["species"]}: {label} / 후보 점수 {pet["breed_confidence"]:.1%}')
        except Exception as error:
            row["error"] = str(error)
            logging.exception("사진 처리 실패: %s", filename)
            print(f"[{index}/{len(files)}] {filename}: 처리 실패 — {error}", flush=True)
        # [설명] 사람이 확인한 정답 개수와 탐지 개수의 오차를 계산합니다.
        for species in TARGETS:
            actual = truth.get(filename, {}).get("actual_" + species)
            row["actual_" + species] = actual if actual is not None else ""
            if actual is not None and row["status"] != "error":
                row[species + "_error"] = abs(row[species] - actual)
        rows.append(row)
        save_csv(output / "pet_counts.csv", COUNT_FIELDS, rows)
        save_csv(output / "pet_breeds.csv", BREED_FIELDS, breeds)
    print(f"결과: {output}")
    return int(any(row["status"] == "error" for row in rows))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"실행 실패: {error}", file=sys.stderr)
        raise SystemExit(1)
