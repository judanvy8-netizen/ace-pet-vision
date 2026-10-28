# [설명] 3단계: 추가 탐지·종 충돌 검토·불확실 품종 보완·자동 평가를 연결한 현재 실행본입니다.
"""기본 제공 05번 main.py 흐름에 품종 분류와 정답 비교를 추가했습니다."""
from pathlib import Path
import argparse
import sys
import logging

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.common import (yolo, image_files, read_image, save_image, save_csv,
                           breed_model, yolo_rescue, yolo_strong, zoomed_pet_rescue, classify_animal_proposals, classify_pets, draw_pets, result_path,
                           read_ground_truth, read_photo_sources, single_pet_review,
                           COUNT_FIELDS, BREED_FIELDS)
from shared.pet_model import prepare_models
from shared.evaluation import write_evaluation

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
    ap.add_argument("--deep-rescue", action="store_true",
                    help="하위 호환용 옵션: 고해상도 재검사는 이제 자동 실행됩니다.")
    ap.add_argument("--single-pet-review", action="store_true",
                    help="동물이 반드시 1마리라는 사람이 확인한 사진에서만 사용: 검출 실패 시 품종만 추정해 검토용으로 기록")
    ap.add_argument("--limit", type=int, default=20, help="처음 20장; 0이면 전체")
    ap.add_argument("--truth", type=Path, default=ROOT / "data" / "ground_truth.csv")
    ap.add_argument("--metadata", type=Path, default=ROOT / "data" / "metadata.csv",
                    help="출처 링크를 읽을 CSV (기본 data/metadata.csv)")
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
    # [설명] 정답표는 사후 평가 전용이며 사진 추론이나 품종 선정에 사용하지 않습니다.
    truth = read_ground_truth((ROOT / args.truth).resolve())
    sources = read_photo_sources((ROOT / args.metadata).resolve())
    print(f"입력: {source} / 처리: {len(files)}장", flush=True)
    # 필수 기능(개·고양이 탐지 + 37품종 분류)을 먼저 준비합니다.
    # 모델이 없거나 손상되어 실행에 실패하면 기존 CSV를 지우지 않습니다.
    # [설명] 모델 로딩에 실패하더라도 기존 결과 CSV를 지우지 않도록 초기화 순서를 지킵니다.
    model = yolo()
    classifier = breed_model(args.breed_conf)
    ids = [i for i, name in model.names.items() if name in TARGETS]
    if len(ids) != 2:
        raise ValueError("YOLO 모델에 cat/dog 탐지 클래스가 모두 필요합니다.")
    output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=output / "errors.log", filemode="w", level=logging.WARNING, encoding="utf-8", force=True)
    rows, breeds, used = [], [], set()
    save_csv(output / "pet_counts.csv", COUNT_FIELDS, rows)
    save_csv(output / "pet_breeds.csv", BREED_FIELDS, breeds)
    # [설명] 기본 탐지가 실패할 때만 추가 탐지기를 로드해 계산 비용을 줄입니다.
    rescue_model = None  # Load extra detectors only for missed photos.
    strong_model = None
    # [설명] 이미지별 탐지·품종·검토 상태·오류를 추적합니다.
    for index, path in enumerate(files, 1):
        filename = path.relative_to(input_root).as_posix()
        row = dict.fromkeys(COUNT_FIELDS, "")
        row.update(filename=filename, status="error", source_url=sources.get(filename, ""))
        try:
            image = read_image(path)
            result = model.predict(image, classes=ids, conf=args.conf, iou=0.45,
                                   imgsz=640, device="cpu", verbose=False)[0]
            pets = classify_pets(result, image, classifier)
            detector_name = "yolo11n"
            # [설명] 0마리일 때 YOLO11s와 고해상도 TTA를 순서대로 시도합니다.
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
                if not pets:
                    # This stage is automatic: no extra CLI flag is needed.
                    deep_result = rescue_model.predict(
                        image, classes=rescue_ids, conf=0.06, iou=0.45,
                        imgsz=1280, augment=True, device="cpu", verbose=False,
                    )[0]
                    pets = classify_pets(deep_result, image, classifier)
                    detector_name = "yolo11s_tta"
                # [설명] 계속 0마리이면 YOLO11m이 다른 동물로 잘못 분류한 실제 박스까지 확인합니다.
                if not pets and not args.single_pet_review:
                    # Try a stronger model on the full photo and then zoomed crops.
                    # Never count an animal without an actual detector box.
                    try:
                        if strong_model is None:
                            strong_model = yolo_strong()
                        # No COCO class filter here: a partly hidden Samoyed
                        # may have been labeled as bear/sheep rather than dog.
                        strong_result = strong_model.predict(
                            image, classes=None, conf=0.05, iou=0.45,
                            imgsz=960, device="cpu", verbose=False,
                        )[0]
                        pets = classify_pets(strong_result, image, classifier)
                        detector_name = "yolo11m"
                        if not pets:
                            pets = classify_animal_proposals(strong_result, image, classifier)
                            if pets:
                                detector_name = "yolo11m_other_animal_review"
                        if not pets:
                            # Store actual raw labels for diagnosis, not guessed causes.
                            raw_labels = [
                                f'{strong_result.names[int(box.cls.item())]}:{float(box.conf.item()):.3f}'
                                for box in strong_result.boxes
                            ]
                            logging.warning("%s: YOLO11m detected labels before filtering: %s",
                                            filename, ", ".join(raw_labels[:12]) or "(none)")
                            pets = zoomed_pet_rescue(image, strong_model, classifier)
                            detector_name = ("yolo11m_zoomed_other_animal_review"
                                             if pets and any(p.get("reason", "").startswith("other_animal_class:")
                                                             for p in pets)
                                             else "yolo11m_zoomed")
                    except (OSError, ValueError, RuntimeError) as extra_error:
                        # If the extra weight download fails, preserve normal
                        # no_detection output and surface the warning.
                        logging.warning("추가 탐지 모델 준비/실행 실패: %s - %s", filename, extra_error)
                        row["error"] = "추가 탐지 실패: " + str(extra_error)
                        print(f"  [경고] {filename}: YOLO11m 추가 검사 실패 — {extra_error}", flush=True)
                        detector_name = "yolo11m_unavailable"
            # [설명] 한 마리 사진이라는 수동 확인이 있을 때만 품종 추측을 별도 REVIEW로 남깁니다.
            if not pets and args.single_pet_review:
                # The user explicitly confirms exactly one animal in this photo.
                # This is NOT a detected box, and must not be scored as detection.
                pets = single_pet_review(image, classifier)
                detector_name = "resnet_single_pet_review"
            counts = {name: sum(p["species"] == name for p in pets) for name in TARGETS}
            target = result_path(filename, used)
            save_image(output / target, draw_pets(image, pets))
            # [설명] 정상 탐지와 모델 충돌·미탐·오류를 구분해 검증 지표가 부풀려지지 않게 합니다.
            result_status = ("review_only" if detector_name == "resnet_single_pet_review"
                             else "review_detection" if any(
                                 p.get("reason", "").startswith("other_animal_class:") for p in pets)
                             else "review_species" if any(
                                 p.get("reason") == "species_corrected_high_confidence" for p in pets)
                             else "partial_error" if row["error"] and not pets
                             else "ok" if pets else "no_detection")
            row.update(**counts, status=result_status, detector=detector_name,
                       result_image=target.as_posix())
            for number, pet in enumerate(pets, 1):
                record = {"filename": filename, "object_id": number, **pet}
                breeds.append({key: record.get(key, "") for key in BREED_FIELDS})
            print(f'[{index}/{len(files)}] {filename}: 개 {counts["dog"]}마리 / 고양이 {counts["cat"]}마리 '
                  f'({detector_name})', flush=True)
            for pet in pets:
                label = pet["breed"] if pet["status"] == "classified" else f'{pet["candidate_breed"]} (검토)'
                print(f'  최종={pet["species"]} {label} {pet["breed_confidence"]:.1%} | '
                      f'YOLO={pet.get("yolo_label", "?")} {pet["detection_confidence"]:.1%} | '
                      f'ResNet1위={pet.get("resnet_top_breed", pet["candidate_breed"])} '
                      f'{pet.get("resnet_top_confidence", pet["breed_confidence"]):.1%} '
                      f'| 판단={pet.get("reason", "-") or "일치"}')
        except Exception as error:
            row["error"] = str(error)
            logging.exception("사진 처리 실패: %s", filename)
            print(f"[{index}/{len(files)}] {filename}: 처리 실패 — {error}", flush=True)
        # [설명] 정상 탐지는 직접 개수 오차를 계산하고 REVIEW는 뒤의 별도 평가에서 다룹니다.
        for species in TARGETS:
            actual = truth.get(filename, {}).get("actual_" + species)
            row["actual_" + species] = actual if actual is not None else ""
            if actual is not None and row["status"] in ("ok", "no_detection"):
                row[species + "_error"] = abs(row[species] - actual)
        rows.append(row)
        save_csv(output / "pet_counts.csv", COUNT_FIELDS, rows)
        save_csv(output / "pet_breeds.csv", BREED_FIELDS, breeds)
    try:
        # [설명] 최종 개수와 수기 정답의 비교 보고서를 outputs에 생성합니다.
        stats = write_evaluation(rows, breeds, truth, (ROOT / args.metadata).resolve(), output)
        print(f"검증: 수작업 정답 {stats['truth_confirmed']}/{stats['processed']}장 / "
              f"정상 탐지 개수 일치 {stats['count_exact']}/{stats['count_scored']}장 "
              f"(REVIEW 제외) / 최종 교차 판정 개수 일치 "
              f"{stats['final_count_exact']}/{stats['final_count_scored']}장", flush=True)
    except (OSError, ValueError) as error:
        logging.exception("검증 요약 파일 생성 실패")
        print(f"[경고] 검증 요약 생성 실패: {error}", flush=True)
    print(f"결과: {output}")
    return int(any(row["status"] in ("error", "partial_error") for row in rows))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"실행 실패: {error}", file=sys.stderr)
        raise SystemExit(1)
