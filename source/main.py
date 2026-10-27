"""통합·실행 담당: 입력 선택 → 탐지·분류 → 결과 저장.
IntelliJ에서 이 파일을 실행합니다. 예제 사진은 --samples 옵션을 사용하세요.
설명할 함수: main, process_images. 기본 실행값은 config.py에서 관리합니다.
"""
import argparse
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

from config import (INPUT_DIR, DATASET_DIR, MODEL_DIR, OUTPUT_DIR,
                    DETECTION_CONFIDENCE, BREED_CONFIDENCE, IMAGE_LIMIT)
from data_loader import project_path, find_images, read_image
from visualizer import annotate, save_image, write_csv, FIELDS, SUMMARY_FIELDS


def process_images(files, input_root, run_dir, model):
    all_rows, summary = [], []
    for index, path in enumerate(files, 1):
        filename = path.relative_to(input_root).as_posix()
        started = time.perf_counter()
        info = {"filename": filename, "status": "error", "animal_count": 0,
                "processing_ms": 0, "error": "", "result_image": ""}
        try:
            image = read_image(path)
            predictions = model.predict(image)
            # 확장자도 보존하여 같은 이름의 JPG/PNG가 서로 덮어쓰지 않게 합니다.
            result_path = Path("images") / (filename + ".png")
            destination = run_dir / result_path
            save_image(annotate(image, predictions), destination)
            rows = []
            for number, prediction in enumerate(predictions, 1):
                crop_path = Path("crops") / filename / f"animal_{number:02d}.png"
                crop = image.crop(tuple(prediction[k] for k in ("x1", "y1", "x2", "y2")))
                save_image(crop, run_dir / crop_path)
                details = {key: value for key, value in prediction.items() if key != "top3"}
                rows.append({"filename": filename, "object_id": number, **details,
                             "top3_json": json.dumps(prediction["top3"], ensure_ascii=False),
                             "result_image": result_path.as_posix(), "crop_image": crop_path.as_posix(),
                             "image_processing_ms": 0})
            elapsed = round((time.perf_counter() - started) * 1000, 2)
            for row in rows:
                row["image_processing_ms"] = elapsed
            all_rows.extend(rows)
            info.update(status="ok" if predictions else "no_detection", animal_count=len(predictions),
                        result_image=result_path.as_posix())
        except Exception as error:
            info["error"] = f"{type(error).__name__}: {error}"
            logging.exception("사진 처리 실패: %s", filename)
        info["processing_ms"] = round((time.perf_counter() - started) * 1000, 2)
        summary.append(info)
        write_csv(run_dir / "results.csv", all_rows, FIELDS)
        write_csv(run_dir / "image_summary.csv", summary, SUMMARY_FIELDS)
        print(f'[{index}/{len(files)}] {filename}: {info["status"]}, {info["animal_count"]}마리', flush=True)
    return all_rows, summary


def probability(value):
    value = float(value)
    if not 0 <= value <= 1:
        raise argparse.ArgumentTypeError("0~1 사이의 값을 입력하세요.")
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description="ACE 반려동물 탐지·Oxford 37품종 분류")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--input", type=Path, help="직접 준비한 사진 폴더 또는 사진 한 장")
    mode.add_argument("--samples", action="store_true", help="data/images의 기존 예제 사진 사용")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument("--split", choices=["test", "trainval"], default="test")
    parser.add_argument("--limit", type=int, default=IMAGE_LIMIT, help="처음 처리할 사진 수. 0이면 전체; 기본값은 config.py 참조")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--conf", type=probability, default=DETECTION_CONFIDENCE, help="동물 탐지 임계값")
    parser.add_argument("--breed-conf", type=probability, default=BREED_CONFIDENCE, help="품종 후보 보류 임계값")
    parser.add_argument("--prepare-models", action="store_true", help="사진 처리 없이 모델만 미리 다운로드")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit는 0 이상이어야 합니다.")
    from pet_model import PetModel, prepare_models, BREED_REVISION
    if args.prepare_models:
        prepare_models(MODEL_DIR)
        print("모델 준비 완료. Oxford 데이터셋으로 실행하거나 --samples로 기존 예제를 실행하세요.")
        return 0
    truth = None
    output = project_path(args.output)
    if args.input is not None or args.samples:
        source = project_path(args.input or INPUT_DIR)
        files = find_images(source)
        total = len(files)
        if args.limit:
            files = files[:args.limit]
        source_mode = "samples" if args.samples else "custom"
    else:
        from oxford_data import load_split
        source, files, truth, total = load_split(project_path(args.dataset), args.split, args.limit)
        source_mode = "oxford"
    if not files:
        print(f"사진이 없습니다. JPG/PNG 등을 다음 위치에 넣으세요: {source}")
        return 1
    input_root = source.parent if source.is_file() else source
    if output == input_root or input_root in output.parents:
        raise ValueError("결과 폴더는 입력 사진 폴더 바깥에 지정하세요.")
    run_dir = output / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")
    run_dir.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(filename=run_dir / "errors.log", level=logging.ERROR, encoding="utf-8", force=True)
    config = {"input": str(source), "input_count": len(files), "detection_conf": args.conf,
              "breed_conf": args.breed_conf, "breed_model": "flaviodell/oxford-pets-resnet50",
              "breed_revision": BREED_REVISION, "device": "cpu", "completed": False}
    config.update(source_mode=source_mode, available_images=total, limit=args.limit,
                  split=args.split if source_mode == "oxford" else None)
    config_path = run_dir / "run_config.json"
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        print("모델 준비 중입니다. 첫 실행에는 모델 다운로드가 필요합니다.", flush=True)
        model = PetModel(MODEL_DIR, args.conf, args.breed_conf)
        rows, summary = process_images(files, input_root, run_dir, model)
        if truth is not None:
            from oxford_data import compare_predictions
            comparisons = compare_predictions(truth, rows, summary)
            write_csv(run_dir / "dataset_comparison.csv", comparisons, list(comparisons[0]))
            config["label_matches"] = sum(row["comparison"] == "match" for row in comparisons)
            config["comparison_policy"] = "largest detected box per image; not a generalization benchmark"
        errors = sum(row["status"] == "error" for row in summary)
        config.update(completed=True, failed_images=errors, detected_animals=len(rows))
        config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n결과 폴더: {run_dir}")
        print("results.csv: 개체별 품종 후보 / image_summary.csv: 미탐·오류 포함 사진별 결과")
        return 1 if errors else 0
    except Exception:
        logging.exception("실행 실패")
        print(f"오류 기록: {run_dir / 'errors.log'}", file=sys.stderr)
        raise


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"실행 실패: {type(error).__name__}: {error}", file=sys.stderr)
        print("라이브러리 오류는 setup.bat을, 다운로드 오류는 인터넷 연결을 확인하세요.", file=sys.stderr)
        raise SystemExit(1)
