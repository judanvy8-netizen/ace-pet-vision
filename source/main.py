"""IntelliJ에서 실행: input/images -> 동물 탐지·품종 분류 -> output."""
import argparse
import csv
from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
FIELDS = ["filename", "object_id", "species", "breed", "candidate_breed", "breed_confidence",
          "detection_confidence", "status", "reason", "x1", "y1", "x2", "y2", "top3_json",
          "result_image", "crop_image", "image_processing_ms"]
SUMMARY_FIELDS = ["filename", "status", "animal_count", "processing_ms", "error", "result_image"]


def project_path(path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def find_images(folder):
    if folder.is_file():
        return [folder] if folder.suffix.lower() in IMAGE_EXTENSIONS else []
    if not folder.is_dir():
        raise ValueError(f"입력 폴더가 없습니다: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def write_csv(path, rows, fields):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def annotate(image, predictions):
    from PIL import ImageDraw, ImageFont
    result = image.copy()
    draw = ImageDraw.Draw(result)
    size = max(12, min(28, round(image.width / 50)))
    font = ImageFont.load_default(size=size)
    for number, prediction in enumerate(predictions, 1):
        x1, y1, x2, y2 = (prediction[k] for k in ("x1", "y1", "x2", "y2"))
        color = (50, 210, 120) if prediction["species"] == "dog" else (80, 180, 255)
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), outline=color, width=max(2, round(image.width / 500)))
        label = (f'#{number} {prediction["species"]}: {prediction["breed"]}'
                 + (f' {prediction["breed_confidence"]:.0%}' if prediction["status"] == "classified" else " (review)"))
        current_font = font
        while draw.textbbox((0, 0), label, font=current_font)[2] > image.width - 8 and size > 8:
            size -= 1
            current_font = ImageFont.load_default(size=size)
        box = draw.textbbox((0, 0), label, font=current_font)
        width, height = box[2] + 8, box[3] - box[1] + 8
        x = min(x1, max(0, image.width - width))
        y = max(0, y1 - height)
        draw.rectangle((x, y, min(image.width - 1, x + width), y + height), fill=color)
        draw.text((x + 4, y + 4 - box[1]), label, font=current_font, fill=(0, 0, 0))
    if not predictions:
        draw.text((8, 8), "No cat/dog detected", font=font, fill=(255, 70, 70), stroke_width=1, stroke_fill=(0, 0, 0))
    return result


def process_images(files, input_root, run_dir, model):
    from PIL import Image, ImageOps
    all_rows, summary = [], []
    for index, path in enumerate(files, 1):
        filename = path.relative_to(input_root).as_posix()
        started = time.perf_counter()
        info = {"filename": filename, "status": "error", "animal_count": 0,
                "processing_ms": 0, "error": "", "result_image": ""}
        try:
            with Image.open(path) as raw:
                image = ImageOps.exif_transpose(raw).convert("RGB")
            predictions = model.predict(image)
            # 확장자도 보존하여 같은 이름의 JPG/PNG가 서로 덮어쓰지 않게 합니다.
            result_path = Path("images") / (filename + ".png")
            destination = run_dir / result_path
            destination.parent.mkdir(parents=True, exist_ok=True)
            annotate(image, predictions).save(destination)
            rows = []
            for number, prediction in enumerate(predictions, 1):
                crop_path = Path("crops") / filename / f"animal_{number:02d}.png"
                (run_dir / crop_path).parent.mkdir(parents=True, exist_ok=True)
                image.crop(tuple(prediction[k] for k in ("x1", "y1", "x2", "y2"))).save(run_dir / crop_path)
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
    mode.add_argument("--samples", action="store_true", help="input/images의 기존 예제 사진 사용")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "oxford-iiit-pet")
    parser.add_argument("--split", choices=["test", "trainval"], default="test")
    parser.add_argument("--limit", type=int, default=20, help="처음 처리할 사진 수. 기본 20장, 0이면 전체")
    parser.add_argument("--output", type=Path, default=ROOT / "output")
    parser.add_argument("--conf", type=probability, default=0.25, help="동물 탐지 임계값")
    parser.add_argument("--breed-conf", type=probability, default=0.5, help="품종 후보 보류 임계값")
    parser.add_argument("--prepare-models", action="store_true", help="사진 처리 없이 모델만 미리 다운로드")
    args = parser.parse_args(argv)
    if args.limit < 0:
        parser.error("--limit는 0 이상이어야 합니다.")
    from pet_model import PetModel, prepare_models, BREED_REVISION
    if args.prepare_models:
        prepare_models(ROOT / "models")
        print("모델 준비 완료. 이제 input/images에 사진을 넣고 main.py를 실행하세요.")
        return 0
    truth = None
    output = project_path(args.output)
    if args.input is not None or args.samples:
        source = project_path(args.input or ROOT / "input" / "images")
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
        model = PetModel(ROOT / "models", args.conf, args.breed_conf)
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
