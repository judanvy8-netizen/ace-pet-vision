"""기본 제공 main.py의 공통 함수 규격을 구현합니다.
원본 shared/common.py는 제공되지 않아, 확인한 호출 규격을 기준으로 작성했습니다.
"""
from pathlib import Path
import csv
import math
import os
import cv2
import numpy as np
from PIL import Image
from .pet_model import (MODEL_DIR, YOLO_URL, YOLO_SHA256, YOLO_S_URL, YOLO_S_SHA256,
                        YOLO_M_URL, YOLO_M_SHA256,
                        BREED_URL, BREED_SHA256, model_file, BreedClassifier, BREED_CONFIDENCE)
from .visualizer import annotate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COUNT_FIELDS = ["filename", "cat", "dog", "actual_cat", "actual_dog",
                "cat_error", "dog_error", "status", "detector", "source_url", "error", "result_image"]
BREED_FIELDS = ["filename", "object_id", "species", "breed", "candidate_breed",
                "breed_confidence", "detection_confidence", "status", "reason",
                "x1", "y1", "x2", "y2"]


def yolo():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_DIR / "settings"))
    from ultralytics import YOLO
    model = YOLO(str(model_file("yolo11n.pt", YOLO_URL, YOLO_SHA256)))
    if model.names.get(15) != "cat" or model.names.get(16) != "dog":
        raise ValueError("COCO 개·고양이 클래스가 있는 YOLO 모델이 필요합니다.")
    return model


def yolo_rescue():
    """Extra YOLO11s detector loaded only for images with zero primary detections."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_DIR / "settings"))
    from ultralytics import YOLO
    model = YOLO(str(model_file("yolo11s.pt", YOLO_S_URL, YOLO_S_SHA256)))
    if model.names.get(15) != "cat" or model.names.get(16) != "dog":
        raise ValueError("YOLO11s에 COCO cat/dog 클래스가 필요합니다.")
    return model


def yolo_strong():
    """More capable detector; load only if YOLO11n and YOLO11s both miss."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_DIR / "settings"))
    from ultralytics import YOLO
    model = YOLO(str(model_file("yolo11m.pt", YOLO_M_URL, YOLO_M_SHA256)))
    if model.names.get(15) != "cat" or model.names.get(16) != "dog":
        raise ValueError("YOLO11m에 COCO cat/dog 클래스가 필요합니다.")
    return model


def zoomed_pet_rescue(image, model, classifier, *, conf=0.07):
    """Run real YOLO detection on overlapping crops; restore original coordinates.

    This is used only after all full-image attempts fail. It never constructs a
    detection box from filenames, metadata or breed-classification predictions.
    """
    height, width = image.shape[:2]
    if min(height, width) < 50:
        return []
    regions = [
        (int(width * .08), int(height * .08), int(width * .92), int(height * .92)),
        (0, 0, int(width * .78), height),
        (int(width * .22), 0, width, height),
    ]
    ids = [i for i, name in model.names.items() if name in ("cat", "dog")]
    for left, top, right, bottom in regions:
        crop = image[top:bottom, left:right]
        if crop.size == 0:
            continue
        result = model.predict(crop, classes=ids, conf=conf, iou=0.45,
                               imgsz=960, device="cpu", verbose=False)[0]
        pets = classify_pets(result, crop, classifier)
        if pets:
            for pet in pets:
                pet["x1"] += left
                pet["x2"] += left
                pet["y1"] += top
                pet["y2"] += top
            return deduplicate_pets(pets)
    return []


def breed_model(confidence=BREED_CONFIDENCE):
    return BreedClassifier(model_file("oxford-pets-resnet50.pth", BREED_URL, BREED_SHA256), confidence)


def image_files(source):
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"지원하지 않는 이미지 형식: {source.name}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(f"입력 폴더가 없습니다: {source}")
    files = sorted(p for p in source.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not files:
        raise ValueError(f"사진이 없습니다: {source}")
    return files


def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"읽을 수 없는 사진: {Path(path).name}")
    return image


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"결과 이미지 저장 실패: {path}")
    encoded.tofile(path)


def save_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _same_animal_box(left, right):
    """Treat very similar or nested boxes of the same species as a duplicate.

    Mere overlap between different animals is not sufficient; crowded scenes
    still require manual count validation.
    """
    if left["species"] != right["species"]:
        return False
    ax1, ay1, ax2, ay2 = (left[k] for k in ("x1", "y1", "x2", "y2"))
    bx1, by1, bx2, by2 = (right[k] for k in ("x1", "y1", "x2", "y2"))
    overlap_w = max(0, min(ax2, bx2) - max(ax1, bx1))
    overlap_h = max(0, min(ay2, by2) - max(ay1, by1))
    intersection = overlap_w * overlap_h
    if not intersection:
        return False
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    if min(area_a, area_b) <= 0:
        return False
    iou = intersection / (area_a + area_b - intersection)
    small_coverage = intersection / min(area_a, area_b)
    return iou >= 0.55 or (small_coverage >= 0.90 and iou >= 0.25)


def deduplicate_pets(predictions):
    """Keep the most credible breed prediction within each duplicate group."""
    ranked = sorted(
        predictions,
        key=lambda pet: (pet["status"] == "classified", float(pet["breed_confidence"]),
                         float(pet["detection_confidence"])),
        reverse=True,
    )
    kept = []
    for candidate in ranked:
        if not any(_same_animal_box(candidate, earlier) for earlier in kept):
            kept.append(candidate)
    return sorted(kept, key=lambda pet: (pet["x1"], pet["y1"], pet["x2"], pet["y2"]))


def classify_pets(result, image, classifier):
    """기본 YOLO 결과의 동물 영역을 잘라 추가 품종 모델에 전달합니다."""
    height, width = image.shape[:2]
    predictions = []
    for box in result.boxes:
        species = result.names[int(box.cls.item())]
        if species not in ("cat", "dog"):
            continue
        a, b, c, d = box.xyxy[0].cpu().tolist()
        x1, y1, x2, y2 = max(0, math.floor(a)), max(0, math.floor(b)), min(width, math.ceil(c)), min(height, math.ceil(d))
        if x1 >= x2 or y1 >= y2:
            continue
        crop = Image.fromarray(cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2RGB))
        breed = classifier.classify(crop, species)
        predictions.append({"species": species, "detection_confidence": float(box.conf.item()),
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2, **breed})
    return deduplicate_pets(predictions)


def draw_pets(image, predictions):
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.asarray(annotate(rgb, predictions)), cv2.COLOR_RGB2BGR)


def result_path(relative, used):
    relative = Path(relative)
    path = relative.with_name(relative.stem + "_pets.jpg")
    number = 2
    while path.as_posix().casefold() in used:
        path = relative.with_name(f"{relative.stem}_{number}_pets.jpg")
        number += 1
    used.add(path.as_posix().casefold())
    return path


def read_photo_sources(path):
    """Return original source-page URLs from metadata.csv, keyed by input filename.

    Photos not present in the metadata deliberately get an empty source field;
    this program never invents a copyright or attribution claim.
    """
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "source_page"}.issubset(reader.fieldnames or []):
            raise ValueError("사진 출처표에 filename, source_page 열이 필요합니다.")
        sources = {}
        for row in reader:
            filename = (row["filename"] or "").strip().replace("\\", "/")
            source_url = (row["source_page"] or "").strip()
            if filename and filename not in sources:
                sources[filename] = source_url if source_url.startswith(("https://", "http://")) else ""
    return sources


def single_pet_review(image, classifier):
    """Explicit user-confirmed single-animal photo fallback; NOT YOLO detection.

    Never call this automatically on unknown or no-animal photos.
    """
    height, width = image.shape[:2]
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    prediction = classifier.classify_single_pet(rgb)
    return [{"x1": 0, "y1": 0, "x2": width, "y2": height,
             "detection_confidence": 0.0, **prediction}]


def read_ground_truth(path):
    """정답은 사람이 작성합니다. 빈 칸을 0마리로 취급하지 않습니다."""
    if not path.is_file():
        return {}
    truth = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "actual_cat", "actual_dog"}.issubset(reader.fieldnames or []):
            raise ValueError("정답표에 filename, actual_cat, actual_dog 열이 필요합니다.")
        for row in reader:
            name = row["filename"].strip().replace("\\", "/")
            if not name or name in truth:
                raise ValueError(f"정답표 파일명이 비었거나 중복입니다: {name}")
            counts = {}
            for key in ("actual_cat", "actual_dog"):
                value = (row[key] or "").strip()
                if value and not (value.isascii() and value.isdigit()):
                    raise ValueError(f"정답표 {name}: {key}는 0 이상의 정수 또는 빈 칸이어야 합니다.")
                counts[key] = int(value) if value else None
            truth[name] = counts
    return truth


