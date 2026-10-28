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
from .pet_model import (MODEL_DIR, YOLO_URL, YOLO_SHA256, BREED_URL, BREED_SHA256,
                        model_file, BreedClassifier, BREED_CONFIDENCE)
from .visualizer import annotate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COUNT_FIELDS = ["filename", "cat", "dog", "actual_cat", "actual_dog",
                "cat_error", "dog_error", "status", "error", "result_image"]
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
    return sorted(predictions, key=lambda p: (p["x1"], p["y1"], p["x2"], p["y2"]))


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

# 세 버전 모두 동일한 수기 정답표 형식(제목 없는 CSV)을 읽습니다.
from shared.truth_reader import read_ground_truth
