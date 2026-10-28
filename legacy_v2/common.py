# [설명] main-2 전용 과거 구현: 입력·탐지·분류 연결·저장. 단계별 변화 비교에 쓰며 최신 공통 코드를 대신하지 않습니다.
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
                        BREED_URL, BREED_SHA256, model_file, BreedClassifier, BREED_CONFIDENCE)
from .visualizer import annotate

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
COUNT_FIELDS = ["filename", "cat", "dog", "actual_cat", "actual_dog",
                "cat_error", "dog_error", "status", "detector", "error", "result_image"]
BREED_FIELDS = ["filename", "object_id", "species", "breed", "candidate_breed",
                "breed_confidence", "detection_confidence", "status", "reason",
                "x1", "y1", "x2", "y2"]


# [설명] 이전 기본 탐지에서 YOLO11n을 불러와 개·고양이를 찾습니다.
def yolo():
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_DIR / "settings"))
    from ultralytics import YOLO
    model = YOLO(str(model_file("yolo11n.pt", YOLO_URL, YOLO_SHA256)))
    if model.names.get(15) != "cat" or model.names.get(16) != "dog":
        raise ValueError("COCO 개·고양이 클래스가 있는 YOLO 모델이 필요합니다.")
    return model


# [설명] 2단계에서 기본 탐지가 0마리일 때 YOLO11s로 추가 검사합니다.
def yolo_rescue():
    """Extra YOLO11s detector loaded only for images with zero primary detections."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_DIR / "settings"))
    from ultralytics import YOLO
    model = YOLO(str(model_file("yolo11s.pt", YOLO_S_URL, YOLO_S_SHA256)))
    if model.names.get(15) != "cat" or model.names.get(16) != "dog":
        raise ValueError("YOLO11s에 COCO cat/dog 클래스가 필요합니다.")
    return model


# [설명] YOLO가 찾은 동물 영역을 37품종 모델로 보낼 준비를 합니다.
def breed_model(confidence=BREED_CONFIDENCE):
    return BreedClassifier(model_file("oxford-pets-resnet50.pth", BREED_URL, BREED_SHA256), confidence)


# [설명] 입력 폴더에 들어 있는 처리 가능한 이미지 목록을 만듭니다.
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


# [설명] 한글 파일명·이미지 회전에도 안전하게 사진을 불러옵니다.
def read_image(path):
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"읽을 수 없는 사진: {Path(path).name}")
    return image


# [설명] 원본을 수정하지 않고 결과 이미지 사본을 저장합니다.
def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    success, encoded = cv2.imencode(path.suffix, image)
    if not success:
        raise OSError(f"결과 이미지 저장 실패: {path}")
    encoded.tofile(path)


# [설명] Excel에서 결과를 확인하도록 한글 CSV를 저장합니다.
def save_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


# [설명] 전신과 얼굴처럼 겹친 박스가 같은 한 마리인지 판단합니다.
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


# [설명] 한 마리를 두 번 세지 않도록 중복 박스 중 한 개만 남깁니다.
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


# [설명] YOLO의 개·고양이 박스마다 ResNet50 품종을 추론합니다.
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


# [설명] 이미지에 개체 수·박스·품종을 표시합니다.
def draw_pets(image, predictions):
    rgb = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return cv2.cvtColor(np.asarray(annotate(rgb, predictions)), cv2.COLOR_RGB2BGR)


# [설명] 입력 사진명과 구분되는 결과 파일 이름을 만듭니다.
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
# [설명] 1·2·3단계가 같은 수기 정답표를 읽도록 공통 입력 모듈을 사용합니다.
from shared.truth_reader import read_ground_truth
