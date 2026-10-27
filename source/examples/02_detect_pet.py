"""예제 2: 탐지 담당 모듈을 사용해 개·고양이 위치를 표시합니다."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import INPUT_DIR, MODEL_DIR, OUTPUT_DIR
from data_loader import read_image
from model_loader import download_verified, YOLO_URL, YOLO_SHA256
from pet_detector import PetDetector
from visualizer import annotate, save_image

if __name__ == "__main__":
    photo_path = INPUT_DIR / "beagle.jpg"  # 다른 사진으로 바꿔 실습하세요.
    weights = download_verified(YOLO_URL, MODEL_DIR / "yolo11n.pt", YOLO_SHA256)
    photo = read_image(photo_path)
    detections = PetDetector(weights).detect(photo)
    for animal in detections:
        print(animal)
    # 탐지만 수행하므로 품종 결과는 표시하지 않습니다.
    labels = [{**animal, "breed": "not classified", "status": "detected"}
              for animal in detections]
    output = OUTPUT_DIR / "examples" / "detection.png"
    save_image(annotate(photo, labels), output)
    print("탐지 이미지 저장:", output)
