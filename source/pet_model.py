"""모델 통합: 탐지 → 동물별 크롭 → 품종 분류를 연결합니다.
설명할 함수: PetModel.predict. 세부 구현은 각 담당 모듈에 있습니다.
"""
from config import DETECTION_CONFIDENCE, BREED_CONFIDENCE
from pet_detector import PetDetector
from breed_classifier import BreedClassifier, choose_breed, EXPECTED_CLASSES
from model_loader import (prepare_models, download_verified, BREED_REVISION,
                          YOLO_URL, YOLO_SHA256)


class PetModel:
    def __init__(self, model_dir, detection_conf=DETECTION_CONFIDENCE,
                 breed_conf=BREED_CONFIDENCE):
        detector_path, breed_path = prepare_models(model_dir)
        self.detector = PetDetector(detector_path, detection_conf)
        self.classifier = BreedClassifier(breed_path, breed_conf)

    def predict(self, image):
        results = []
        for detection in self.detector.detect(image):
            bounds = tuple(detection[key] for key in ("x1", "y1", "x2", "y2"))
            breed = self.classifier.classify(image.crop(bounds), detection["species"])
            results.append({**detection, **breed})
        return results
