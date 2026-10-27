"""동물 탐지 담당: YOLO11n으로 개·고양이의 위치를 찾습니다.
설명할 함수: detect. 탐지 기준은 config.DETECTION_CONFIDENCE입니다.
"""
import math
import os
from pathlib import Path
from config import DETECTION_CONFIDENCE


class PetDetector:
    def __init__(self, weights_path, confidence=DETECTION_CONFIDENCE):
        os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(weights_path).parent / "ultralytics_settings"))
        from ultralytics import YOLO
        self.detector = YOLO(str(weights_path))
        if self.detector.names.get(15) != "cat" or self.detector.names.get(16) != "dog":
            raise ValueError("개/고양이 클래스가 있는 COCO 탐지 모델이 필요합니다.")
        self.detection_conf = confidence

    def detect(self, image):
        import numpy as np
        bgr = np.asarray(image)[:, :, ::-1].copy()
        boxes = self.detector.predict(bgr, classes=[15, 16], conf=self.detection_conf,
                                      device="cpu", imgsz=640, verbose=False)[0].boxes
        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
            bounds = (max(0, math.floor(x1)), max(0, math.floor(y1)),
                      min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
            if bounds[0] >= bounds[2] or bounds[1] >= bounds[3]:
                continue
            detections.append((bounds, "cat" if int(box.cls.item()) == 15 else "dog", float(box.conf.item())))
        return [
            {"species": species, "detection_confidence": score,
             **dict(zip(("x1", "y1", "x2", "y2"), bounds))}
            for bounds, species, score in sorted(detections, key=lambda item: item[0])
        ]
