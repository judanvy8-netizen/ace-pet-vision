"""AI 처리: 모델 다운로드 → 동물 탐지 → 크롭 → 37품종 분류."""
import hashlib
import math
import os
from pathlib import Path
import tempfile
import urllib.request

DETECTION_CONFIDENCE = 0.25
BREED_CONFIDENCE = 0.50

# 1. 모델 파일 준비
BREED_REVISION = "3253ad292cf0778f5b64fa5a9156f2bdb0a84a3b"
BREED_URL = f"https://huggingface.co/flaviodell/oxford-pets-resnet50/resolve/{BREED_REVISION}/best_model.pth"
BREED_SHA256 = "889b2d4919ea0c48cb8a89e2b25ff7c56af1bd3e1627e36b36d63c495b3842d4"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_verified(url, destination, expected_hash):
    """가중치를 검증한 뒤 원자적으로 교체. 손상된 다운로드를 모델로 읽지 않습니다."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and sha256(destination) == expected_hash:
        return destination
    print(f"모델 다운로드: {destination.name}", flush=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".part", delete=False) as output:
            temporary = Path(output.name)
            request = urllib.request.Request(url, headers={"User-Agent": "ACE-Pet-Vision/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                received = 0
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    output.write(chunk)
                    received += len(chunk)
                    if received % (10 * 1024 * 1024) == 0:
                        print(f"  {received // (1024 * 1024)} MB 다운로드", flush=True)
        if sha256(temporary) != expected_hash:
            raise RuntimeError(f"모델 파일 검증 실패: {destination.name}. 인터넷 연결을 확인하고 다시 실행하세요.")
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination


def prepare_models(model_dir):
    model_dir = Path(model_dir)
    detector = download_verified(YOLO_URL, model_dir / "yolo11n.pt", YOLO_SHA256)
    classifier = download_verified(BREED_URL, model_dir / "oxford-pets-resnet50.pth", BREED_SHA256)
    return detector, classifier

# 2. 품종 분류
CAT_BREEDS = {"Abyssinian", "Bengal", "Birman", "Bombay", "British Shorthair",
              "Egyptian Mau", "Maine Coon", "Persian", "Ragdoll", "Russian Blue", "Siamese", "Sphynx"}
EXPECTED_CLASSES = (
    "Abyssinian", "American Bulldog", "American Pit Bull Terrier", "Basset Hound", "Beagle",
    "Bengal", "Birman", "Bombay", "Boxer", "British Shorthair", "Chihuahua", "Egyptian Mau",
    "English Cocker Spaniel", "English Setter", "German Shorthaired", "Great Pyrenees", "Havanese",
    "Japanese Chin", "Keeshond", "Leonberger", "Maine Coon", "Miniature Pinscher", "Newfoundland",
    "Persian", "Pomeranian", "Pug", "Ragdoll", "Russian Blue", "Saint Bernard", "Samoyed",
    "Scottish Terrier", "Shiba Inu", "Siamese", "Sphynx", "Staffordshire Bull Terrier",
    "Wheaten Terrier", "Yorkshire Terrier",
)


def choose_breed(probabilities, classes, species, threshold):
    """37개 전체 softmax 값을 사용하며, 종별 재정규화를 하지 않습니다."""
    order = sorted(range(len(classes)), key=lambda i: float(probabilities[i]), reverse=True)
    best = order[0]
    candidate, score = classes[best], float(probabilities[best])
    predicted_species = "cat" if candidate in CAT_BREEDS else "dog"
    reason = "species_mismatch" if predicted_species != species else ("low_confidence" if score < threshold else "")
    return {"breed": "unknown" if reason else candidate, "candidate_breed": candidate,
            "breed_confidence": score, "status": "review" if reason else "classified", "reason": reason,
            "top3": [{"breed": classes[i], "score": float(probabilities[i])} for i in order[:3]]}



class BreedClassifier:
    def __init__(self, weights_path, confidence=BREED_CONFIDENCE):
        import torch
        from torchvision import models, transforms
        # 임의의 Python 객체를 실행하지 않는 제한된 가중치 로딩을 사용합니다.
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.classes = checkpoint["classes"]
        if tuple(self.classes) != EXPECTED_CLASSES or checkpoint["num_classes"] != 37:
            raise ValueError("Oxford 37품종 모델의 클래스 정보가 예상과 다릅니다.")
        self.classifier = models.resnet50(weights=None)
        self.classifier.fc = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(2048, 37))
        self.classifier.load_state_dict(checkpoint["model_state"], strict=True)
        self.classifier.eval()
        # 제공자가 학습/추론 예제에 명시한 전처리와 동일합니다.
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.confidence = confidence

    def classify(self, crop, species):
        import torch
        tensor = self.transform(crop).unsqueeze(0)
        with torch.inference_mode():
            probabilities = self.classifier(tensor).softmax(dim=1)[0].tolist()
        return choose_breed(probabilities, self.classes, species, self.confidence)

# 3. 개·고양이 탐지
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

# 4. 탐지와 분류 연결
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
