"""AI 처리: 모델 다운로드 → 동물 탐지 → 크롭 → 37품종 분류."""
import hashlib
from pathlib import Path
import tempfile
import shutil
import urllib.request
from .fusion import choose_supported_breed

MODEL_DIR = Path(__file__).resolve().parent / "models"

DETECTION_CONFIDENCE = 0.25
BREED_CONFIDENCE = 0.50

# 1. 모델 파일 준비
BREED_REVISION = "3253ad292cf0778f5b64fa5a9156f2bdb0a84a3b"
BREED_URL = f"https://huggingface.co/flaviodell/oxford-pets-resnet50/resolve/{BREED_REVISION}/best_model.pth"
BREED_SHA256 = "889b2d4919ea0c48cb8a89e2b25ff7c56af1bd3e1627e36b36d63c495b3842d4"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt"
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"
# SHA-256 verified against Ultralytics/assets official v8.4.0 release metadata.
YOLO_S_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt"
YOLO_S_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
# Ultralytics/assets v8.4.0 official GitHub release digest.
YOLO_M_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11m.pt"
YOLO_M_SHA256 = "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95"
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


def model_file(filename, url, checksum):
    """기존 models 파일을 재사용하고, 모델은 shared/models에 모읍니다."""
    destination = MODEL_DIR / filename
    legacy = MODEL_DIR.parent.parent / "models" / filename
    if not destination.exists() and legacy.is_file() and sha256(legacy) == checksum:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, destination)
    return download_verified(url, destination, checksum)


def prepare_models():
    return (
        model_file("yolo11n.pt", YOLO_URL, YOLO_SHA256),
        model_file("oxford-pets-resnet50.pth", BREED_URL, BREED_SHA256),
        model_file("yolo11s.pt", YOLO_S_URL, YOLO_S_SHA256),
        model_file("yolo11m.pt", YOLO_M_URL, YOLO_M_SHA256),
    )


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
        # 배포 모델 설명서는 classes와 model_state를 명시하지만
        # num_classes 메타데이터는 보장하지 않습니다.
        if not isinstance(checkpoint, dict) or not {"classes", "model_state"}.issubset(checkpoint):
            raise ValueError("품종 모델 체크포인트에 classes 또는 model_state가 없습니다.")
        self.classes = tuple(checkpoint["classes"])
        if self.classes != EXPECTED_CLASSES or checkpoint.get("num_classes", len(self.classes)) != 37:
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

    def _probabilities(self, crop):
        import torch
        tensor = self.transform(crop).unsqueeze(0)
        with torch.inference_mode():
            return self.classifier(tensor).softmax(dim=1)[0].tolist()

    def classify(self, crop, species):
        return self.classify_with_context(crop, species)

    @staticmethod
    def _aspect_preserved_view(crop):
        """가로·세로가 긴 동물 사진을 정사각형으로 찌그러뜨리지 않고 채웁니다."""
        from PIL import Image, ImageOps
        # 사진 평균 색상으로 여백을 채워 인위적인 검은 테두리를 피합니다.
        fill = crop.convert("RGB").resize((1, 1)).getpixel((0, 0))
        return ImageOps.pad(crop.convert("RGB"), (224, 224),
                            method=Image.Resampling.BILINEAR, color=fill)

    def classify_with_context(self, crop, species, context=None):
        primary = self._probabilities(crop)
        outcome = choose_supported_breed(primary, self.classes, species, self.confidence, CAT_BREEDS)
        initial_top = outcome["resnet_top_breed"]
        initial_confidence = outcome["resnet_top_confidence"]
        outcome["tta_used"] = False
        outcome["variant_diagnostics"] = f"원본:{initial_top} {initial_confidence:.1%}"
        # 기존에 높은 신뢰도로 판단된 결과(16/19번 포함)는 훼손하지 않습니다.
        if outcome["reason"] not in ("low_confidence", "species_conflict"):
            outcome["initial_top_breed"] = initial_top
            outcome["initial_top_confidence"] = initial_confidence
            return outcome

        from PIL import Image
        variants = [("원본", primary),
                    ("좌우반전", self._probabilities(crop.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))]
        if context is not None and context.size != crop.size:
            variants.append(("주변영역", self._probabilities(context)))
        # 근접·가로 사진(예: 500x231 치와와)을 224x224로 왜곡하지 않는 관점도 검사합니다.
        if min(crop.size) and max(crop.size) / min(crop.size) >= 1.4:
            variants.append(("비율유지", self._probabilities(self._aspect_preserved_view(crop))))

        averaged = [sum(probs[i] for _, probs in variants) / len(variants)
                    for i in range(len(primary))]
        outcome = choose_supported_breed(averaged, self.classes, species, self.confidence, CAT_BREEDS)
        outcome["tta_used"] = len(variants) > 1
        outcome["initial_top_breed"] = initial_top
        outcome["initial_top_confidence"] = initial_confidence
        def variant_summary(name, probs):
            idx = max(range(len(probs)), key=lambda i: probs[i])
            return f"{name}:{self.classes[idx]} {probs[idx]:.1%}"
        outcome["variant_diagnostics"] = " | ".join(
            variant_summary(name, probs) for name, probs in variants)
        return outcome

    def classify_single_pet(self, crop):
        """Breed-only hypothesis for a photo the USER says has exactly one pet.

        This is closed-set image classification, not an object-detection result.
        Therefore the prediction is always marked for human review.
        """
        probabilities = self._probabilities(crop)
        best = max(range(len(self.classes)), key=lambda i: probabilities[i])
        species = "cat" if self.classes[best] in CAT_BREEDS else "dog"
        outcome = choose_supported_breed(probabilities, self.classes, species,
                                         self.confidence, CAT_BREEDS)
        outcome.update(breed="unknown", status="review",
                       reason="single_pet_assumed_not_yolo_detected", tta_used=False)
        return {"species": species, **outcome}

