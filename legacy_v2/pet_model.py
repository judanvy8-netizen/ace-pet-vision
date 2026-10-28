# [설명] main-2 전용 과거 구현: ResNet50 모델 로딩과 37품종 분류. 단계별 변화 비교에 쓰며 최신 공통 코드를 대신하지 않습니다.
"""AI 처리: 모델 다운로드 → 동물 탐지 → 크롭 → 37품종 분류."""
import hashlib
from pathlib import Path
import tempfile
import shutil
import urllib.request

# 이전 단계도 현재 프로젝트와 같은 검증된 모델 가중치 파일을 사용합니다.
MODEL_DIR = Path(__file__).resolve().parents[1] / "shared" / "models"

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
# [설명] 다운로드된 모델 가중치의 SHA-256으로 무결성을 검사합니다.
def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# [설명] 해시 검증을 통과한 가중치만 보관해 파일 손상을 막습니다.
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


# [설명] 사전학습 모델을 저장하거나 기존 파일을 재사용합니다.
def model_file(filename, url, checksum):
    """기존 models 파일을 재사용하고, 모델은 shared/models에 모읍니다."""
    destination = MODEL_DIR / filename
    legacy = MODEL_DIR.parent.parent / "models" / filename
    if not destination.exists() and legacy.is_file() and sha256(legacy) == checksum:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(legacy, destination)
    return download_verified(url, destination, checksum)


# [설명] 실행에 필요한 모델을 미리 다운로드해 확인합니다.
def prepare_models():
    return (
        model_file("yolo11n.pt", YOLO_URL, YOLO_SHA256),
        model_file("oxford-pets-resnet50.pth", BREED_URL, BREED_SHA256),
        model_file("yolo11s.pt", YOLO_S_URL, YOLO_S_SHA256),
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


# [설명] 이전 품종 분류 방식에서 낮은 점수·종 불일치를 검토 상태로 표시합니다.
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



# [설명] YOLO에서 자른 이미지를 Oxford 37품종으로 분류합니다.
class BreedClassifier:
    # [설명] ResNet50 체크포인트·품종 순서를 검사하고 추론 준비를 합니다.
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

    # [설명] 동물 영역을 전처리해 품종 후보를 계산합니다.
    def classify(self, crop, species):
        import torch
        tensor = self.transform(crop).unsqueeze(0)
        with torch.inference_mode():
            probabilities = self.classifier(tensor).softmax(dim=1)[0].tolist()
        return choose_breed(probabilities, self.classes, species, self.confidence)

