"""모델 준비: 고정된 버전의 가중치를 다운로드하고 SHA256으로 검증합니다.
설명할 함수: download_verified, prepare_models. 모델 파일은 Git에 올리지 않습니다.
"""
import hashlib
from pathlib import Path
import tempfile
import urllib.request

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
