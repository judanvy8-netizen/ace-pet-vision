"""예제 2: YOLO11n으로 개·고양이만 탐지하고 박스 이미지 저장."""
from pathlib import Path
import os
import sys
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from pet_model import download_verified, YOLO_URL, YOLO_SHA256

if __name__ == "__main__":
    # 다른 사진으로 실습하려면 이 파일명을 바꾸세요.
    photo_path = ROOT / "input" / "images" / "beagle.jpg"
    weights = download_verified(YOLO_URL, ROOT / "models" / "yolo11n.pt", YOLO_SHA256)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "models" / "ultralytics_settings"))
    from ultralytics import YOLO
    model = YOLO(str(weights))
    with Image.open(photo_path) as raw:
        photo = ImageOps.exif_transpose(raw).convert("RGB")
    result = model.predict(photo, classes=[15, 16], conf=0.25, device="cpu", verbose=False)[0]
    for box in result.boxes:
        print(model.names[int(box.cls.item())], box.xyxy[0].tolist(), f"{box.conf.item():.1%}")
    output = ROOT / "output" / "examples" / "detection.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result.plot()[:, :, ::-1]).save(output)
    print("탐지 이미지 저장:", output)
