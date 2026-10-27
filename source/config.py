"""공통 설정: 폴더 경로와 기본 실행값을 한곳에서 관리합니다.
실습: DETECTION_CONFIDENCE나 BREED_CONFIDENCE를 바꾸고 결과를 비교하세요.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = ROOT / "data" / "images"
DATASET_DIR = ROOT / "data" / "oxford-iiit-pet"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
DETECTION_CONFIDENCE = 0.25
BREED_CONFIDENCE = 0.50
IMAGE_LIMIT = 20
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
