"""예제 1: Oxford 데이터셋의 사진 경로와 정답 품종 5개 확인."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from oxford_data import load_split

if __name__ == "__main__":
    try:
        image_dir, photos, answers, total = load_split(ROOT / "data" / "oxford-iiit-pet", limit=5)
        print(f"test 목록 총 {total}장 중 처음 {len(photos)}장")
        for photo in photos:
            print(photo.name, "->", answers[photo.name]["breed"])
    except ValueError as error:
        print(error)
        raise SystemExit(1)
