"""예제 3: 사진 한 장을 탐지·크롭한 뒤 품종 상위 후보 3개 출력."""
from pathlib import Path
import sys
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "source"))
from pet_model import PetModel

if __name__ == "__main__":
    # 다른 사진으로 실습하려면 이 파일명을 바꾸세요.
    photo_path = ROOT / "input" / "images" / "beagle.jpg"
    model = PetModel(ROOT / "models")
    with Image.open(photo_path) as raw:
        photo = ImageOps.exif_transpose(raw).convert("RGB")
    animals = model.predict(photo)
    if not animals:
        print("개·고양이를 찾지 못했습니다.")
    for number, animal in enumerate(animals, 1):
        print(f"동물 {number}: {animal['species']} / {animal['breed']} / {animal['status']}")
        for candidate in animal["top3"]:
            print(f"  {candidate['breed']}: {candidate['score']:.1%}")
        print("점수는 모델의 후보 점수이며 실제 정확도와 같지 않습니다.")
