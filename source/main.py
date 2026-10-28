"""05: 반려동물 종류(개/고양이)를 사진에서 탐지."""
from pathlib import Path
import sys
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from shared.common import yolo, image_files, read_image, save_image, save_csv

TARGETS = ("cat", "dog")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/images"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    model = yolo()
    ids = [i for i, name in model.names.items() if name in TARGETS]
    rows = []
    for path in image_files(args.input):
        result = model.predict(read_image(path), classes=ids, conf=args.conf, device="cpu", verbose=False)[0]
        counts = {name: 0 for name in TARGETS}
        for box in result.boxes:
            counts[result.names[int(box.cls[0].item())]] += 1
        save_image(ROOT / "outputs" / f"{path.stem}_pets.jpg", result.plot())
        rows.append({"filename": path.name, **counts})
        print(path.name, counts)
    save_csv(ROOT / "outputs/pet_counts.csv", ["filename", *TARGETS], rows)

if __name__ == "__main__":
    main()