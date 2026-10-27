"""Oxford-IIIT Pet 공식 images + annotations 폴더 읽기."""
import re
from pathlib import Path
from pet_model import EXPECTED_CLASSES


def load_split(dataset_dir, split="test", limit=20):
    dataset_dir = Path(dataset_dir)
    images = dataset_dir / "images"
    annotations = dataset_dir / "annotations" / f"{split}.txt"
    if not images.is_dir() or not annotations.is_file():
        raise ValueError(
            f"Oxford 데이터셋이 아직 준비되지 않았습니다.\n"
            f"images.tar.gz와 annotations.tar.gz를 풀어 다음 위치에 놓으세요:\n"
            f"  {images}\n  {annotations}\n"
            "먼저 기존 예제로 확인하려면 실행 옵션에 --samples를 넣으세요."
        )
    names = {name.lower().replace(" ", "_"): name for name in EXPECTED_CLASSES}
    truth, files, seen = {}, [], set()
    for line_number, line in enumerate(annotations.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 4 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*_[0-9]+", parts[0]):
            raise ValueError(f"정답 파일 형식 오류: {annotations.name}:{line_number}")
        image_id, class_id, species, breed_id = parts
        try:
            if not (1 <= int(class_id) <= 37 and species in {"1", "2"} and 1 <= int(breed_id) <= 25):
                raise ValueError()
        except ValueError:
            raise ValueError(f"정답 라벨 오류: {annotations.name}:{line_number}") from None
        breed = names.get(image_id.rsplit("_", 1)[0].lower())
        if breed is None or image_id in seen:
            raise ValueError(f"알 수 없거나 중복된 이미지 ID: {image_id}")
        seen.add(image_id)
        filename = image_id + ".jpg"
        files.append(images / filename)
        truth[filename] = {"breed": breed, "species": "cat" if species == "1" else "dog"}
    total = len(files)
    if not total:
        raise ValueError("데이터셋 정답 파일에 사진 목록이 없습니다.")
    selected = files[:limit] if limit else files
    missing = [p.name for p in selected if not p.is_file()]
    if missing:
        raise ValueError(f"선택된 데이터셋 사진 {len(missing)}장이 없습니다. 압축 해제 위치를 확인하세요: {missing[:3]}")
    return images, selected, {p.name: truth[p.name] for p in selected}, total


def compare_predictions(truth, results, summary):
    by_file = {}
    for row in results:
        by_file.setdefault(row["filename"], []).append(row)
    rows = []
    for image in summary:
        name = image["filename"]
        candidates = by_file.get(name, [])
        # 정답을 보지 않고 가장 큰 탐지 박스를 대표 개체로 선택합니다.
        selected = max(candidates, key=lambda r: (r["x2"]-r["x1"])*(r["y2"]-r["y1"])) if candidates else None
        actual = truth[name]
        matched = bool(selected and selected["breed"] == actual["breed"] and selected["species"] == actual["species"])
        status = ("error" if image["status"] == "error" else "no_detection") if selected is None else ("match" if matched else "mismatch")
        rows.append({"filename": name, "expected_species": actual["species"], "expected_breed": actual["breed"],
                     "predicted_breed": selected["breed"] if selected else "", "comparison": status,
                     "animal_count": len(candidates), "multiple_animals_review": len(candidates) > 1})
    return rows
