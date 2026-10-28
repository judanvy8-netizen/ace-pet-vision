# [설명] 최초 실행 정리: PNG 원본 05·07번을 실제 JPEG로 바꾸고 파일 정보·해시를 갱신합니다.
"""교사 파일명 img_###.jpg 준수: 두 PNG 내용을 실제 JPEG로 바꿉니다.

다른 파일은 건드리지 않습니다. 이미 변환된 JPEG라면 아무 일도 하지 않습니다.
변환 뒤 metadata.csv의 실제 파일 길이·SHA256·변경 사유를 갱신합니다.
"""
from pathlib import Path
from PIL import Image
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
META = ROOT / "data" / "metadata.csv"

# [설명] 확장자만 JPG였던 사진을 변환하며, 이미 JPEG인 사진은 다시 변환하지 않습니다.
def convert_photos():
    changed = []
    for number in (5, 7):
        jpg = IMAGES / f"img_{number:03d}.jpg"
        png = IMAGES / f"img_{number:03d}.png"
        source = png if png.exists() else jpg
        if not source.exists():
            raise FileNotFoundError(f"샘플 사진이 없습니다: {source}")
        with Image.open(source) as original:
            if original.format == "JPEG" and source == jpg:
                continue
            if original.format != "PNG":
                raise ValueError(f"예상과 다른 사진 형식: {source}")
            if original.mode in ("RGBA", "LA") or "transparency" in original.info:
                rgba = original.convert("RGBA")
                white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                white.alpha_composite(rgba)
                converted = white.convert("RGB")
            else:
                converted = original.convert("RGB")
            tmp = jpg.with_suffix(".temp.jpg")
            converted.save(tmp, format="JPEG", quality=92, optimize=True)
        with Image.open(tmp) as verified:
            if verified.format != "JPEG":
                raise ValueError(f"JPG 저장 확인 실패: {tmp}")
        tmp.replace(jpg)
        if source == png:
            png.unlink()
        changed.append(jpg)

    if changed:
        with META.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            target = IMAGES / row["filename"]
            if target not in changed:
                continue
            content = target.read_bytes()
            with Image.open(target) as photo:
                row["width"] = str(photo.width)
                row["height"] = str(photo.height)
            row["bytes"] = str(len(content))
            row["sha256"] = hashlib.sha256(content).hexdigest()
            row["modifications"] = (row.get("modifications", "") + "; PNG→JPEG (quality 92)").strip("; ")
        tmp = META.with_suffix(".temp.csv")
        with tmp.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        tmp.replace(META)
    print("사진 JPG 확인: " + (", ".join(x.name for x in changed) if changed else "이미 완료"))

if __name__ == "__main__":
    convert_photos()
