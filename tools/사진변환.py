"""원래 PNG인 05·07번을 JPG로 변환하고 사진 메타데이터를 고칩니다."""
from pathlib import Path
from PIL import Image
import csv
import hashlib

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "data" / "images"
META = ROOT / "data" / "metadata.csv"

def convert_photos():
    changed = []
    for n in (5, 7):
        src = IMAGES / f"img_{n:03d}.png"
        dst = IMAGES / f"img_{n:03d}.jpg"
        if not src.exists():
            continue
        if dst.exists():
            raise FileExistsError(str(dst))
        with Image.open(src) as image:
            if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                rgba = image.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                background.alpha_composite(rgba)
                rgb = background.convert("RGB")
            else:
                rgb = image.convert("RGB")
            rgb.save(dst, "JPEG", quality=92, optimize=True)
        with Image.open(dst) as verify:
            if verify.format != "JPEG":
                raise ValueError("JPG 변환 실패: " + str(dst))
        src.unlink()
        changed.append(dst)

    if changed:
        with META.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            fields = reader.fieldnames
            rows = list(reader)
        for row in rows:
            target = IMAGES / row["filename"]
            if target not in changed:
                continue
            raw = target.read_bytes()
            with Image.open(target) as img:
                row["width"], row["height"] = str(img.width), str(img.height)
            row["bytes"] = str(len(raw))
            row["sha256"] = hashlib.sha256(raw).hexdigest()
            row["modifications"] = (row.get("modifications", "") + "; PNG to JPEG quality 92").strip("; ")
        temp = META.with_suffix(".tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(META)
    print("변환 완료: " + (", ".join(x.name for x in changed) if changed else "이미 JPG"))

if __name__ == "__main__":
    convert_photos()
