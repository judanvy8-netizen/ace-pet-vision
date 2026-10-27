"""입력 담당: 사진 목록을 찾고 EXIF 방향을 반영해 RGB 사진으로 읽습니다.
설명할 함수: find_images, read_image. 원본 사진은 변경하지 않습니다.
"""
from config import ROOT, IMAGE_EXTENSIONS

def project_path(path):
    return (path if path.is_absolute() else ROOT / path).resolve()


def find_images(folder):
    if folder.is_file():
        return [folder] if folder.suffix.lower() in IMAGE_EXTENSIONS else []
    if not folder.is_dir():
        raise ValueError(f"입력 폴더가 없습니다: {folder}")
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)



def read_image(path):
    from PIL import Image, ImageOps
    with Image.open(path) as raw:
        return ImageOps.exif_transpose(raw).convert("RGB")
