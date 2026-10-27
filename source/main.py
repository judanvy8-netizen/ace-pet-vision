"""ACE 시작 파일: 현재는 입력 사진과 프로젝트 경로를 확인합니다."""
from pathlib import Path
import argparse
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_DIR = PROJECT_ROOT / "input" / "images"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def find_images(folder):
    """하위 폴더까지 검색합니다. 실행 버튼의 작업 폴더 설정에 의존하지 않습니다."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"입력 폴더가 없습니다: {folder}")
    return sorted(path for path in folder.rglob("*")
                  if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def main():
    parser = argparse.ArgumentParser(description="ACE 프로젝트 입력 사진 확인")
    parser.add_argument("--input", type=Path, default=INPUT_DIR,
                        help="선택 사항: 확인할 다른 사진 폴더")
    args = parser.parse_args()
    # 직접 지정한 상대 경로도 프로젝트 루트를 기준으로 해석합니다.
    folder = args.input if args.input.is_absolute() else PROJECT_ROOT / args.input
    files = find_images(folder)
    print(f"Python: {sys.version.split()[0]}")
    print(f"프로젝트: {PROJECT_ROOT}")
    print(f"입력 폴더: {folder}")
    print(f"결과 폴더: {OUTPUT_DIR}")
    print(f"모델 폴더: {MODEL_DIR}")
    print(f"입력 사진: {len(files)}장")
    for index, path in enumerate(files, 1):
        print(f"  {index:02d}. {path.relative_to(folder)}")
    if not files:
        print("input/images 폴더에 사진을 넣어 주세요.")
        return 1
    print("입력 경로 확인 완료. 탐지·품종 분류·결과 저장 기능은 다음 구현 단계입니다.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError) as error:
        print(f"입력 확인 실패: {error}", file=sys.stderr)
        raise SystemExit(1)
