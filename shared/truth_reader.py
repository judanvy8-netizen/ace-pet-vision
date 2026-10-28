"""수기 정답표 읽기: 제목 없는 5열 한국어 CSV와 이전 제목 포함 CSV를 지원합니다.

새 형식(제목 행 없음): 파일명,동물 종류,고양이 수,개 수,장면 설명
기존 형식(제목 행 있음): filename,actual_cat,actual_dog[,scene,note]
"""
from pathlib import Path
import csv


def read_ground_truth(path):
    path = Path(path)
    if not path.is_file():
        return {}

    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [(line_no, [cell.strip() for cell in row])
                for line_no, row in enumerate(csv.reader(handle), 1)
                if any(cell.strip() for cell in row)]

    if not rows:
        return {}

    header = rows[0][1]
    has_header = header[0] == "filename"
    if has_header:
        if not {"filename", "actual_cat", "actual_dog"}.issubset(header):
            raise ValueError("정답표 제목에 filename, actual_cat, actual_dog가 필요합니다.")
        filename_col = header.index("filename")
        cat_col = header.index("actual_cat")
        dog_col = header.index("actual_dog")
        kind_col = header.index("species") if "species" in header else None
        data = rows[1:]
    else:
        # 사람이 읽기 쉬운 제목 없는 5열: img_001.jpg,고양이,1,0,고양이 한 마리
        filename_col, kind_col, cat_col, dog_col = 0, 1, 2, 3
        data = rows

    truth = {}
    for line_no, row in data:
        required = max(filename_col, cat_col, dog_col,
                       kind_col if kind_col is not None else 0) + 1
        if len(row) < required or (not has_header and len(row) != 5):
            raise ValueError(f"정답표 {line_no}행의 열 개수가 올바르지 않습니다.")
        name = row[filename_col].replace("\\", "/")
        if not name or name in truth:
            raise ValueError(f"정답표 파일명이 비었거나 중복입니다: {name}")

        values = {}
        for key, column in (("actual_cat", cat_col), ("actual_dog", dog_col)):
            value = row[column]
            if value and not (value.isascii() and value.isdigit()):
                raise ValueError(f"정답표 {name}: {key}는 0 이상의 정수 또는 빈 칸이어야 합니다.")
            values[key] = int(value) if value else None

        if kind_col is not None:
            kind = row[kind_col]
            known = {
                "고양이": lambda c, d: c > 0 and d == 0,
                "개": lambda c, d: d > 0 and c == 0,
                "개와 고양이": lambda c, d: c > 0 and d > 0,
                "대상 없음": lambda c, d: c == 0 and d == 0,
            }
            if kind not in known:
                raise ValueError(f"정답표 {name}: 동물 종류는 고양이/개/개와 고양이/대상 없음 중 하나여야 합니다.")
            cat_count, dog_count = values["actual_cat"], values["actual_dog"]
            if cat_count is not None and dog_count is not None and not known[kind](cat_count, dog_count):
                raise ValueError(f"정답표 {name}: 동물 종류와 실제 개·고양이 수가 일치하지 않습니다.")

        truth[name] = values
    return truth
