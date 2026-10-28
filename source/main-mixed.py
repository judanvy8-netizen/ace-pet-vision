"""ACE 혼합 사진 실전 검증: JPEG 확인 → main-3 추론 → 수기 정답과 비교.

기존 main.py/main-1.py/main-2.py/main-3.py는 수정하지 않습니다.
실제 모델 추론을 수행하며, 미탐지 동물을 정답표로 보완하지 않습니다.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.truth_reader import read_ground_truth

INPUT = ROOT / "data" / "mixed_samples"
TRUTH = ROOT / "data" / "mixed_ground_truth.csv"
OUTPUT = ROOT / "outputs" / "mixed_samples"


def check_samples(input_dir=INPUT, truth_csv=TRUTH):
    """이미지 진짜 JPEG 여부, 파일명, 사람이 작성한 개·고양이 수를 확인합니다."""
    from PIL import Image

    input_dir, truth_csv = Path(input_dir), Path(truth_csv)
    files = sorted(input_dir.glob("*.jpg"))
    if not files:
        raise ValueError(f"혼합 JPG 사진이 없습니다: {input_dir}")
    truth = read_ground_truth(truth_csv)
    names = {path.name for path in files}
    if names != set(truth):
        missing = sorted(names - set(truth))
        extra = sorted(set(truth) - names)
        raise ValueError(f"JPG와 정답표 파일명이 다릅니다. 정답 누락: {missing}, 사진 누락: {extra}")
    if any(path.suffix.lower() == ".webp" for path in input_dir.iterdir() if path.is_file()):
        raise ValueError("혼합 폴더에 WEBP 파일이 남아 있습니다. JPG로 변환해 주세요.")
    for path in files:
        with Image.open(path) as image:
            if image.format != "JPEG":
                raise ValueError(f"확장자는 JPG이지만 실제 JPEG가 아닙니다: {path.name}")
            if min(image.size) < 40:
                raise ValueError(f"사진이 너무 작습니다: {path.name}, {image.size}")
            image.verify()
        actual = truth[path.name]
        if actual["actual_cat"] is None or actual["actual_dog"] is None:
            raise ValueError(f"정답 개수가 아직 작성되지 않았습니다: {path.name}")
    return files, truth


def _read_csv(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"추론 결과 CSV가 없습니다: {path}")
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def compare_predictions(files, truth, counts, breeds):
    """실제 모델이 센 개·고양이를 수기 정답과 비교하고 PASS/FAIL/REVIEW로 표시합니다."""
    by_name = defaultdict(list)
    for row in counts:
        by_name[str(row.get("filename", ""))].append(row)
    breeds_by_name = defaultdict(list)
    for row in breeds:
        breeds_by_name[str(row.get("filename", ""))].append(row)
    results = []
    for path in files:
        name = path.name
        target = truth[name]
        exp_cat, exp_dog = target["actual_cat"], target["actual_dog"]
        matches = by_name.get(name, [])
        state, reason, pred_cat, pred_dog = "실패", "", "", ""
        status = "미생성"
        if len(matches) == 1:
            row = matches[0]
            status = row.get("status", "")
            try:
                pred_cat, pred_dog = int(row["cat"]), int(row["dog"])
                predicted = breeds_by_name.get(name, [])
                # 정답표로 동물을 만들어 내지 않습니다. 탐지 박스 좌표가 있는 실제 결과만 셉니다.
                def has_detection_box(pet):
                    if pet.get("reason") == "single_pet_assumed_not_yolo_detected":
                        return False
                    try:
                        x1, y1, x2, y2 = (float(pet[k]) for k in ("x1", "y1", "x2", "y2"))
                        return x1 < x2 and y1 < y2 and float(pet["detection_confidence"]) > 0
                    except (KeyError, ValueError, TypeError):
                        return False

                box_counts = Counter(b.get("species", "") for b in predicted if has_detection_box(b))
                boxes_match = box_counts["cat"] == pred_cat and box_counts["dog"] == pred_dog
                counts_match = pred_cat == exp_cat and pred_dog == exp_dog
                if not counts_match:
                    reason = f"개수 불일치 (고양이 {pred_cat}/{exp_cat}, 개 {pred_dog}/{exp_dog})"
                elif not boxes_match:
                    reason = "개수는 같지만 실제 박스별 결과가 부족합니다"
                elif status == "ok":
                    state, reason = "통과", "고양이·개를 각각 탐지"
                elif status.startswith("review"):
                    state, reason = "검토", "마릿수는 맞지만 모델 간 판단 충돌"
                else:
                    reason = f"처리 상태 확인 필요: {status}"
            except (ValueError, TypeError, KeyError):
                reason = "예측 CSV에 유효한 개수 정보가 없습니다"
        elif not matches:
            reason = "이 사진의 추론 결과가 없습니다"
        else:
            reason = "한 사진의 결과 행이 중복되었습니다"
        results.append({
            "파일명": name, "실제 고양이": exp_cat, "실제 개": exp_dog,
            "예측 고양이": pred_cat, "예측 개": pred_dog,
            "모델 상태": status, "판정": state, "설명": reason,
        })
    return results


def save_report(rows, output_dir, *, ran_model):
    """수기 정답 비교표와 통과·검토·실패 요약을 남깁니다."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / "혼합_테스트_결과.csv"
    columns = list(rows[0])
    with dest.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    stats = Counter(row["판정"] for row in rows)
    summary = [
        "# 개·고양이 혼합 JPG 테스트",
        "",
        f"- 검사 사진: **{len(rows)}장**",
        f"- 모두 올바르게 탐지: **{stats['통과']}장**",
        f"- 마릿수는 맞지만 검토 필요: **{stats['검토']}장**",
        f"- 미탐지·개수 오류·처리 실패: **{stats['실패']}장**",
        f"- 실행 종류: **{'이번에 실제 모델 추론 실행' if ran_model else '기존 결과만 다시 비교'}**",
        "- 각 사진에 **고양이 1마리와 개 1마리**가 있다는 수기 정답을 사용합니다.",
        "- 정답표를 AI 추론에 사용하지 않으며 결과가 맞지 않으면 통과라고 표시하지 않습니다.",
        "- 저해상도(약 117픽셀) 예제라 새로운 사진의 일반 성능을 대표하지 않습니다.",
        "",
        "## 사진별 결과",
        "",
        "| 파일명 | 고양이 예측/정답 | 개 예측/정답 | 판정 | 설명 |",
        "|---|---:|---:|---|---|",
    ]
    for row in rows:
        summary.append(
            f'| {row["파일명"]} | {row["예측 고양이"]}/{row["실제 고양이"]} | '
            f'{row["예측 개"]}/{row["실제 개"]} | {row["판정"]} | {row["설명"]} |'
        )
    (output_dir / "혼합_테스트_요약.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return stats


def run_model(input_dir, truth_csv, output_dir, conf):
    """기존 main-3.py를 수정하지 않고 동일 모델로 혼합 사진을 실제 추론합니다."""
    source = ROOT / "source" / "main-3.py"
    spec = importlib.util.spec_from_file_location("ace_mixed_main3", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("main-3.py를 불러올 수 없습니다")
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    return program.main([
        "--input", str(input_dir), "--truth", str(truth_csv),
        "--output", str(output_dir), "--limit", "0", "--conf", str(conf),
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(description="혼합 JPG 4장 AI 탐지·수기 정답 비교")
    parser.add_argument("--input", type=Path, default=INPUT, help="혼합 JPG 사진 폴더")
    parser.add_argument("--truth", type=Path, default=TRUTH, help="제목 없는 수기 정답표")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="출력 폴더")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO 1차 탐지 기준(0~1)")
    parser.add_argument("--check-files", action="store_true", help="모델 없이 JPEG·정답표만 검사")
    parser.add_argument("--report-only", action="store_true", help="모델을 실행하지 않고 저장된 결과 재검사")
    args = parser.parse_args(argv)
    if not 0 <= args.conf <= 1:
        parser.error("--conf는 0~1 사이여야 합니다")
    try:
        files, truth = check_samples(args.input, args.truth)
        print(f"혼합 JPG 확인 완료: {len(files)}장 / 사진당 개·고양이 각 1마리 수기 정답 준비")
        if args.check_files:
            return 0
        if not args.report_only:
            # 추론 실패 시 이전 결과를 새 실행 성공이라고 기록하지 않습니다.
            run_model(args.input, args.truth, args.output, args.conf)
        counts = _read_csv(args.output / "pet_counts.csv")
        breeds = _read_csv(args.output / "pet_breeds.csv")
        compared = compare_predictions(files, truth, counts, breeds)
        stats = save_report(compared, args.output, ran_model=not args.report_only)
        for row in compared:
            print(f'{row["파일명"]}: {row["판정"]} - {row["설명"]}')
        print(f"결과: {args.output / '혼합_테스트_요약.md'}")
        return 0 if stats["통과"] == len(files) else 1
    except (OSError, ValueError, RuntimeError) as error:
        print(f"혼합 테스트 실패: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
