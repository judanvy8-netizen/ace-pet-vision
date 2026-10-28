"""ACE 실행 결과 검증: 수작업 정답, REVIEW, 출처 라벨을 분리합니다."""
from collections import Counter, defaultdict
from pathlib import Path
import csv
import re

DETAIL_FIELDS = [
    "filename", "status", "detector", "pred_cat", "pred_dog",
    "actual_cat", "actual_dog", "count_check", "count_absolute_error",
    "source_species", "source_breed", "candidate_breed", "global_top_breed",
    "candidate_vs_source", "global_top_vs_source", "review_reason", "source_url",
    "final_count_check", "final_count_absolute_error",
]


def _normalize_breed(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def _source_labels(path):
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"filename", "label", "species"}.issubset(reader.fieldnames or []):
            raise ValueError("metadata.csv에 filename, label, species 열이 필요합니다.")
        return {row["filename"].strip().replace("\\", "/"): row
                for row in reader if (row.get("filename") or "").strip()}


def _as_optional_int(value):
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError("개수는 정수여야 합니다.")
    result = int(value)
    if result < 0:
        raise ValueError("개수는 0 이상이어야 합니다.")
    return result


def summarize_evaluation(count_rows, breed_rows, ground_truth, metadata_path):
    """Count metrics require confirmed truth; source labels are ONLY references."""
    metadata = _source_labels(metadata_path)
    breeds_by_file = defaultdict(list)
    for pet in breed_rows:
        breeds_by_file[pet["filename"]].append(pet)
    diagnostics, stats = [], Counter()
    for row in count_rows:
        filename = row["filename"]
        status = str(row.get("status") or "error")
        truth = ground_truth.get(filename) or {}
        actual_cat = _as_optional_int(truth.get("actual_cat"))
        actual_dog = _as_optional_int(truth.get("actual_dog"))
        pred_cat = _as_optional_int(row.get("cat"))
        pred_dog = _as_optional_int(row.get("dog"))
        known = actual_cat is not None and actual_dog is not None
        stats["processed"] += 1
        if known:
            stats["truth_confirmed"] += 1
        stats[f"status_{status}"] += 1
        if status.startswith("review"):
            stats["review"] += 1

        count_error = ""
        if status in ("error", "partial_error"):
            check = "처리 오류(평가 제외)"
        elif status.startswith("review"):
            check = "모델 판단 충돌(평가 제외)"
        elif not known:
            check = "수작업 정답 미작성"
        elif pred_cat is None or pred_dog is None:
            check = "예측 개수 없음(평가 제외)"
        else:
            stats["count_scored"] += 1
            count_error = abs(pred_cat - actual_cat) + abs(pred_dog - actual_dog)
            if count_error == 0:
                stats["count_exact"] += 1
                check = "정답 개수 일치"
            else:
                stats["count_wrong"] += 1
                check = "정답 개수 불일치"

        # The final pipeline count is a separate metric from pure YOLO detection.
        # A genuine YOLO box reclassified via ResNet can be evaluated here, but
        # the explicit human-assumed "review_only" mode must NOT inflate results.
        final_error = ""
        if status in ("ok", "no_detection", "review_detection", "review_species") and known \
                and pred_cat is not None and pred_dog is not None:
            stats["final_count_scored"] += 1
            final_error = abs(pred_cat - actual_cat) + abs(pred_dog - actual_dog)
            if final_error == 0:
                stats["final_count_exact"] += 1
                final_check = ("최종 개수 일치(REVIEW 포함)" if status.startswith("review")
                               else "최종 개수 일치")
            else:
                stats["final_count_wrong"] += 1
                final_check = "최종 개수 불일치"
        elif not known:
            final_check = "수작업 정답 미작성"
        else:
            final_check = "최종 개수 평가 제외"

        source = metadata.get(filename, {})
        source_breed = (source.get("label") or "").strip()
        source_species = (source.get("species") or "").strip()
        found = breeds_by_file.get(filename, [])
        candidate = global_top = reason = ""
        candidate_match = global_match = "비교 불가"
        if len(found) == 1:
            pet = found[0]
            candidate = str(pet.get("candidate_breed") or "")
            global_top = str(pet.get("resnet_top_breed") or candidate)
            reason = str(pet.get("reason") or "")
            if source_breed:
                stats["source_comparable"] += 1
                candidate_match = ("출처 표기 일치"
                    if _normalize_breed(candidate) == _normalize_breed(source_breed)
                    else "출처 표기 불일치")
                global_match = ("출처 표기 일치"
                    if _normalize_breed(global_top) == _normalize_breed(source_breed)
                    else "출처 표기 불일치")
                stats["source_candidate_match"] += candidate_match == "출처 표기 일치"
                stats["source_global_match"] += global_match == "출처 표기 일치"
        elif len(found) > 1:
            candidate = "복수 개체—개별 검증 필요"
        diagnostics.append({
            "filename": filename, "status": status, "detector": row.get("detector", ""),
            "pred_cat": "" if pred_cat is None else pred_cat,
            "pred_dog": "" if pred_dog is None else pred_dog,
            "actual_cat": "" if actual_cat is None else actual_cat,
            "actual_dog": "" if actual_dog is None else actual_dog,
            "count_check": check, "count_absolute_error": count_error,
            "source_species": source_species, "source_breed": source_breed,
            "candidate_breed": candidate, "global_top_breed": global_top,
            "candidate_vs_source": candidate_match, "global_top_vs_source": global_match,
            "review_reason": reason, "source_url": row.get("source_url", source.get("source_page", "")),
            "final_count_check": final_check, "final_count_absolute_error": final_error,
        })
    return diagnostics, stats


def _write_csv(path, rows):
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DETAIL_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_evaluation(count_rows, breed_rows, ground_truth, metadata_path, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, stats = summarize_evaluation(count_rows, breed_rows, ground_truth, metadata_path)
    _write_csv(output_dir / "검증_상세.csv", rows)
    total = stats["processed"]
    scored = stats["count_scored"]
    matched = stats["count_exact"]
    source_scored = stats["source_comparable"]
    source_match = stats["source_candidate_match"]
    count_line = (f"{matched}/{scored}장 ({matched / scored:.1%})" if scored else
                  "계산 보류 — 비교 가능한 수작업 정답·정상 탐지가 없음")
    source_line = (f"{source_match}/{source_scored}장" if source_scored else
                   "비교 가능한 사진 없음")
    final_scored = stats["final_count_scored"]
    final_exact = stats["final_count_exact"]
    final_line = (f"{final_exact}/{final_scored}장 ({final_exact / final_scored:.1%})"
                  if final_scored else "계산 보류 — 수작업 정답과 최종 종별 개수 비교 불가")
    unconfirmed = [r["filename"] for r in rows if r["count_check"] == "수작업 정답 미작성"]
    review = [r["filename"] for r in rows if r["status"].startswith("review")]
    lines = [
        "# ACE Pet Vision — 사진 검증 요약 (실행 시 자동 생성)", "",
        f"- 이번 실행에서 처리한 사진: **{total}장**",
        f"- 수작업 개수 정답이 작성된 사진: **{stats['truth_confirmed']}/{total}장**",
        f"- 정상 탐지 기준 사진별 cat/dog 개수 일치(REVIEW 제외): **{count_line}**",
        f"- 교차 모델 최종 개수 일치(REVIEW 2종 포함, review_only 제외): **{final_line}**",
        f"- REVIEW 결과(정상 탐지 정확도 계산 제외): **{stats['review']}장**",
        f"- 원본 출처 설명의 품종 표기와 후보 일치: **{source_line}** — 정확도가 아닌 참고 지표",
        "", "## 평가 기준", "",
        "1. data/ground_truth.csv에서 사람이 사진을 확인한 cat/dog 개수만 정답으로 취급합니다. 빈칸은 0이 아닙니다.",
        "2. 정답이 양쪽 모두 작성되고 상태가 ok 또는 no_detection인 경우만 개수 정확도 분모에 포함합니다.",
        "3. review_detection·review_species는 최종 교차 모델 개수 평가에는 넣되 일반 탐지 정확도에서는 제외합니다.",
        "4. review_only는 YOLO의 실제 검출 박스가 없으므로 두 정확도 지표에서 모두 제외합니다.",
        "5. metadata.csv의 품종은 출처 설명에서 가져온 참고 표기이며 독립적인 혈통 검증이 아닙니다.",
        "6. 품종 예측의 %는 모델 점수이지 실제 정확도나 검증 통과율이 아닙니다.",
        "7. 복수 동물·동물 없음·새로운 사진은 별도 확인이 필요합니다. 누락된 결과를 임의로 만들지 않습니다.",
        "", "## 후속 확인 목록", "",
        "- 수작업 정답 미작성: " + (", ".join(unconfirmed) if unconfirmed else "없음"),
        "- 모델 검토 필요: " + (", ".join(review) if review else "없음"),
        "- 상세 파일: outputs/검증_상세.csv", "",
    ]
    (output_dir / "검증_요약.md").write_text("\n".join(lines), encoding="utf-8")
    return stats
