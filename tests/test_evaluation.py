# [설명] 정답 비교 규칙 테스트: REVIEW와 정상 탐지의 숫자를 섞지 않고 저장하는지 확인합니다.
import csv
import tempfile
import unittest
from pathlib import Path
from shared.evaluation import summarize_evaluation, write_evaluation


# [설명] 수기 정답의 빈칸 처리·오차 계산·정상 및 REVIEW 지표 분리를 검사합니다.
class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.metadata = self.root / "metadata.csv"
        self.metadata.write_text(
            "filename,label,species,source_page\n"
            "cat.jpg,sphynx,cat,https://commons.wikimedia.org/test\n"
            "dog.jpg,chihuahua,dog,https://commons.wikimedia.org/test2\n"
            "miss.jpg,samoyed,dog,https://commons.wikimedia.org/test3\n",
            encoding="utf-8-sig",
        )

    def test_blank_truth_does_not_become_zero(self):
        counts = [dict(filename="cat.jpg", cat=0, dog=0, status="no_detection", detector="yolo11m")]
        detail, stats = summarize_evaluation(
            counts, [], {"cat.jpg": {"actual_cat": None, "actual_dog": None}}, self.metadata)
        self.assertEqual(detail[0]["count_check"], "수작업 정답 미작성")
        self.assertEqual(stats["count_scored"], 0)

    def test_review_is_excluded_from_strict_accuracy(self):
        counts = [dict(filename="cat.jpg", cat=1, dog=0, status="review_species", detector="yolo11n")]
        breeds = [dict(filename="cat.jpg", species="cat", candidate_breed="Sphynx",
                       resnet_top_breed="Sphynx", reason="species_corrected_high_confidence")]
        truth = {"cat.jpg": {"actual_cat": 1, "actual_dog": 0}}
        detail, stats = summarize_evaluation(counts, breeds, truth, self.metadata)
        self.assertEqual(detail[0]["count_check"], "모델 판단 충돌(평가 제외)")
        self.assertEqual(stats["count_scored"], 0)
        self.assertEqual(stats["final_count_scored"], 1)
        self.assertEqual(stats["final_count_exact"], 1)
        self.assertEqual(detail[0]["final_count_check"], "최종 개수 일치(REVIEW 포함)")
        self.assertEqual(stats["source_candidate_match"], 1)

    def test_count_error_and_source_label_agreement_are_separate(self):
        counts = [
            dict(filename="dog.jpg", cat=0, dog=1, status="ok", detector="yolo11n"),
            dict(filename="miss.jpg", cat=0, dog=0, status="no_detection", detector="yolo11m"),
        ]
        breeds = [dict(filename="dog.jpg", species="dog", candidate_breed="Chihuahua",
                       resnet_top_breed="Bengal")]
        truth = {"dog.jpg": {"actual_cat": 0, "actual_dog": 1},
                 "miss.jpg": {"actual_cat": 0, "actual_dog": 1}}
        detail, stats = summarize_evaluation(counts, breeds, truth, self.metadata)
        self.assertEqual(stats["count_scored"], 2)
        self.assertEqual(stats["count_exact"], 1)
        self.assertEqual(detail[0]["candidate_vs_source"], "출처 표기 일치")
        self.assertEqual(detail[0]["global_top_vs_source"], "출처 표기 불일치")
        self.assertEqual(detail[1]["count_absolute_error"], 1)

    def test_report_created_without_fabricated_accuracy(self):
        count = [dict(filename="cat.jpg", cat=1, dog=0, status="ok", detector="yolo11n")]
        stat = write_evaluation(count, [], {}, self.metadata, self.root / "outputs")
        self.assertEqual(stat["count_scored"], 0)
        with (self.root / "outputs" / "검증_상세.csv").open(encoding="utf-8-sig") as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 1)
        summary = (self.root / "outputs" / "검증_요약.md").read_text(encoding="utf-8")
        self.assertIn("계산 보류", summary)
        self.assertNotIn("100.0%", summary)

    def test_confirmed_review_and_normal_counts_are_separate(self):
        counts = [
            dict(filename="dog.jpg", cat=0, dog=1, status="ok", detector="yolo11n"),
            dict(filename="cat.jpg", cat=1, dog=0, status="review_species",
                 detector="yolo11s_tta"),
            dict(filename="miss.jpg", cat=0, dog=1, status="review_detection",
                 detector="yolo11m"),
        ]
        truth = {
            "dog.jpg": {"actual_cat": 0, "actual_dog": 1},
            "cat.jpg": {"actual_cat": 1, "actual_dog": 0},
            "miss.jpg": {"actual_cat": 0, "actual_dog": 1},
        }
        rows, stats = summarize_evaluation(counts, [], truth, self.metadata)
        self.assertEqual(stats["truth_confirmed"], 3)
        self.assertEqual((stats["count_scored"], stats["count_exact"]), (1, 1))
        self.assertEqual((stats["final_count_scored"], stats["final_count_exact"]), (3, 3))
        self.assertEqual(stats["review"], 2)
        self.assertIn("REVIEW", rows[1]["final_count_check"])
        self.assertIn("REVIEW", rows[2]["final_count_check"])

    def test_review_only_cannot_inflate_final_counts(self):
        rows, stats = summarize_evaluation(
            [dict(filename="cat.jpg", cat=1, dog=0, status="review_only",
                  detector="resnet_single_pet_review")], [],
            {"cat.jpg": {"actual_cat": 1, "actual_dog": 0}}, self.metadata)
        self.assertEqual(stats["truth_confirmed"], 1)
        self.assertEqual(stats["count_scored"], 0)
        self.assertEqual(stats["final_count_scored"], 0)
        self.assertEqual(rows[0]["final_count_check"], "최종 개수 평가 제외")

    # [설명] 엄격한 탐지 개수와 교차 판정 개수 일치가 각각 보고되는지 확인합니다.
    def test_report_shows_strict_and_review_aware_metrics(self):
        counts = [
            dict(filename="dog.jpg", cat=0, dog=1, status="ok", detector="yolo11n"),
            dict(filename="cat.jpg", cat=1, dog=0, status="review_species",
                 detector="yolo11s_tta"),
        ]
        truth = {
            "dog.jpg": {"actual_cat": 0, "actual_dog": 1},
            "cat.jpg": {"actual_cat": 1, "actual_dog": 0},
        }
        out = self.root / "outputs"
        stats = write_evaluation(counts, [], truth, self.metadata, out)
        self.assertEqual(stats["count_exact"], 1)
        self.assertEqual(stats["final_count_exact"], 2)
        s = (out / "검증_요약.md").read_text(encoding="utf-8")
        self.assertIn("1/1장 (100.0%)", s)
        self.assertIn("2/2장 (100.0%)", s)
        with (out / "검증_상세.csv").open(encoding="utf-8-sig") as h:
            values = list(csv.DictReader(h))
        self.assertEqual(values[1]["final_count_check"], "최종 개수 일치(REVIEW 포함)")


if __name__ == "__main__":
    unittest.main()
