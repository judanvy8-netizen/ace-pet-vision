"""혼합 사진 테스트 버전의 결과 비교를 실제 AI 없이 검증합니다."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("mixed_runner_test", ROOT / "source" / "main-mixed.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class MixedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.img = self.base / "mixed"
        self.img.mkdir()
        self.jpg = self.img / "mix_01.jpg"
        Image.new("RGB", (117, 117), "white").save(self.jpg, "JPEG")
        self.csv = self.base / "truth.csv"
        self.csv.write_text("mix_01.jpg,개와 고양이,1,1,개와 고양이\n", encoding="utf-8-sig")
        self.truth = {"mix_01.jpg": {"actual_cat": 1, "actual_dog": 1}}
        self.boxes = [
            {"filename": "mix_01.jpg", "species": species, "x1": "5", "y1": "6",
             "x2": "54", "y2": "110", "detection_confidence": ".9"}
            for species in ("cat", "dog")
        ]

    def test_jpeg_and_ground_truth(self):
        files, truth = runner.check_samples(self.img, self.csv)
        self.assertEqual([x.name for x in files], ["mix_01.jpg"])
        self.assertEqual(truth, self.truth)

    def test_webp_renamed_as_jpg_is_rejected(self):
        self.jpg.write_bytes(b"RIFF\x00\x00\x00\x00WEBPbad")
        with self.assertRaises((OSError, ValueError)):
            runner.check_samples(self.img, self.csv)

    def test_pass_only_if_both_species_have_boxes(self):
        row = {"filename": "mix_01.jpg", "cat": "1", "dog": "1", "status": "ok"}
        result = runner.compare_predictions([self.jpg], self.truth, [row], self.boxes)
        self.assertEqual(result[0]["판정"], "통과")
        result = runner.compare_predictions([self.jpg], self.truth, [row], self.boxes[:1])
        self.assertEqual(result[0]["판정"], "실패")

    def test_one_species_missing_is_not_a_pass(self):
        row = {"filename": "mix_01.jpg", "cat": "1", "dog": "0", "status": "ok"}
        result = runner.compare_predictions([self.jpg], self.truth, [row], self.boxes[:1])
        self.assertEqual(result[0]["판정"], "실패")
        self.assertIn("개수 불일치", result[0]["설명"])

    def test_review_not_counted_as_normal_pass(self):
        row = {"filename": "mix_01.jpg", "cat": "1", "dog": "1", "status": "review_species"}
        result = runner.compare_predictions([self.jpg], self.truth, [row], self.boxes)
        self.assertEqual(result[0]["판정"], "검토")

    def test_report_writes_files(self):
        row = runner.compare_predictions([self.jpg], self.truth,
                                         [{"filename": "mix_01.jpg", "cat": "1", "dog": "1", "status": "ok"}], self.boxes)
        output = self.base / "out"
        stats = runner.save_report(row, output, ran_model=False)
        self.assertEqual(stats["통과"], 1)
        self.assertTrue((output / "혼합_테스트_요약.md").exists())
        self.assertTrue((output / "혼합_테스트_결과.csv").exists())


if __name__ == "__main__":
    unittest.main()
