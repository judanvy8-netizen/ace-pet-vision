"""수기로 작성한 정답표의 새 5열 형식과 이전 형식 호환 검사."""
from pathlib import Path
import tempfile
import unittest
from shared.truth_reader import read_ground_truth


class TruthReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ground_truth.csv"

    def write(self, content):
        self.path.write_text(content, encoding="utf-8-sig")

    def test_headerless_five_columns(self):
        self.write("img_001.jpg,고양이,1,0,고양이 한 마리\n"
                   "img_002.jpg,개,0,1,개 한 마리\n")
        result = read_ground_truth(self.path)
        self.assertEqual(result["img_001.jpg"], {"actual_cat": 1, "actual_dog": 0})
        self.assertEqual(result["img_002.jpg"], {"actual_cat": 0, "actual_dog": 1})
        self.assertEqual(len(result), 2)

    def test_previous_header_format(self):
        self.write("filename,actual_cat,actual_dog,scene,note\n"
                   "a.jpg,,0,미확인,\n")
        self.assertIsNone(read_ground_truth(self.path)["a.jpg"]["actual_cat"])

    def test_no_animals_and_both_animals(self):
        self.write("none.jpg,대상 없음,0,0,빈 공간\n"
                   "both.jpg,개와 고양이,1,2,여러 마리\n")
        result = read_ground_truth(self.path)
        self.assertEqual(result["none.jpg"]["actual_dog"], 0)
        self.assertEqual(result["both.jpg"]["actual_dog"], 2)

    def test_wrong_species_or_negative_count(self):
        self.write("img_001.jpg,개,1,0,고양이 한 마리\n")
        with self.assertRaises(ValueError):
            read_ground_truth(self.path)
        self.write("img_001.jpg,고양이,-1,0,고양이 한 마리\n")
        with self.assertRaises(ValueError):
            read_ground_truth(self.path)

    def test_missing_invalid_or_duplicate_row(self):
        self.write("img_001.jpg,고양이,1,0\n")
        with self.assertRaises(ValueError):
            read_ground_truth(self.path)
        self.write("img_001.jpg,고양이,1,0,고양이 한 마리\n"
                   "img_001.jpg,고양이,1,0,고양이 한 마리\n")
        with self.assertRaises(ValueError):
            read_ground_truth(self.path)


if __name__ == "__main__":
    unittest.main()
