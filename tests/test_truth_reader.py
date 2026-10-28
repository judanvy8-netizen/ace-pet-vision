# [설명] 수기 정답표 테스트: 제목 없는 5열 CSV가 개·고양이 숫자와 함께 정상 해석되는지 확인합니다.
"""수기로 작성한 정답표의 새 5열 형식과 이전 형식 호환 검사."""
from pathlib import Path
import tempfile
import unittest
from shared.truth_reader import read_ground_truth


# [설명] 파일명 중복·음수 개수·종 불일치 등을 막아 정답표의 신뢰성을 지킵니다.
class TruthReaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "ground_truth.csv"

    def write(self, content):
        self.path.write_text(content, encoding="utf-8-sig")

    # [설명] 선생님 수기 형식인 제목 없는 5열 CSV를 정상 처리하는지 확인합니다.
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
