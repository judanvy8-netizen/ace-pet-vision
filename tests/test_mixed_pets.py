"""혼합 사진 입력 확인: 실제 모델 추론과 구분된 입력/모의 파이프라인 테스트."""
from pathlib import Path
from types import SimpleNamespace
from PIL import Image
from unittest.mock import MagicMock
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared import common


class MixedPetTests(unittest.TestCase):
    def test_four_images_and_manual_counts(self):
        """각 사진에 고양이 1마리, 개 1마리라는 수기 기준을 확인합니다."""
        files = common.image_files(ROOT / "data/mixed_samples")
        self.assertEqual([p.name for p in files], [f"mix_{i:02d}.jpg" for i in range(1, 5)])
        truth = common.read_ground_truth(ROOT / "data/mixed_ground_truth.csv")
        self.assertEqual(set(truth), {p.name for p in files})
        for path in files:
            with Image.open(path) as actual_photo:
                self.assertEqual(actual_photo.format, "JPEG")
                self.assertGreaterEqual(min(actual_photo.size), 100)
            image = common.read_image(path)
            self.assertGreaterEqual(min(image.shape[:2]), 100)
            self.assertEqual(truth[path.name], {"actual_cat": 1, "actual_dog": 1})

    def test_two_species_generate_two_separate_detections(self):
        """동일 사진의 두 YOLO 박스가 품종 분류·개수 집계에 각각 반영되는지 검증합니다."""
        def box(species_id, xyxy):
            result = MagicMock()
            result.cls.item.return_value = species_id
            result.conf.item.return_value = .90
            result.xyxy[0].cpu().tolist.return_value = xyxy
            return result

        class FakeBreedClassifier:
            def classify(self, crop, species):
                candidate = "British Shorthair" if species == "cat" else "Shiba Inu"
                return {"breed": candidate, "candidate_breed": candidate,
                        "breed_confidence": .85, "status": "classified", "reason": ""}

        yolo_result = SimpleNamespace(
            names={15: "cat", 16: "dog"},
            boxes=[box(15, [5, 10, 100, 110]), box(16, [130, 10, 235, 110])],
        )
        pets = common.classify_pets(yolo_result, np.zeros((120, 240, 3), dtype=np.uint8), FakeBreedClassifier())
        self.assertEqual([p["species"] for p in pets], ["cat", "dog"])
        self.assertEqual(sum(p["species"] == "cat" for p in pets), 1)
        self.assertEqual(sum(p["species"] == "dog" for p in pets), 1)


if __name__ == "__main__":
    unittest.main()
