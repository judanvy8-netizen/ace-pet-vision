import unittest
import sys
import csv
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from types import SimpleNamespace
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source"))
from shared.fusion import choose_supported_breed
from shared.pet_model import BreedClassifier, CAT_BREEDS
from shared import common
from shared.visualizer import annotate
import main as app

CLASSES = ["Bengal", "Chihuahua", "Sphynx", "Beagle"]

class FusionTests(unittest.TestCase):
    def choose(self, probabilities, yolo_species):
        return choose_supported_breed(probabilities, CLASSES, yolo_species, .50, CAT_BREEDS)

    def test_sphynx_95_corrects_yolo_dog_to_cat_review(self):
        result = self.choose([.01, .02, .95, .02], "dog")
        self.assertEqual((result["species"], result["candidate_breed"], result["reason"]),
                         ("cat", "Sphynx", "species_corrected_high_confidence"))
        self.assertEqual(result["yolo_species"], "dog")
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["breed_confidence"], .95)

    def test_chihuahua_uses_dog_candidate_not_wrong_cat_global(self):
        result = self.choose([.33, .25, .27, .15], "dog")
        self.assertEqual((result["species"], result["candidate_breed"], result["status"]),
                         ("dog", "Chihuahua", "review"))
        self.assertEqual(result["resnet_top_breed"], "Bengal")
        self.assertEqual(result["breed_confidence"], .25)
        self.assertIn("Bengal 33.0%", result["top3_global"])

    def test_bengal_49_point_6_is_below_50_despite_rounding(self):
        result = self.choose([.496, .19, .18, .134], "cat")
        self.assertEqual(result["reason"], "low_confidence")
        self.assertEqual(result["resnet_top_breed"], "Bengal")

    def test_weak_opposite_species_never_forces_override(self):
        result = self.choose([.70, .11, .10, .09], "dog")
        self.assertEqual((result["species"], result["reason"]), ("dog", "species_conflict"))

    def test_crop_pipeline_corrects_species_and_visualizes(self):
        box = MagicMock()
        box.cls.item.return_value = 16
        box.conf.item.return_value = .47
        box.xyxy[0].cpu().tolist.return_value = [5.1, 4.2, 55.2, 56.3]
        detected = SimpleNamespace(names={16: "dog"}, boxes=[box])
        image = np.asarray(Image.new("RGB", (70, 70), "white"))[:, :, ::-1].copy()
        class Classifier:
            def classify_with_context(self, crop, species, context):
                self.args = (species, crop.size, context.size)
                return choose_supported_breed([.01, .02, .95, .02],
                                               CLASSES, species, .5, CAT_BREEDS)
        classifier = Classifier()
        predictions = common.classify_pets(detected, image, classifier)
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["species"], "cat")
        self.assertEqual(predictions[0]["yolo_label"], "dog")
        self.assertEqual(classifier.args[0], "dog")
        self.assertGreater(annotate(Image.new("RGB", (70, 70), "white"), predictions).height, 70)

    def test_conditional_tta_only_for_uncertain_case(self):
        obj = object.__new__(BreedClassifier)
        obj.classes = tuple(CLASSES)
        obj.confidence = .5
        values = iter([[.01, .02, .95, .02], [.33, .25, .27, .15],
                       [.22, .34, .28, .16]])
        obj._probabilities = lambda image: next(values)
        crop = Image.new("RGB", (80, 80), "white")
        strong = obj.classify_with_context(crop, "dog")
        uncertain = obj.classify_with_context(crop, "dog")
        self.assertFalse(strong["tta_used"])
        self.assertEqual(strong["species"], "cat")
        self.assertTrue(uncertain["tta_used"])
        self.assertEqual(uncertain["species"], "dog")
        self.assertEqual(uncertain["initial_top_breed"], "Bengal")

    def test_wide_chihuahua_photo_uses_ratio_preserving_variant(self):
        """500x231 같은 사진은 비율을 유지한 추가 관점으로도 검사합니다."""
        obj = object.__new__(BreedClassifier)
        obj.classes = tuple(CLASSES)
        obj.confidence = .5
        visited = []
        samples = iter([[.35, .30, .20, .15], [.32, .37, .17, .14],
                        [.18, .61, .13, .08]])
        def infer(photo):
            visited.append(photo.size)
            return next(samples)
        obj._probabilities = infer
        result = obj.classify_with_context(Image.new("RGB", (500, 231), "white"), "dog")
        self.assertEqual(visited, [(500, 231), (500, 231), (224, 224)])
        self.assertTrue(result["tta_used"])
        self.assertEqual(result["species"], "dog")
        self.assertIn("비율유지:Chihuahua", result["variant_diagnostics"])
        self.assertEqual(result["initial_top_breed"], "Bengal")

    def test_strong_sphynx_correction_does_not_invoke_extra_views(self):
        """95% 스핑크스처럼 강한 교정 근거는 불필요한 증강으로 희석하지 않습니다."""
        obj = object.__new__(BreedClassifier)
        obj.classes = tuple(CLASSES)
        obj.confidence = .5
        visited = []
        def infer(photo):
            visited.append(photo.size)
            return [.01, .02, .95, .02]
        obj._probabilities = infer
        result = obj.classify_with_context(Image.new("RGB", (500, 200), "white"), "dog")
        self.assertEqual(result["species"], "cat")
        self.assertEqual(result["reason"], "species_corrected_high_confidence")
        self.assertEqual(visited, [(500, 200)])

    def test_full_main_writes_both_model_opinions_and_corrected_cat_count(self):
        box = MagicMock()
        box.cls.item.return_value = 16
        box.conf.item.return_value = .47
        box.xyxy[0].cpu().tolist.return_value = [8, 7, 70, 69]
        class Detector:
            names = {15: "cat", 16: "dog"}
            def predict(self, image, **kwargs):
                return [SimpleNamespace(names=self.names, boxes=[box])]
        class Classifier:
            def classify_with_context(self, crop, species, context):
                return choose_supported_breed([.01, .02, .95, .02],
                                               CLASSES, species, .5, CAT_BREEDS)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, out = root / "images", root / "outputs"
            source.mkdir()
            Image.new("RGB", (100, 80), "white").save(source / "sphynx.png")
            with patch.object(app, "SAMPLE_INPUT", source), patch.object(app, "OUTPUT", out), \
                 patch.object(app, "yolo", return_value=Detector()), \
                 patch.object(app, "breed_model", return_value=Classifier()):
                self.assertEqual(app.main([]), 0)
            with (out / "pet_counts.csv").open(encoding="utf-8-sig") as stream:
                row = list(csv.DictReader(stream))[0]
            self.assertEqual((row["cat"], row["dog"], row["status"]),
                             ("1", "0", "review_species"))
            with (out / "pet_breeds.csv").open(encoding="utf-8-sig") as stream:
                item = list(csv.DictReader(stream))[0]
            self.assertEqual((item["species"], item["yolo_label"], item["resnet_top_breed"]),
                             ("cat", "dog", "Sphynx"))
            self.assertIn("Sphynx 95.0%", item["top3_species"])
            self.assertTrue((out / "sphynx_pets.jpg").exists())

    def test_other_animal_review_keeps_original_yolo_label(self):
        box = MagicMock()
        box.cls.item.return_value = 18
        box.conf.item.return_value = .4
        box.xyxy[0].cpu().tolist.return_value = [5, 7, 68, 59]
        result = SimpleNamespace(names={18: "sheep"}, boxes=[box])
        frame = np.asarray(Image.new("RGB", (80, 70), "white"))[:, :, ::-1].copy()
        class Classifier:
            def classify_single_pet(self, crop):
                return dict(species="dog", candidate_breed="Samoyed", breed="unknown",
                            breed_confidence=.82, status="review", reason="breed_only")
        pets = common.classify_animal_proposals(result, frame, Classifier())
        self.assertEqual(len(pets), 1)
        self.assertEqual((pets[0]["species"], pets[0]["yolo_label"], pets[0]["yolo_species"]),
                         ("dog", "sheep", "other"))

if __name__ == "__main__":
    unittest.main()
