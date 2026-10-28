from pathlib import Path
import csv
import hashlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "source"))
from shared import common, pet_model
import main as app


class PipelineTests(unittest.TestCase):
    def test_breed_rejection(self):
        names = ["Beagle", "Siamese", "Pug"]
        self.assertEqual(pet_model.choose_breed([.8, .1, .1], names, "dog", .5)["breed"], "Beagle")
        self.assertEqual(pet_model.choose_breed([.4, .3, .3], names, "dog", .5)["reason"], "low_confidence")
        self.assertEqual(pet_model.choose_breed([.1, .8, .1], names, "dog", .5)["reason"], "species_mismatch")

    def test_verified_download_and_legacy_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "models" / "model.pt"
            old.parent.mkdir()
            old.write_bytes(b"original")
            checksum = hashlib.sha256(b"original").hexdigest()
            with patch.object(pet_model, "MODEL_DIR", root / "shared" / "models"), patch("shared.pet_model.urllib.request.urlopen") as network:
                new = pet_model.model_file("model.pt", "https://example.com/model", checksum)
                self.assertEqual(new.read_bytes(), b"original")
                network.assert_not_called()
            with patch("shared.pet_model.urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(RuntimeError):
                    pet_model.download_verified("https://example.com/model", new, "0" * 64)
            self.assertEqual(new.read_bytes(), b"original")

    def test_io_orientation_and_unique_names(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            exif = Image.Exif(); exif[274] = 6
            source = root / "사진.jpg"
            Image.new("RGB", (40, 20), "red").save(source, exif=exif)
            image = common.read_image(source)
            self.assertEqual(image.shape[:2], (40, 20))
            common.save_image(root / "결과.jpg", image)
            self.assertTrue((root / "결과.jpg").exists())
            with self.assertRaises(FileNotFoundError):
                common.image_files(root / "missing")
            used = set()
            self.assertNotEqual(common.result_path("pet.jpg", used), common.result_path("pet.png", used))

    def test_truth_blank_zero_and_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truth.csv"
            path.write_text("filename,actual_cat,actual_dog\na.jpg,,0\n")
            truth = common.read_ground_truth(path)
            self.assertIsNone(truth["a.jpg"]["actual_cat"])
            self.assertEqual(truth["a.jpg"]["actual_dog"], 0)
            path.write_text("filename,actual_cat,actual_dog\na.jpg,-1,0\n")
            with self.assertRaises(ValueError):
                common.read_ground_truth(path)

    def test_breed_checkpoint_without_optional_num_classes(self):
        """배포자 문서에 보장되지 않은 num_classes 키는 필수로 요구하지 않습니다."""
        from unittest.mock import MagicMock
        from shared.pet_model import BreedClassifier, EXPECTED_CLASSES
        checkpoint = {"classes": list(EXPECTED_CLASSES), "model_state": {"sentinel": 1}}
        fake_model = MagicMock()
        with patch("torch.load", return_value=checkpoint), patch("torchvision.models.resnet50", return_value=fake_model):
            classifier = BreedClassifier("dummy.pth")
        self.assertEqual(classifier.classes, EXPECTED_CLASSES)
        fake_model.load_state_dict.assert_called_once_with(checkpoint["model_state"], strict=True)

    def test_model_startup_failure_does_not_erase_existing_csv(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "images", root / "outputs"
            source.mkdir(); output.mkdir()
            Image.new("RGB", (80, 80), "white").save(source / "pet.png")
            existing = output / "pet_counts.csv"
            existing.write_bytes(b"previous results must survive\n")
            with patch.object(app, "SAMPLE_INPUT", source), patch.object(app, "OUTPUT", output), \
                 patch.object(app, "yolo", side_effect=RuntimeError("YOLO weights unavailable")):
                with self.assertRaisesRegex(RuntimeError, "unavailable"):
                    app.main([])
            self.assertEqual(existing.read_bytes(), b"previous results must survive\n")

    def test_actual_crop_classification_and_csv_integration(self):
        """YOLO 박스→RGB 크롭→품종 분류→결과 CSV/이미지 경로를 검증합니다."""
        from unittest.mock import MagicMock
        from types import SimpleNamespace
        from PIL import Image

        class Detector:
            names = {15: "cat", 16: "dog"}
            def predict(self, image, **kwargs):
                box = MagicMock()
                box.cls.item.return_value = 16
                box.conf.item.return_value = 0.94
                box.xyxy[0].cpu().tolist.return_value = [10.5, 8.0, 62.5, 47.2]
                return [SimpleNamespace(names=self.names, boxes=[box])]

        class Classifier:
            def __init__(self):
                self.calls = []
            def classify(self, crop, species):
                self.calls.append((crop.size, species, crop.getpixel((0, 0))))
                return pet_model.choose_breed([.92, .08], ["Beagle", "Siamese"], species, .5)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "images", root / "outputs"
            source.mkdir()
            Image.new("RGB", (90, 70), (30, 60, 200)).save(source / "dog.png")
            classifier = Classifier()
            with patch.object(app, "SAMPLE_INPUT", source), patch.object(app, "OUTPUT", output), \
                 patch.object(app, "yolo", return_value=Detector()), \
                 patch.object(app, "breed_model", return_value=classifier):
                self.assertEqual(app.main([]), 0)
            self.assertEqual(classifier.calls, [((53, 40), "dog", (30, 60, 200))])
            with (output / "pet_breeds.csv").open(encoding="utf-8-sig", newline="") as handle:
                breed_rows = list(csv.DictReader(handle))
            self.assertEqual((breed_rows[0]["species"], breed_rows[0]["breed"], breed_rows[0]["status"]),
                             ("dog", "Beagle", "classified"))
            with (output / "pet_counts.csv").open(encoding="utf-8-sig", newline="") as handle:
                count_rows = list(csv.DictReader(handle))
            self.assertEqual((count_rows[0]["cat"], count_rows[0]["dog"], count_rows[0]["status"]),
                             ("0", "1", "ok"))
            self.assertTrue((output / "dog_pets.jpg").is_file())

    def test_nested_duplicate_prefers_stronger_breed_and_keeps_other_animals(self):
        from shared.common import deduplicate_pets
        def pet(species, box, breed, confidence, detection=.8, status="classified"):
            return dict(species=species, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                        breed=breed, candidate_breed=breed, breed_confidence=confidence,
                        detection_confidence=detection, status=status)
        # 버만/브리티시 쇼트헤어 사례처럼 얼굴 박스가 전신 박스 안에 포함됨.
        big = pet("cat", (45, 60, 780, 500), "Birman", .57)
        small = pet("cat", (380, 60, 755, 373), "Birman", .83)
        cats = deduplicate_pets([big, small])
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0]["breed_confidence"], .83)
        big = pet("cat", (103, 67, 598, 548), "British Shorthair", .68)
        small = pet("cat", (267, 67, 591, 548), "Bengal", .41, status="review")
        cats = deduplicate_pets([big, small])
        self.assertEqual(len(cats), 1)
        self.assertEqual(cats[0]["breed"], "British Shorthair")
        # 서로 멀리 떨어진 두 동물은 따로 세야 합니다.
        nearby = pet("cat", (420, 80, 790, 450), "Birman", .85)
        self.assertEqual(len(deduplicate_pets([big, nearby])), 2)
        dog = pet("dog", (103, 67, 598, 548), "Beagle", .76)
        self.assertEqual(len(deduplicate_pets([big, dog])), 2)

    def test_rescue_only_for_images_without_detections(self):
        from types import SimpleNamespace
        from PIL import Image
        class Detector:
            names = {15: "cat", 16: "dog"}
            def __init__(self, kind):
                self.kind = kind
                self.calls = []
            def predict(self, image, **kwargs):
                self.calls.append(kwargs)
                is_empty = int(image[0, 0, 0]) < 100
                if self.kind == "primary" and is_empty:
                    return [SimpleNamespace(names=self.names, boxes=[])]
                return [SimpleNamespace(names=self.names, boxes=[1])]
        def classify(result, image, classifier):
            if not result.boxes:
                return []
            return [dict(species="dog", breed="Samoyed", candidate_breed="Samoyed",
                         breed_confidence=.7, detection_confidence=.8,
                         status="classified", reason="", x1=1, y1=1, x2=45, y2=50)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "images", root / "outputs"
            source.mkdir()
            Image.new("RGB", (90, 70), "white").save(source / "found.jpg")
            Image.new("RGB", (90, 70), "black").save(source / "missed.jpg")
            primary, rescue = Detector("primary"), Detector("rescue")
            with patch.object(app, "SAMPLE_INPUT", source), patch.object(app, "OUTPUT", output), \
                 patch.object(app, "yolo", return_value=primary), \
                 patch.object(app, "yolo_rescue", return_value=rescue) as init_rescue, \
                 patch.object(app, "breed_model"), patch.object(app, "classify_pets", side_effect=classify):
                self.assertEqual(app.main([]), 0)
            init_rescue.assert_called_once()
            self.assertEqual(len(primary.calls), 2)
            self.assertEqual(len(rescue.calls), 1)
            self.assertEqual(rescue.calls[0]["imgsz"], 960)
            self.assertEqual(rescue.calls[0]["conf"], .18)
            with (output / "pet_counts.csv").open(encoding="utf-8-sig") as file:
                rows = {r["filename"]: r for r in csv.DictReader(file)}
            self.assertEqual(rows["found.jpg"]["detector"], "yolo11n")
            self.assertEqual(rows["missed.jpg"]["detector"], "yolo11s_rescue")
            self.assertEqual(rows["missed.jpg"]["dog"], "1")
            with patch.object(app, "SAMPLE_INPUT", source), patch.object(app, "OUTPUT", output), \
                 patch.object(app, "yolo", return_value=primary), \
                 patch.object(app, "yolo_rescue", side_effect=AssertionError("should not rescue")), \
                 patch.object(app, "breed_model"), patch.object(app, "classify_pets", side_effect=classify):
                self.assertEqual(app.main(["--no-rescue"]), 0)

    def test_default_main_counts_breeds_and_corrupt_image(self):
        from PIL import Image
        class Detector:
            names = {15: "cat", 16: "dog"}
            def predict(self, image, **kwargs):
                return [image]
        def classify(result, image, classifier):
            if not image.any():
                return []
            return [{"species": species, "detection_confidence": .9, "x1": i * 40, "y1": 0, "x2": i * 40 + 35, "y2": 35,
                     **pet_model.choose_breed(scores, ["Beagle", "Siamese"], species, .5)}
                    for i, (species, scores) in enumerate([("dog", [.9,.1]), ("cat", [.1,.9])])]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "images", root / "outputs"
            source.mkdir()
            Image.new("RGB", (100, 100), "white").save(source / "two.png")
            Image.new("RGB", (100, 100), "black").save(source / "none.png")
            (source / "bad.jpg").write_bytes(b"broken")
            with patch.object(app, "SAMPLE_INPUT", source), patch.object(app, "OUTPUT", output), \
                 patch.object(app, "yolo", return_value=Detector()), patch.object(app, "breed_model"), \
                 patch.object(app, "classify_pets", side_effect=classify), patch.object(app.logging, "basicConfig"):
                self.assertEqual(app.main(["--no-rescue"]), 1)
            with (output / "pet_counts.csv").open(encoding="utf-8-sig") as handle:
                rows = {r["filename"]: r for r in csv.DictReader(handle)}
            self.assertEqual((rows["two.png"]["cat"], rows["two.png"]["dog"]), ("1", "1"))
            self.assertEqual((rows["none.png"]["cat"], rows["none.png"]["dog"]), ("0", "0"))
            self.assertEqual(rows["bad.jpg"]["status"], "error")
            with (output / "pet_breeds.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)
            self.assertTrue((output / "two_pets.jpg").exists())


if __name__ == "__main__":
    unittest.main()
