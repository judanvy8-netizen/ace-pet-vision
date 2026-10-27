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
                self.assertEqual(app.main([]), 1)
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
