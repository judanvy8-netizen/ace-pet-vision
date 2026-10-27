from pathlib import Path
import csv
import hashlib
import io
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source"))
from pet_model import choose_breed, download_verified
from main import process_images
from visualizer import animal_counts
import main as app


class PipelineTests(unittest.TestCase):
    def test_breed_threshold_and_species(self):
        names = ["Beagle", "Siamese", "Pug"]
        self.assertEqual(choose_breed([.8, .1, .1], names, "dog", .5)["breed"], "Beagle")
        self.assertEqual(choose_breed([.4, .3, .3], names, "dog", .5)["reason"], "low_confidence")
        self.assertEqual(choose_breed([.1, .8, .1], names, "dog", .5)["reason"], "species_mismatch")

    def test_download_cache_and_bad_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "weights.pth"
            path.write_bytes(b"original")
            with patch("pet_model.urllib.request.urlopen") as download:
                download_verified("https://example.com/model", path, hashlib.sha256(b"original").hexdigest())
                download.assert_not_called()
            with patch("pet_model.urllib.request.urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaises(RuntimeError):
                    download_verified("https://example.com/model", path, "0" * 64)
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(path.parent.glob("*.part")), [])



    def test_multianimal_no_detection_and_corrupt_input(self):
        from PIL import Image
        class FakeModel:
            def predict(self, image):
                if image.getpixel((0, 0)) == (0, 0, 0):
                    return []
                common = {"species": "dog", "detection_confidence": .9,
                          **choose_breed([.8, .1, .1], ["Beagle", "Siamese", "Pug"], "dog", .5)}
                return [{**common, "x1": 0, "y1": 0, "x2": 40, "y2": 40},
                        {**common, "species": "cat", "breed": "Siamese", "candidate_breed": "Siamese", "x1": 50, "y1": 10, "x2": 95, "y2": 60}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "사진", root / "결과"
            source.mkdir(); output.mkdir()
            Image.new("RGB", (100, 100), "white").save(source / "두마리.png")
            Image.new("RGB", (100, 100), "black").save(source / "없음.png")
            (source / "손상.jpg").write_bytes(b"broken")
            rows, summary = process_images(sorted(source.iterdir()), source, output, FakeModel())
            self.assertEqual(len(rows), 2)
            counts = {r["filename"]: r for r in summary}
            self.assertEqual((counts["두마리.png"]["dog_count"], counts["두마리.png"]["cat_count"]), (1, 1))
            self.assertEqual((counts["없음.png"]["dog_count"], counts["없음.png"]["cat_count"]), (0, 0))
            self.assertEqual(counts["손상.jpg"]["dog_count"], "")
            with Image.open(output / "images" / "두마리.png.png") as annotated:
                self.assertEqual(annotated.width, 100)
                self.assertGreater(annotated.height, 100)
            self.assertEqual({x["status"] for x in summary}, {"ok", "no_detection", "error"})
            self.assertTrue((output / "images" / "두마리.png.png").is_file())
            with (output / "image_summary.csv").open(encoding="utf-8-sig") as handle:
                saved = list(csv.DictReader(handle))
                self.assertEqual(len(saved), 3)
                self.assertEqual(next(r for r in saved if r["filename"] == "두마리.png")["cat_count"], "1")


    def test_default_run_uses_local_images_without_dataset(self):
        from PIL import Image
        class NoAnimals:
            def predict(self, image):
                return []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "data" / "images", root / "outputs"
            source.mkdir(parents=True)
            for name in ("01_cat_test.png", "02_dog_test.png"):
                Image.new("RGB", (80, 80)).save(source / name)
            with patch.object(app, "INPUT_DIR", source), patch.object(app, "OUTPUT_DIR", output), \
                 patch.object(app, "PetModel", return_value=NoAnimals()), patch.object(app.logging, "basicConfig"):
                self.assertEqual(app.main([]), 0)
            runs = list(output.iterdir())
            self.assertEqual(len(runs), 1)
            with (runs[0] / "image_summary.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["cat_count"] == "0" and row["dog_count"] == "0" for row in rows))
            self.assertFalse((root / "data" / "oxford-iiit-pet").exists())

    def test_species_counts_include_each_detection(self):
        self.assertEqual(animal_counts([]), {"dog_count": 0, "cat_count": 0})
        self.assertEqual(animal_counts([{"species": "dog"}, {"species": "cat"}, {"species": "dog"}]),
                         {"dog_count": 2, "cat_count": 1})


if __name__ == "__main__":
    unittest.main()
