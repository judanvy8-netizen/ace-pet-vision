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
            self.assertEqual((counts["두마리.png"]["dog"], counts["두마리.png"]["cat"]), (1, 1))
            self.assertEqual((counts["없음.png"]["dog"], counts["없음.png"]["cat"]), (0, 0))
            self.assertEqual(counts["손상.jpg"]["dog"], "")
            with Image.open(output / "두마리_pets.jpg") as annotated:
                self.assertEqual(annotated.width, 100)
                self.assertGreater(annotated.height, 100)
            self.assertEqual({x["status"] for x in summary}, {"ok", "no_detection", "error"})
            self.assertTrue((output / "두마리_pets.jpg").is_file())
            with (output / "pet_counts.csv").open(encoding="utf-8-sig") as handle:
                saved = list(csv.DictReader(handle))
                self.assertEqual(len(saved), 3)
                self.assertEqual(next(r for r in saved if r["filename"] == "두마리.png")["cat"], "1")


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
            self.assertTrue((output / "01_cat_test_pets.jpg").is_file())
            with (output / "pet_counts.csv").open(encoding="utf-8-sig") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["cat"] == "0" and row["dog"] == "0" for row in rows))
            self.assertFalse((root / "data" / "oxford-iiit-pet").exists())

    def test_species_counts_include_each_detection(self):
        self.assertEqual(animal_counts([]), {"dog_count": 0, "cat_count": 0})
        self.assertEqual(animal_counts([{"species": "dog"}, {"species": "cat"}, {"species": "dog"}]),
                         {"dog_count": 2, "cat_count": 1})

    def test_ground_truth_blank_zero_and_invalid(self):
        from PIL import Image
        class NoAnimals:
            def predict(self, image):
                return []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            gt = root / "ground_truth.csv"
            gt.write_text("filename,actual_cat,actual_dog\npet.png,1,0\nblank.png,,,\n", encoding="utf-8")
            truth = app.read_ground_truth(gt)
            self.assertIsNone(truth["blank.png"]["actual_cat"])
            self.assertEqual(truth["pet.png"]["actual_dog"], 0)
            source, output = root / "images", root / "output"
            source.mkdir(); output.mkdir()
            Image.new("RGB", (100, 100)).save(source / "pet.png")
            Image.new("RGB", (100, 100)).save(source / "blank.png")
            _, summary = app.process_images(sorted(source.iterdir()), source, output, NoAnimals(), truth)
            by_name = {row["filename"]: row for row in summary}
            self.assertEqual(by_name["pet.png"]["cat_error"], 1)
            self.assertEqual(by_name["pet.png"]["dog_error"], 0)
            self.assertEqual(by_name["blank.png"]["cat_error"], "")
            gt.write_text("filename,actual_cat,actual_dog\npet.png,-1,0\n")
            with self.assertRaises(ValueError):
                app.read_ground_truth(gt)

    def test_opencv_exif_orientation_and_korean_path(self):
        from PIL import Image, ImageOps
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "회전사진.jpg"
            exif = Image.Exif()
            exif[274] = 6
            Image.new("RGB", (40, 20), "red").save(path, exif=exif)
            with Image.open(path) as original:
                expected = ImageOps.exif_transpose(original).size
            self.assertEqual(app.read_image(path).size, expected)

    def test_same_stem_preserves_both_outputs_and_inputs(self):
        from PIL import Image
        class NoAnimals:
            def predict(self, image):
                return []
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "input", root / "output"
            source.mkdir(); output.mkdir()
            for name in ("pet.jpg", "pet.png"):
                Image.new("RGB", (60, 60)).save(source / name)
            files = sorted(source.iterdir())
            hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in files]
            _, summary = app.process_images(files, source, output, NoAnimals())
            names = [row["result_image"] for row in summary]
            self.assertEqual(len(set(names)), 2)
            self.assertTrue(all((output / name).exists() for name in names))
            self.assertEqual(hashes, [hashlib.sha256(p.read_bytes()).hexdigest() for p in files])


if __name__ == "__main__":
    unittest.main()
