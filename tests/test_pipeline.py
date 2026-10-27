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
from main import load_split, compare_predictions, process_images


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

    def test_official_split_format_and_missing_image(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            (root / "images" / "Abyssinian_1.jpg").touch()
            (root / "annotations" / "test.txt").write_text("Abyssinian_1 1 1 1\nbeagle_1 5 2 4\n")
            _, files, truth, total = load_split(root, limit=1)
            self.assertEqual(total, 2)
            self.assertEqual(truth[files[0].name], {"breed": "Abyssinian", "species": "cat"})
            with self.assertRaises(ValueError):
                load_split(root, limit=0)

    def test_traversal_annotation_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "images").mkdir()
            (root / "annotations").mkdir()
            (root / "annotations" / "test.txt").write_text("../beagle_1 5 2 4\n")
            with self.assertRaises(ValueError):
                load_split(root)

    def test_multianimal_no_detection_and_corrupt_input(self):
        from PIL import Image
        class FakeModel:
            def predict(self, image):
                if image.getpixel((0, 0)) == (0, 0, 0):
                    return []
                common = {"species": "dog", "detection_confidence": .9,
                          **choose_breed([.8, .1, .1], ["Beagle", "Siamese", "Pug"], "dog", .5)}
                return [{**common, "x1": 0, "y1": 0, "x2": 40, "y2": 40},
                        {**common, "x1": 50, "y1": 10, "x2": 95, "y2": 60}]
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / "사진", root / "결과"
            source.mkdir(); output.mkdir()
            Image.new("RGB", (100, 100), "white").save(source / "두마리.png")
            Image.new("RGB", (100, 100), "black").save(source / "없음.png")
            (source / "손상.jpg").write_bytes(b"broken")
            rows, summary = process_images(sorted(source.iterdir()), source, output, FakeModel())
            self.assertEqual(len(rows), 2)
            self.assertEqual({x["status"] for x in summary}, {"ok", "no_detection", "error"})
            self.assertTrue((output / "images" / "두마리.png.png").is_file())
            with (output / "image_summary.csv").open(encoding="utf-8-sig") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 3)

    def test_comparison_uses_largest_box_not_ground_truth(self):
        truth = {"x.jpg": {"breed": "Beagle", "species": "dog"}}
        rows = [{"filename": "x.jpg", "breed": "Beagle", "species": "dog", "x1": 0,"y1": 0,"x2": 10,"y2": 10},
                {"filename": "x.jpg", "breed": "Pug", "species": "dog", "x1": 0,"y1": 0,"x2": 20,"y2": 20}]
        result = compare_predictions(truth, rows, [{"filename": "x.jpg", "status": "ok"}])[0]
        self.assertEqual(result["comparison"], "mismatch")
        self.assertTrue(result["multiple_animals_review"])


if __name__ == "__main__":
    unittest.main()
