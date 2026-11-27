"""Independent checks on the real prepared datasets, plus unsafe-archive tests."""
from __future__ import annotations

import argparse
import io
import json
import runpy
import tarfile
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import SequentialSampler

try:
    from .data import (HERE, CLASSES, EXPECTED_SHA256, EXPECTED_CORRUPT, NotMNISTSmall,
                       load_mnist_test, make_loader, safe_extract, digest, write_json,
                       tensor_statistics, preprocess_image, download_notmnist)
except ImportError:
    from data import (HERE, CLASSES, EXPECTED_SHA256, EXPECTED_CORRUPT, NotMNISTSmall,
                      load_mnist_test, make_loader, safe_extract, digest, write_json,
                      tensor_statistics, preprocess_image, download_notmnist)

DATA_ROOT, OUTPUT_ROOT = HERE / "data", HERE / "outputs"


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.info = json.loads((OUTPUT_ROOT / "dataset_info.json").read_text(encoding="utf-8"))
        cls.notmnist = NotMNISTSmall(DATA_ROOT / "notMNIST_small")
        cls.mnist = load_mnist_test(Path(cls.info["mnist_source"]["root"]))

    def test_archive_identity(self):
        archive = DATA_ROOT / "notMNIST_small.tar.gz"
        self.assertEqual(archive.stat().st_size, 8458043)
        self.assertEqual(digest(archive), EXPECTED_SHA256)
        self.assertEqual(self.info["notmnist_source"]["sha256"], EXPECTED_SHA256)

    def test_all_classes_and_counts(self):
        expected = dict(zip(CLASSES, (1872, 1873, 1873, 1873, 1873, 1872, 1872, 1872, 1872, 1872)))
        self.assertEqual(len(self.notmnist), 18724)
        self.assertEqual(sum(self.notmnist.raw_counts.values()), 18726)
        self.assertEqual(self.notmnist.class_counts, expected)
        self.assertEqual(self.info["notmnist"]["class_counts"], expected)
        self.assertEqual(self.notmnist.source_modes, {"L": 18724})

    def test_corrupt_files_excluded(self):
        self.assertEqual(tuple(x["path"] for x in self.notmnist.corrupt), EXPECTED_CORRUPT)
        self.assertTrue(set(EXPECTED_CORRUPT).isdisjoint(self.notmnist.paths))
        self.assertEqual(self.info["notmnist"]["corrupt_images"], 2)

    def test_every_extracted_file_matches_archive_manifest(self):
        manifest = json.loads((DATA_ROOT / "notmnist_file_manifest.json").read_text())
        files = {p.relative_to(self.notmnist.root).as_posix(): p
                 for p in self.notmnist.root.rglob("*.png")}
        self.assertEqual(set(files), set(manifest))
        for name, path in files.items():
            self.assertEqual(digest(path), manifest[name], name)

    def test_tensors_finite_dtype_range_and_shape(self):
        for dataset, count in ((self.notmnist, 18724), (self.mnist, 10000)):
            stats = tensor_statistics(dataset)
            self.assertEqual(stats["tensor_shape"], [count, 784])
            self.assertEqual(stats["dtype"], "float32")
            self.assertEqual((stats["pixel_min"], stats["pixel_max"]), (0, 1))
            self.assertEqual((stats["nan_count"], stats["inf_count"]), (0, 0))

    def test_notmnist_every_pixel_matches_native_scaling(self):
        # Independent NumPy reference for every readable image: no flips, inversion or resampling.
        for index, relative in enumerate(self.notmnist.paths):
            with Image.open(self.notmnist.root / relative) as image:
                raw = np.asarray(image).copy()
            expected = raw.astype(np.float32).reshape(784) / np.float32(255)
            np.testing.assert_array_equal(self.notmnist.images[index].numpy(), expected)

    def test_mnist_matches_existing_pipeline_exactly(self):
        source = HERE.parent / "mnist_classification" / "data.py"
        if source.exists():
            # run_path reads source without importing a package or creating __pycache__.
            reference = runpy.run_path(str(source))["read_idx"]
            root = Path(self.info["mnist_source"]["root"])
            expected_x = reference(root / "t10k-images-idx3-ubyte.gz", True).float().div_(255)
            expected_y = reference(root / "t10k-labels-idx1-ubyte.gz", False).long()
            self.assertTrue(torch.equal(self.mnist.tensors[0], expected_x))
            self.assertTrue(torch.equal(self.mnist.tensors[1], expected_y))
        else:
            self.skipTest("Existing MNIST source unavailable; IDX/hash tests still run (standalone Colab)")

    def test_loader_order_coverage_and_partial_batch(self):
        for dataset in (self.notmnist, self.mnist):
            loader = make_loader(dataset)
            self.assertIsInstance(loader.sampler, SequentialSampler)
            self.assertFalse(loader.drop_last)
            xs, ys, sizes = [], [], []
            for x, y in loader:
                xs.append(x); ys.append(y); sizes.append(len(x))
            reference_x = dataset.images if hasattr(dataset, "images") else dataset.tensors[0]
            reference_y = dataset.targets if hasattr(dataset, "targets") else dataset.tensors[1]
            self.assertTrue(torch.equal(torch.cat(xs), reference_x))
            self.assertTrue(torch.equal(torch.cat(ys), reference_y))
            self.assertEqual(sizes[0], 256)
            self.assertEqual(sizes[-1], len(dataset) % 256)
            self.assertEqual(sum(sizes), len(dataset))

    def test_preprocessing_polarity_and_no_implicit_resize(self):
        image = np.zeros((28, 28), dtype=np.uint8)
        image[10:18, 13:15] = 255
        x = preprocess_image(Image.fromarray(image)).reshape(28, 28)
        self.assertEqual(float(x[0, 0]), 0)
        self.assertEqual(float(x[12, 14]), 1)
        with self.assertRaises(ValueError):
            preprocess_image(Image.new("L", (32, 32)))

    def test_sample_figure_and_class_coverage(self):
        with Image.open(OUTPUT_ROOT / "figures" / "dataset_samples.png") as image:
            image.verify()
        selections = self.info["sample_figure"]["selection"]
        self.assertEqual([x["digit_class"] for x in selections], list(range(10)))
        self.assertEqual([x["letter_class"] for x in selections], list(CLASSES))
        for row in selections:
            self.assertEqual(len(row["notmnist_files"]), 2)

    def test_bad_cached_archive_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "notMNIST_small.tar.gz").write_bytes(b"not an archive")
            with self.assertRaises(ValueError):
                download_notmnist(root)

    def test_unsafe_archives_rejected_before_writing(self):
        names = ["../escape.png", "/absolute.png", "notMNIST_small/A/../../escape.png",
                 "notMNIST_small/A/C:\\escape.png", "notMNIST_small/K/image.png",
                 "notMNIST_small/A/valid.png"]
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                archive = root / "malicious.tar.gz"
                with tarfile.open(archive, "w:gz") as tar:
                    member = tarfile.TarInfo(name)
                    if name.endswith("valid.png"):
                        member.type, member.linkname = tarfile.SYMTYPE, "../../escape.png"
                        tar.addfile(member)
                    else:
                        member.size = 1
                        tar.addfile(member, io.BytesIO(b"x"))
                with self.assertRaises(ValueError):
                    safe_extract(archive, root / "extracted")
                self.assertFalse(any((root / "extracted").rglob("*.png")))


def main():
    global DATA_ROOT, OUTPUT_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    DATA_ROOT, OUTPUT_ROOT = args.data_root.resolve(), args.output_root.resolve()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DataTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    write_json(OUTPUT_ROOT / "test_results.json", {
        "successful": result.wasSuccessful(), "tests_run": result.testsRun,
        "failures": [{"test": str(test), "traceback": message} for test, message in result.failures],
        "errors": [{"test": str(test), "traceback": message} for test, message in result.errors],
        "skipped": [{"test": str(test), "reason": reason} for test, reason in result.skipped],
        "scope": "Real complete datasets, exact preprocessing, all loader batches, archive identity and extraction safety",
    })
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
