"""Regression tests for Windows/Linux NotMNIST ordering, with zero model forwards."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
from types import SimpleNamespace
import unittest

import torch

from data import HERE, NotMNISTSmall, write_json
from colab_data import align_notmnist_order
from run_experiment import dataset_metadata

DATA_ROOT = HERE / "data/notMNIST_small"


def permuted(dataset, key):
    indices = sorted(range(len(dataset.paths)), key=lambda index: key(dataset.paths[index]))
    result = copy.copy(dataset)
    result.paths = [dataset.paths[index] for index in indices]
    indices = torch.tensor(indices, dtype=torch.int64)
    result.images = dataset.images.index_select(0, indices)
    result.targets = dataset.targets.index_select(0, indices)
    return result


class OrderTests(unittest.TestCase):
    def test_platform_orders_align_with_pixels_and_labels_preserved(self):
        fixture = SimpleNamespace(paths=["A/a.png", "A/B.png", "A/Z.png"],
                                  images=torch.tensor([[1.], [2.], [3.]]), targets=torch.tensor([4, 5, 6]))
        linux = permuted(fixture, PurePosixPath)
        windows = permuted(fixture, PureWindowsPath)
        self.assertNotEqual(linux.paths, windows.paths)
        before = {path: (image.clone(), int(target)) for path, image, target in zip(linux.paths, linux.images, linux.targets)}
        record = align_notmnist_order(linux)
        self.assertGreater(record["positions_changed"], 0)
        self.assertEqual(linux.paths, windows.paths)
        self.assertTrue(torch.equal(linux.images, windows.images))
        self.assertTrue(torch.equal(linux.targets, windows.targets))
        for path, image, target in zip(linux.paths, linux.images, linux.targets):
            self.assertTrue(torch.equal(image, before[path][0]))
            self.assertEqual(int(target), before[path][1])

    def test_alignment_is_idempotent(self):
        fixture = SimpleNamespace(paths=["A/a.png", "A/B.png"], images=torch.tensor([[1.], [2.]]), targets=torch.tensor([0, 0]))
        record = align_notmnist_order(fixture)
        self.assertEqual(record["positions_changed"], 0)
        self.assertEqual(record["native_tensor_sha256"], record["aligned_tensor_sha256"])

    def test_ambiguous_casefold_names_fail(self):
        fixture = SimpleNamespace(paths=["A/a.png", "A/A.png"], images=torch.ones(2, 1), targets=torch.zeros(2))
        with self.assertRaisesRegex(ValueError, "unique"):
            align_notmnist_order(fixture)

    def test_all_real_images_linux_and_windows_match_pinned_local_hash(self):
        torch.set_num_threads(2)
        reference = json.loads((HERE / "colab_reference.json").read_text(encoding="utf-8"))
        dataset = NotMNISTSmall(DATA_ROOT)
        self.assertEqual(len(dataset), 18724)
        linux = permuted(dataset, PurePosixPath)
        windows = permuted(dataset, PureWindowsPath)
        self.assertNotEqual(hashlib.sha256(linux.images.numpy().tobytes()).hexdigest(),
                            reference["dataset_tensor_sha256"]["notmnist"])
        report = align_notmnist_order(linux)
        self.assertEqual(report["positions_changed"], 16719)
        self.assertEqual(report["aligned_tensor_sha256"], reference["dataset_tensor_sha256"]["notmnist"])
        align_notmnist_order(windows)
        self.assertEqual(linux.paths, windows.paths)
        self.assertTrue(torch.equal(linux.images, windows.images))
        self.assertTrue(torch.equal(linux.targets, windows.targets))
        with tempfile.TemporaryDirectory() as tmp:
            record = dataset_metadata({"notmnist": linux}, Path(tmp))["notmnist"]
        self.assertEqual(record["order_sha256"], reference["dataset_order_sha256"]["notmnist"])


def main():
    global DATA_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    DATA_ROOT = args.data_root
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(OrderTests))
    write_json(args.output, {"successful": result.wasSuccessful(), "tests_run": result.testsRun,
                            "failures": [{"test": str(test), "traceback": msg} for test, msg in result.failures],
                            "errors": [{"test": str(test), "traceback": msg} for test, msg in result.errors],
                            "scope": "Explicit Windows + Linux order simulation on ALL 18,724 real images, including local tensor/order hashes",
                            "model_forwards": 0, "new_training": False, "real_colab_test": False})
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__": main()
