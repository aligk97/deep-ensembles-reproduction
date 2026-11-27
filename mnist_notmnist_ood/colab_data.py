"""Preserve the validated local sample order independently of the host OS."""
from __future__ import annotations

import hashlib
from pathlib import PurePosixPath

import torch

from data import write_json
from inference import load_datasets
from run_experiment import dataset_metadata


def align_notmnist_order(dataset):
    """Match Windows' case-insensitive class/filename order, without changing pixels.

    data.py deliberately remains unchanged. Its sorted(Path) order differs on
    Windows and Linux. Sorting *relative strings* with casefold gives the same
    reference order on both hosts. Ambiguous case-folded names fail explicitly.
    """
    paths = list(dataset.paths)
    keys = [tuple(part.casefold() for part in PurePosixPath(path).parts) for path in paths]
    if len(set(keys)) != len(paths) or len(paths) != len(dataset.images) or len(paths) != len(dataset.targets):
        raise ValueError("NotMNIST sample paths must be unique and match image/label counts")
    permutation = sorted(range(len(paths)), key=keys.__getitem__)
    before = hashlib.sha256(dataset.images.numpy().tobytes()).hexdigest()
    moved = sum(index != source for index, source in enumerate(permutation))
    if moved:
        indices = torch.tensor(permutation, dtype=torch.int64)
        dataset.images = dataset.images.index_select(0, indices)
        dataset.targets = dataset.targets.index_select(0, indices)
        dataset.paths = [paths[index] for index in permutation]
    return {"policy": "case-insensitive class/filename order matching the validated local dataset",
            "examples": len(paths), "positions_changed": moved,
            "native_tensor_sha256": before,
            "aligned_tensor_sha256": hashlib.sha256(dataset.images.numpy().tobytes()).hexdigest(),
            "pixels_modified": False, "labels_permuted_with_images": True}


def load_reference_datasets(config, reference, output):
    datasets = load_datasets(config)
    alignment = align_notmnist_order(datasets["notmnist"])
    records = dataset_metadata(datasets, output)
    for name, record in records.items():
        if record["tensor_sha256"] != reference["dataset_tensor_sha256"][name]:
            raise ValueError(f"Prepared {name} tensors differ even after reference-order alignment; "
                             "archive/preprocessing must be investigated; hash checks are not bypassed")
        if record["order_sha256"] != reference["dataset_order_sha256"][name]:
            raise ValueError(f"Prepared {name} sample IDs/labels/order differ from the validated local dataset")
    write_json(output / "dataset_order_alignment.json", alignment)
    print(f"NotMNIST reference order verified: {alignment['positions_changed']} positions aligned; "
          "pixels unchanged; tensor and sample-order SHA-256 match", flush=True)
    return datasets
