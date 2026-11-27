"""Download, safely extract, audit and visualize data. Does not load any model."""
from __future__ import annotations

import argparse
import platform
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import PIL
import torch

try:
    from .data import (HERE, CLASSES, EXPECTED_BYTES, EXPECTED_SHA256, MIRROR_COMMIT,
                       download_notmnist, safe_extract, NotMNISTSmall, ensure_mnist_test,
                       load_mnist_test, tensor_statistics, digest, write_json, MNIST_FILES)
except ImportError:
    from data import (HERE, CLASSES, EXPECTED_BYTES, EXPECTED_SHA256, MIRROR_COMMIT,
                      download_notmnist, safe_extract, NotMNISTSmall, ensure_mnist_test,
                      load_mnist_test, tensor_statistics, digest, write_json, MNIST_FILES)


def sample_figure(mnist, notmnist, output: Path) -> list[dict]:
    """Two examples per class per dataset; selection is for display only."""
    figure, axes = plt.subplots(10, 4, figsize=(7, 15), layout="constrained")
    chosen = []
    mx, my = mnist.tensors
    for label in range(10):
        # Deterministic first two examples in each dataset's existing order.
        mi = torch.where(my == label)[0][:2].tolist()
        ni = torch.where(notmnist.targets == label)[0][:2].tolist()
        for column, index in enumerate(mi):
            axes[label, column].imshow(mx[index].reshape(28, 28), cmap="gray",
                                      vmin=0, vmax=1, interpolation="nearest")
            axes[label, column].set_title(f"Digit {label} | #{index}", fontsize=9)
        for column, index in enumerate(ni, start=2):
            axes[label, column].imshow(notmnist.images[index].reshape(28, 28), cmap="gray",
                                      vmin=0, vmax=1, interpolation="nearest")
            axes[label, column].set_title(f"Letter {CLASSES[label]} | #{index}", fontsize=9)
        for axis in axes[label]:
            axis.set_axis_off()
        chosen.append({"digit_class": label, "mnist_indices": mi,
                       "letter_class": CLASSES[label], "notmnist_indices": ni,
                       "notmnist_files": [notmnist.paths[i] for i in ni]})
    figure.suptitle("MNIST test (left)  |  NotMNIST small (right)\n"
                   "Native polarity; float32 / 255; no normalization or augmentation",
                   fontsize=12)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=200, facecolor="white")
    plt.close(figure)
    return chosen


def prepare(data_root: Path, output_root: Path, mnist_root: Path | None, batch_size=256):
    # Small operations are faster and predictable on CPU with two threads.
    torch.set_num_threads(2)
    data_root, output_root = Path(data_root).resolve(), Path(output_root).resolve()
    protected = [HERE.parent / "toy_regression", HERE.parent / "mnist_classification"]
    for output in (data_root, output_root):
        if any(output.is_relative_to(root.resolve()) for root in protected):
            raise ValueError("Refusing to write inside an existing experiment directory")
    archive, provenance = download_notmnist(data_root)
    extracted, manifest = safe_extract(archive, data_root)
    print("Reading every NotMNIST small image...", flush=True)
    notmnist = NotMNISTSmall(extracted)
    mnist_root = ensure_mnist_test(mnist_root, data_root / "mnist_test")
    mnist = load_mnist_test(mnist_root)
    not_stats = tensor_statistics(notmnist, batch_size)
    not_stats["class_counts"] = notmnist.class_counts
    not_stats.update({"raw_images": sum(notmnist.raw_counts.values()),
                      "raw_class_counts": notmnist.raw_counts,
                      "corrupt_images": len(notmnist.corrupt), "corrupt_files": notmnist.corrupt,
                      "source_image_modes": notmnist.source_modes,
                      "all_valid_examples_used": True, "subsampling": False})
    mnist_stats = tensor_statistics(mnist, batch_size)
    chosen = sample_figure(mnist, notmnist, output_root / "figures" / "dataset_samples.png")
    write_json(data_root / "notmnist_file_manifest.json", manifest)
    source = HERE.parent / "mnist_classification" / "data.py"
    info = {
        "stage": "dataset_preparation_only", "schema_version": 1,
        "model_training_performed": False, "model_inference_performed": False,
        "predictive_entropy_computed": False, "auroc_computed": False,
        "future_experiment": {"methods": ["Ensemble", "Ensemble + R", "Ensemble + AT", "MC Dropout"],
                              "M": [1, 3, 5], "seeds": [42, 123, 2026],
                              "status": "not_run"},
        "notmnist_source": {
            **provenance, "dataset": "NotMNIST", "subset": "small/test",
            "archive": "notMNIST_small.tar.gz", "expected_bytes": EXPECTED_BYTES,
            "expected_sha256": EXPECTED_SHA256, "mirror_commit": MIRROR_COMMIT,
            "mirror_documentation": "https://github.com/mathefuchs/robust-pll",
            "historical_reference": "https://github.com/tensorflow/tensorflow/blob/v0.12.0/tensorflow/examples/udacity/1_notmnist.ipynb",
            "version_verification": [
                "Historical TensorFlow v0.12.0 notebook specifies small/test archive size 8458043",
                "Mirror authors identify this exact archive as Yaroslav Bulatov's NotMNIST",
                "Ten A-J classes, 18726 PNG members, 18724 readable native L/28x28 images",
                "Two known unreadable files in A and F; no large-set or resampled substitute",
                ("Original URL download matches the independently downloaded mirror byte-for-byte by SHA-256"
                 if provenance["used_url"] == provenance["original_url"] else
                 "Original unavailable on this run; pinned mirror matches the archived small/test version checks"),
                "SHA-256 was initially pinned from mirror, not from an author-published checksum"],
            "extraction": "Preflight allowlist; regular PNG files only; no links/traversal; bounded sizes; cached bytes rechecked",
            "extracted_file_hash_manifest": "data/notmnist_file_manifest.json",
        },
        "mnist_source": {"root": str(mnist_root), "read_only": True,
                         "files": {name: {"md5": md5, "sha256": digest(mnist_root / name)}
                                   for name, md5 in MNIST_FILES.items()},
                         "preprocessing_reference": "mnist_classification/data.py:prepare/read_idx",
                         "reference_sha256": digest(source) if source.exists() else None,
                         "model_input_reference": "mnist_classification/model.py:MLP (784 features; flatten(1))"},
        "preprocessing": {"width": 28, "height": 28, "channels": 1,
                          "grayscale": "Pillow convert('L'); all valid source files already L",
                          "dtype": "float32", "scaling": "uint8 -> float32 / 255",
                          "range": [0, 1], "sample_shape": [784],
                          "batch_shape": ["N", 784], "flatten_order": "row-major",
                          "polarity": "Native light foreground on dark background in both datasets",
                          "polarity_inversion": False,
                          "polarity_check": "Visual inspection of raw glyphs and paired figure; no border heuristic (glyphs may touch edges)",
                          "resize": False, "normalization": None, "augmentation": None,
                          "class_to_idx": notmnist.class_to_idx,
                          "label_semantics": "A-J are dataset metadata, not MNIST digit targets or OOD scores"},
        "notmnist": not_stats, "mnist_test": mnist_stats,
        "sample_figure": {"path": "figures/dataset_samples.png", "selection": chosen,
                          "note": "Class rows aid display only; letters are not mapped to digit classes"},
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "pillow": PIL.__version__,
                        "matplotlib": matplotlib.__version__, "device": "cpu"},
        "test_status": "Preparation checks passed; see test_results.json for independent tests",
    }
    write_json(output_root / "dataset_info.json", info)
    print(f"NotMNIST: {len(notmnist)} valid, {len(notmnist.corrupt)} corrupt; {notmnist.class_counts}")
    print(f"MNIST test: {len(mnist)}; NotMNIST tensor: {not_stats['tensor_shape']}, float32, [0,1]")
    print(f"Saved {output_root / 'dataset_info.json'} and figures/dataset_samples.png", flush=True)
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=HERE / "data")
    parser.add_argument("--output-root", type=Path, default=HERE / "outputs")
    parser.add_argument("--mnist-root", type=Path, help="Read-only directory with official MNIST test IDX gzip files")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    # Default to existing test data if present. Otherwise use only this stage's cache (e.g. Colab).
    existing = HERE.parent / "mnist_classification" / "data"
    mnist_root = args.mnist_root
    if mnist_root is None and all((existing / name).is_file() for name in MNIST_FILES):
        mnist_root = existing
    prepare(args.data_root, args.output_root, mnist_root, args.batch_size)


if __name__ == "__main__":
    main()
