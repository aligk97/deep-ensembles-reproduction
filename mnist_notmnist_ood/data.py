"""Read-only evaluation datasets; no models, training, or inference.

Both datasets return (float32 [784], int64 label). Letter labels are metadata,
NOT digit targets. NotMNIST already has light glyphs on a dark background.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import tarfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, TensorDataset

HERE = Path(__file__).resolve().parent
CLASSES = tuple("ABCDEFGHIJ")
ORIGINAL_URL = "http://yaroslavvb.com/upload/notMNIST/notMNIST_small.tar.gz"
MIRROR_COMMIT = "398659255438af55e163e0e9e1697c2d423ee97d"
MIRROR_URL = ("https://raw.githubusercontent.com/mathefuchs/robust-pll/"
              f"{MIRROR_COMMIT}/external/notMNIST_small.tar.gz")
EXPECTED_BYTES = 8458043
# Pinned from the downloaded mirror, not an author-published SHA-256.
EXPECTED_SHA256 = "76b2ef7ccc56452964780569db37b279c9cbe1bf442c3b38be784752683b3cec"
EXPECTED_RAW_COUNTS = dict(zip(CLASSES, (1873, 1873, 1873, 1873, 1873,
                                        1873, 1872, 1872, 1872, 1872)))
EXPECTED_CORRUPT = (
    "A/RGVtb2NyYXRpY2FCb2xkT2xkc3R5bGUgQm9sZC50dGY=.png",
    "F/Q3Jvc3NvdmVyIEJvbGRPYmxpcXVlLnR0Zg==.png",
)
MNIST_FILES = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def digest(path: Path, algorithm="sha256") -> str:
    h = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def download_notmnist(root: Path) -> tuple[Path, dict]:
    """Try the original first, then ONLY the pinned, verified small mirror.

    Cached archives must have matching download provenance and SHA-256.
    A partial or unexpected archive is never used or silently substituted.
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "notMNIST_small.tar.gz"
    provenance_path = root / "download_provenance.json"
    if archive.exists():
        if archive.stat().st_size != EXPECTED_BYTES or digest(archive) != EXPECTED_SHA256:
            raise ValueError(f"Unexpected cached archive: {archive}; move it aside explicitly")
        if provenance_path.exists():
            provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
            if (provenance["sha256"] != EXPECTED_SHA256
                    or provenance["used_url"] not in (ORIGINAL_URL, MIRROR_URL)):
                raise ValueError("Cached download provenance does not match the pinned dataset")
            return archive, provenance
        # No fabricated source for a manually supplied archive.
        raise ValueError("Archive has no provenance; move it aside and rerun to download")
    attempts = []
    for url in (ORIGINAL_URL, MIRROR_URL):
        print(f"Downloading {url}", flush=True)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "DeepEnsembles-OOD-data/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                content = response.read(EXPECTED_BYTES + 1)
                resolved_url = response.geturl()
            if len(content) != EXPECTED_BYTES:
                raise ValueError(f"Archive size mismatch: {len(content)}")
            sha256 = hashlib.sha256(content).hexdigest()
            if sha256 != EXPECTED_SHA256:
                raise ValueError(f"Archive SHA-256 mismatch: {sha256}")
            attempts.append({"url": url, "status": "verified"})
            provenance = {"original_url": ORIGINAL_URL, "used_url": url,
                          "resolved_url": resolved_url, "sha256": sha256,
                          "bytes": len(content), "attempts": attempts,
                          "checksum_origin": "Pinned from mirror; not author-published"}
            temporary = archive.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(archive)
            write_json(provenance_path, provenance)
            return archive, provenance
        except (OSError, ValueError) as exc:
            attempts.append({"url": url, "status": "failed", "error": str(exc)})
            print(f"  Failed: {exc}", flush=True)
    raise RuntimeError(f"No verified NotMNIST small source available: {attempts}")


def safe_extract(archive: Path, destination: Path) -> tuple[Path, dict]:
    """Allow only regular PNGs and A-J directories; never trust tar paths/links.

    Preflight ALL members before writing. Verify cached extracted bytes on every
    run. No extractall, absolute paths, traversal, links, device nodes or metadata
    restoration. Size limits bound decompression and individual allocations.
    """
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seen, total, members = set(), 0, []
    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            name = member.name.rstrip("/")
            parts = PurePosixPath(name).parts
            if ("\\" in name or ":" in name or name.startswith("/")
                    or any(p in (".", "..", "") for p in name.split("/"))
                    or not parts or parts[0] != "notMNIST_small"
                    or name in seen or not (member.isdir() or member.isfile())):
                raise ValueError(f"Unsafe tar member: {member.name}")
            if member.isdir():
                valid = len(parts) == 1 or (len(parts) == 2 and parts[1] in CLASSES)
            else:
                valid = (len(parts) == 3 and parts[1] in CLASSES
                         and parts[2].endswith(".png") and 0 <= member.size <= 4 * 1024**2)
            if not valid:
                raise ValueError(f"Unexpected tar member: {member.name}")
            seen.add(name)
            total += member.size
            if total > 128 * 1024**2 or len(seen) > 20000:
                raise ValueError("Archive exceeds NotMNIST small extraction limits")
            target = destination.joinpath(*parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"Tar path escapes destination: {name}")
            if any(p.is_symlink() for p in [target, *target.parents] if p != destination.parent):
                raise ValueError(f"Symlink in extraction path: {name}")
            members.append((member, target))
        file_members = [(m, p) for m, p in members if m.isfile()]
        counts = Counter(p.parent.name for _, p in file_members)
        if dict(counts) != EXPECTED_RAW_COUNTS:
            raise ValueError(f"Not the complete small archive: {dict(counts)}")
        manifest = {}
        for member, target in members:
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            stream = tar.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive content: {member.name}")
            with stream:
                content = stream.read()
            expected = hashlib.sha256(content).hexdigest()
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or digest(target) != expected:
                target.write_bytes(content)
            manifest[target.relative_to(destination / "notMNIST_small").as_posix()] = expected
    extracted = destination / "notMNIST_small"
    actual = {p.relative_to(extracted).as_posix() for p in extracted.rglob("*") if p.is_file()}
    if actual != set(manifest):
        raise ValueError("Unexpected/missing files in extracted dataset; inspect cache")
    return extracted, manifest


def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Same scaling/flattening as MNIST; preserve native light-on-dark polarity."""
    if image.size != (28, 28):
        raise ValueError(f"Unexpected image size {image.size}; refusing implicit resizing")
    array = np.asarray(image.convert("L"), dtype=np.uint8).copy()
    return torch.from_numpy(array).to(torch.float32).div_(255).reshape(784)


class NotMNISTSmall(Dataset):
    """All readable small examples, ordered by class then filename, without sampling."""
    classes = CLASSES
    class_to_idx = {label: index for index, label in enumerate(CLASSES)}

    def __init__(self, root: Path):
        self.root = Path(root)
        if {p.name for p in self.root.iterdir() if p.is_dir()} != set(CLASSES):
            raise ValueError("Expected exactly A-J class directories")
        images, targets, self.paths, self.corrupt = [], [], [], []
        self.raw_counts, modes = {}, Counter()
        for label in CLASSES:
            paths = sorted((self.root / label).iterdir())
            self.raw_counts[label] = len(paths)
            for path in paths:
                if not path.is_file() or path.suffix.lower() != ".png" or path.is_symlink():
                    raise ValueError(f"Unexpected dataset entry: {path}")
                relative = path.relative_to(self.root).as_posix()
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        image.load()
                        tensor = preprocess_image(image)
                        modes[image.mode] += 1
                except (OSError, SyntaxError, UnidentifiedImageError) as exc:
                    # Malformed shapes/preprocessing errors fail instead of being hidden.
                    self.corrupt.append({"path": relative, "class": label,
                                         "error_type": type(exc).__name__, "error": str(exc)})
                    continue
                images.append(tensor)
                targets.append(self.class_to_idx[label])
                self.paths.append(relative)
        if self.raw_counts != EXPECTED_RAW_COUNTS:
            raise ValueError(f"Unexpected raw class counts: {self.raw_counts}")
        if tuple(x["path"] for x in self.corrupt) != EXPECTED_CORRUPT:
            raise ValueError(f"Unexpected corrupt-file list: {self.corrupt}")
        self.images = torch.stack(images)
        self.targets = torch.tensor(targets, dtype=torch.int64)
        self.source_modes = dict(modes)
        self.class_counts = {c: int((self.targets == i).sum()) for i, c in enumerate(CLASSES)}

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.images[index], self.targets[index]


def load_mnist_test(root: Path) -> TensorDataset:
    """Read only the two official test IDX files; never download/write to root."""
    root = Path(root)
    for name, expected in MNIST_FILES.items():
        if digest(root / name, "md5") != expected:
            raise ValueError(f"MNIST checksum mismatch: {name}")
    with gzip.open(root / "t10k-images-idx3-ubyte.gz", "rb") as stream:
        raw = stream.read()
    if struct.unpack(">IIII", raw[:16]) != (2051, 10000, 28, 28) or len(raw) != 16 + 10000 * 784:
        raise ValueError("Invalid MNIST test image header/length")
    x = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=16).copy().reshape(10000, 784))
    with gzip.open(root / "t10k-labels-idx1-ubyte.gz", "rb") as stream:
        raw = stream.read()
    if struct.unpack(">II", raw[:8]) != (2049, 10000) or len(raw) != 10008:
        raise ValueError("Invalid MNIST test label header/length")
    y = torch.from_numpy(np.frombuffer(raw, dtype=np.uint8, offset=8).copy()).long()
    return TensorDataset(x.float().div_(255), y)


def ensure_mnist_test(source: Path | None, cache: Path) -> Path:
    """Use an existing directory read-only, or download test files to OUR cache."""
    if source is not None:
        load_mnist_test(source)  # Validate before returning; no writes.
        return Path(source)
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    for name, expected in MNIST_FILES.items():
        target = cache / name
        if target.exists() and digest(target, "md5") == expected:
            continue
        errors = []
        for base in ("https://ossci-datasets.s3.amazonaws.com/mnist/",
                     "https://storage.googleapis.com/cvdf-datasets/mnist/"):
            try:
                with urllib.request.urlopen(base + name, timeout=30) as response:
                    content = response.read(4 * 1024**2)
                if hashlib.md5(content).hexdigest() != expected:
                    raise ValueError(f"MNIST checksum mismatch: {name}")
                temporary = target.with_suffix(".download")
                temporary.write_bytes(content)
                temporary.replace(target)
                break
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError(f"MNIST test download failed: {errors}")
    return cache


def make_loader(dataset: Dataset, batch_size=256, num_workers=0) -> DataLoader:
    """Sequential evaluation: include the final incomplete batch and every sample."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False,
                      num_workers=num_workers)


def tensor_statistics(dataset: Dataset, batch_size=256) -> dict:
    """Inspect EVERY loader batch, including labels and non-finite counts."""
    total, nan, inf, low, high = 0, 0, 0, float("inf"), float("-inf")
    first_shape, last_shape, label_counts = None, None, Counter()
    for x, y in make_loader(dataset, batch_size):
        if x.dtype != torch.float32 or x.ndim != 2 or x.shape[1] != 784:
            raise AssertionError(f"Unexpected image dtype/shape: {x.dtype}, {x.shape}")
        if y.dtype != torch.int64 or y.shape != (len(x),):
            raise AssertionError("Unexpected labels")
        first_shape = first_shape or list(x.shape)
        last_shape = list(x.shape)
        total += len(x)
        nan += int(torch.isnan(x).sum())
        inf += int(torch.isinf(x).sum())
        low, high = min(low, float(x.min())), max(high, float(x.max()))
        label_counts.update(y.tolist())
    if total != len(dataset) or nan or inf or not (0 <= low <= high <= 1):
        raise AssertionError("Dataset validation failed")
    return {"valid_images": total, "tensor_shape": [total, 784], "sample_shape": [784],
            "image_shape_before_flatten": [1, 28, 28], "dtype": "float32",
            "label_dtype": "int64", "pixel_min": low, "pixel_max": high,
            "nan_count": nan, "inf_count": inf, "batch_size": batch_size,
            "first_batch_shape": first_shape, "last_batch_shape": last_shape,
            "loader_seen": total, "shuffle": False, "drop_last": False,
            "class_counts": {str(k): label_counts[k] for k in sorted(label_counts)}}
