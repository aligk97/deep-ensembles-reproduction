"""Official MNIST IDX files, checksum verification, train-only split and scaling."""
from __future__ import annotations

import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import torch

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
MIRRORS = ("https://ossci-datasets.s3.amazonaws.com/mnist/",
           "https://storage.googleapis.com/cvdf-datasets/mnist/")


def md5(path):
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def download(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    for name, expected in FILES.items():
        path = root / name
        if path.exists() and md5(path) == expected:
            continue
        errors = []
        for mirror in MIRRORS:
            try:
                with urllib.request.urlopen(mirror + name, timeout=60) as response:
                    content = response.read()
                if hashlib.md5(content).hexdigest() != expected:
                    raise ValueError(f"Checksum mismatch: {name}")
                tmp = path.with_suffix(".tmp")
                tmp.write_bytes(content)
                tmp.replace(path)
                print(f"Downloaded and verified {name}", flush=True)
                break
            except (OSError, ValueError) as exc:
                errors.append(str(exc))
        else:
            raise RuntimeError(f"MNIST download failed for {name}: {errors}")
    (root / "checksums.json").write_text(json.dumps(FILES, indent=2), encoding="utf-8")


def read_idx(path: Path, images: bool):
    with gzip.open(path, "rb") as stream:
        content = stream.read()
    if images:
        magic, count, rows, cols = struct.unpack(">IIII", content[:16])
        if (magic, rows, cols) != (2051, 28, 28):
            raise ValueError("Invalid MNIST image header")
        array = np.frombuffer(content, dtype=np.uint8, offset=16).copy().reshape(count, 784)
    else:
        magic, count = struct.unpack(">II", content[:8])
        if magic != 2049:
            raise ValueError("Invalid MNIST label header")
        array = np.frombuffer(content, dtype=np.uint8, offset=8).copy().reshape(count)
    return torch.from_numpy(array)


def prepare(root: Path, config: dict, output: Path, device: torch.device):
    download(root)
    x = read_idx(root / "train-images-idx3-ubyte.gz", True).float().div_(255)
    y = read_idx(root / "train-labels-idx1-ubyte.gz", False).long()
    xt = read_idx(root / "t10k-images-idx3-ubyte.gz", True).float().div_(255)
    yt = read_idx(root / "t10k-labels-idx1-ubyte.gz", False).long()
    assert x.shape == (60000, 784) and xt.shape == (10000, 784)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(config["split_seed"]))
    nval = config["validation_size"]
    valid_idx, train_idx = order[:nval], order[nval:]
    if config["train_limit"]:
        train_idx = train_idx[:config["train_limit"]]
    test_idx = torch.arange(len(yt))
    if config["test_limit"]:
        test_idx = test_idx[:config["test_limit"]]
    np.savez_compressed(output / "splits.npz", train=train_idx.numpy(),
                        validation=valid_idx.numpy(), test=test_idx.numpy())
    # Epsilon uses ONLY the actual training subset's per-pixel range.
    train_x = x[train_idx]
    epsilon = config["epsilon_fraction"] * (train_x.amax(0) - train_x.amin(0))
    result = [train_x, y[train_idx], x[valid_idx], y[valid_idx], xt[test_idx], yt[test_idx], epsilon]
    metadata = {"train_n": len(train_idx), "validation_n": len(valid_idx),
                "test_n": len(test_idx), "split_seed": config["split_seed"],
                "epsilon_min": epsilon.min().item(), "epsilon_max": epsilon.max().item(),
                "epsilon_zero_pixels": int((epsilon == 0).sum()), "checksums": FILES,
                "scaling": "uint8 / 255; no mean/std normalization"}
    (output / "data_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return [item.to(device) for item in result]
