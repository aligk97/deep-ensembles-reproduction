import csv
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys

import numpy as np
import torch

DEFAULT_RUN_ROOT = Path(__file__).resolve().parents[2] / "results" / "raw" / "toy_regression"


def utc_now():
    return datetime.now(timezone.utc)


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def config_hash(config):
    return sha256(json.dumps(config.to_dict(), sort_keys=True).encode()).hexdigest()


def new_run_dir(output_dir, digest):
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    base = utc_now().strftime("%Y%m%d_%H%M%S") + "_" + digest[:10]
    for index in range(10000):
        path = output_dir / (base if index == 0 else f"{base}_{index}")
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise FileExistsError(f"Could not allocate a unique run directory in {output_dir}")


def git_metadata():
    repo = Path(__file__).resolve().parents[2]

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True,
                                       stderr=subprocess.DEVNULL).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def configure_torch(requested, deterministic):
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available()
                          else "cpu" if requested == "auto" else requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu or auto")
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if device.type == "cpu":
        torch.set_num_threads(1)
    return device


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)


def environment(device):
    return {
        "os": platform.platform(), "python": sys.version, "cpu": platform.processor() or platform.machine(),
        "logical_cpu_count": os.cpu_count(), "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda": torch.version.cuda, "torch_threads": torch.get_num_threads(),
        "torch_deterministic": torch.are_deterministic_algorithms_enabled(),
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "packages": {package: version(package) for package in
                     ("torch", "numpy", "matplotlib", "pandas", "PyYAML", "pytest")},
        "reproducibility": "Identical CPU and GPU results are not guaranteed.",
    }
