"""Verify the Figure 2(a) architecture on real MNIST batches without training."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import torch

from ..toy_regression.experiment_io import configure_torch
from .data import make_loaders
from .experiment_config import Config, set_seed
from .model import MNISTMLP

RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / "metrics" / "mnist" / "phase1_sanity.json"


def print_environment(device: torch.device):
    available = torch.cuda.is_available()
    print(f"Python: {platform.python_version()}\nPyTorch: {torch.__version__}")
    print(f"CUDA available: {available}\nPyTorch CUDA runtime: {torch.version.cuda}")
    gpu = torch.cuda.get_device_name(0) if available else None
    memory = torch.cuda.get_device_properties(0).total_memory if available else None
    print(f"GPU: {gpu or 'unavailable'}\nGPU memory: {f'{memory / 2**30:.2f} GiB' if memory else 'unavailable'}")
    print(f"Selected device: {device}")
    return {
        "python_version": platform.python_version(),
        "pytorch_version": str(torch.__version__),
        "cuda_available": available,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_model": gpu,
        "gpu_total_memory_bytes": memory,
        "selected_device": str(device),
    }


@torch.inference_mode()
def check_batch(model, loader, device, split):
    images, labels = next(iter(loader))
    size = min(loader.batch_size, len(loader.dataset))
    if images.shape != (size, 1, 28, 28) or images.dtype != torch.float32:
        raise ValueError(f"{split}: expected float32 grayscale MNIST images")
    if not ((images >= 0) & (images <= 1)).all():
        raise ValueError(f"{split}: expected pixel values in [0, 1]")
    if labels.shape != (size,) or labels.dtype != torch.long or not ((labels >= 0) & (labels < 10)).all():
        raise ValueError(f"{split}: expected integer labels in [0, 9]")

    images = images.to(device, non_blocking=loader.pin_memory)
    logits = model(images)
    if logits.device != images.device or logits.shape != (size, 10) or not logits.isfinite().all():
        raise ValueError(f"{split}: expected ten finite logits per image on {images.device}")
    probabilities = logits.softmax(dim=1)
    sums = probabilities.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), rtol=0, atol=1e-6)
    deviation = (sums - 1).abs().max().item()
    print(f"{split} on {logits.device}: inputs {list(images.shape)}, logits {list(logits.shape)}, "
          f"probabilities {list(probabilities.shape)}, max |sum - 1| = {deviation:.2e}")
    return {
        "device": str(logits.device),
        "input_shape": list(images.shape),
        "logits_shape": list(logits.shape),
        "probabilities_shape": list(probabilities.shape),
        "max_abs_softmax_sum_deviation": deviation,
    }


def run(config: Config, result_path: Path = RESULT_PATH):
    device = configure_torch(config.device, deterministic=True)
    environment = print_environment(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    model_seed, shuffle_seed = config.seeds
    set_seed(model_seed)
    print(f"Outer seed: {config.experiment_seed}, member index: {config.member_index}, "
          f"model seed: {model_seed}, shuffle seed: {shuffle_seed}, dropout: {config.dropout_p}")
    train, test = make_loaders(config, device)
    print(f"Train samples: {len(train.dataset)}\nTest samples: {len(test.dataset)}")
    if (len(train.dataset), len(test.dataset)) != (60_000, 10_000):
        raise ValueError("Expected the official 60,000/10,000 MNIST split")
    model = MNISTMLP(config.dropout_p).to(device).eval()
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if count != 240_610:
        raise ValueError(f"Expected 240,610 trainable parameters, found {count:,}")
    print(f"Trainable parameters: {count:,}")
    batches = {
        "train": check_batch(model, train, device, "Train"),
        "test": check_batch(model, test, device, "Test"),
    }
    peak_memory = None
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)
        print(f"Peak allocated by PyTorch: {peak_memory / 2**20:.2f} MiB")
    report = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "passed",
        "environment": environment,
        "config": {
            "experiment_seed": config.experiment_seed, "member_index": config.member_index,
            "model_seed": model_seed, "shuffle_seed": shuffle_seed,
            "batch_size": config.batch_size, "dropout_p": config.dropout_p,
        },
        "dataset": {"name": "MNIST", "train_samples": len(train.dataset), "test_samples": len(test.dataset)},
        "trainable_parameters": count,
        "batches": batches,
        "gpu_peak_allocated_bytes": peak_memory,
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    # Give concurrent runs separate staging files on the destination filesystem.
    with TemporaryDirectory(dir=result_path.parent) as staging:
        temporary = Path(staging) / result_path.name
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(result_path)
    print(f"All sanity checks passed.\nSaved: {result_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Config.data_root)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--experiment-seed", type=int, default=Config.experiment_seed)
    parser.add_argument("--member-index", type=int, default=Config.member_index)
    parser.add_argument("--dropout-p", type=float, default=Config.dropout_p)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=Config.device)
    parser.add_argument("--no-download", dest="download", action="store_false")
    config = Config(**vars(parser.parse_args()))
    try:
        run(config)
    except torch.cuda.OutOfMemoryError as exc:
        raise SystemExit("GPU memory exhausted. Retry with a smaller --batch-size.") from exc


if __name__ == "__main__":
    main()
