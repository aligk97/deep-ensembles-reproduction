"""Colab-only execution wrapper around the byte-identical final local runner."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import importlib.util
import json
from pathlib import Path
import runpy
import sys

sys.dont_write_bytecode = True
import torch
import engine
from colab_checkpoint import load_reference, checkpoint_records, validate_checkpoint, make_train_member

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        colab = importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ImportError):
        colab = False
    if not colab or os.environ.get("MNIST_COLAB_VALIDATE_ONLY") == "1":
        raise RuntimeError("This entrypoint never starts training outside real Colab")
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before Run all")
    config, signature = load_reference(ROOT)
    output = args.output.resolve()
    if not output.is_relative_to(Path("/content/drive/MyDrive/DeepEnsembles_MNIST_Figure2a")):
        raise ValueError("Final checkpoints must use the dedicated persistent Drive directory")
    output.mkdir(parents=True, exist_ok=True)
    expected = checkpoint_records(output, config)
    known = {path.resolve() for _, _, _, path in expected}
    if any(path.resolve() not in known for path in (output / "checkpoints").rglob("*.pt")):
        raise ValueError("Unexpected checkpoint in the persistent result directory")
    partial = False
    for seed, method, member, path in expected:
        if path.exists():
            state = validate_checkpoint(path, config, signature, method, seed, member)
            partial = partial or state["epoch"] < config["epochs"]
    old_manifest = output / "manifest.json"
    if partial and old_manifest.exists():
        old = json.loads(old_manifest.read_text(encoding="utf-8"))["environment"]
        current = {"torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}
        if any(old.get(key) != value for key, value in current.items()):
            raise RuntimeError("Partial epoch-boundary resume requires the same PyTorch/CUDA/GPU environment; existing files are preserved")
    events = []
    engine.train_member = make_train_member(engine.train_member, output, True, events)
    sys.argv = [str(ROOT / "run_experiment.py"), "--config", str(ROOT / "configs/local.json"),
                "--output", str(output), "--data-dir", str(args.data_dir.resolve()), "--device", "cuda"]
    runpy.run_path(str(ROOT / "run_experiment.py"), run_name="__main__")
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    if manifest["result_rows"] != 63 or manifest["trained_models"] != 48:
        raise AssertionError("Final experiment did not produce exactly 48 models and 63 result rows")
    manifest["colab_resume_summary"] = {action: sum(row["action"] == action for row in events)
                                         for action in ("verified_complete_skipped", "resumed_partial", "trained_new")}
    manifest["checkpoint_validation"] = "strict architecture/config/signature/history/finite weights/optimizer/RNG"
    engine.atomic_json(output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
