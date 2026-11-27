"""Drive-backed orchestration and complete OOD-compatible export; no training math."""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np

from colab_checkpoint import load_reference, checkpoint_records, validate_checkpoint, sha256
from engine import atomic_json

DRIVE_BASE = Path("/content/drive/MyDrive/DeepEnsembles_MNIST_Figure2a")


def in_colab():
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ModuleNotFoundError, ImportError, ValueError):
        return False


def read_only_audit(source, result, data_dir):
    """Reuse original artifact auditor without writing to the inspected result."""
    import verify_artifacts
    config, signature = load_reference(source)
    result, data_dir = Path(result), Path(data_dir)
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["status"] != "complete" or manifest["signature"] != signature
            or manifest["config"] != config or manifest["trained_models"] != 48 or manifest["result_rows"] != 63):
        raise ValueError("Expected the final completed 48-model/63-row experiment")
    expected = checkpoint_records(result, config)
    if {p.resolve() for p in (result / "checkpoints").rglob("*.pt")} != {p.resolve() for _, _, _, p in expected}:
        raise ValueError("Checkpoint grid is not exactly the authorized 48 models")
    before = verify_artifacts.atomic_json
    captured = {}
    try:
        def capture(path, value):
            if Path(path) != result / "verification.json":
                raise AssertionError("Unexpected write from original auditor")
            captured.update(value)
        verify_artifacts.atomic_json = capture
        with contextlib.redirect_stdout(io.StringIO()):
            verify_artifacts.audit(result, data_dir)
    finally:
        verify_artifacts.atomic_json = before
    for seed, method, member, path in expected:
        validate_checkpoint(path, config, signature, method, seed, member, require_complete=True)
    if captured.get("checkpoints") != 48 or captured.get("raw_rows") != 63:
        raise AssertionError("Incomplete export audit")
    return {**captured, "strict_checkpoint_loads": 48, "read_only": True}


def export_bundle(source, result, data_dir, destination, outer, mode):
    """Preserve original checkpoint format and canonical mnist_classification layout."""
    source, result, data_dir, destination = map(Path, (source, result, data_dir, destination))
    if destination.resolve().is_relative_to(result.resolve()) or not re.fullmatch(r"[A-Za-z0-9_-]+", outer):
        raise ValueError("ZIP must be outside results with a safe unique outer folder")
    audit = read_only_audit(source, result, data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    root = Path(outer) / "mnist_classification"
    temporary = destination.with_suffix(".zip.tmp")
    assets = json.loads((source / "bundle_manifest.json").read_text(encoding="utf-8"))
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
        for relative in assets:
            path = source / relative
            if sha256(path) != assets[relative]:
                raise ValueError(f"Source/data bundle changed before export: {relative}")
            package.write(path, root / relative)
        package.write(source / "bundle_manifest.json", root / "bundle_manifest.json")
        for path in sorted(result.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                package.write(path, root / "outputs/local" / path.relative_to(result))
        for path in sorted(data_dir.glob("*.gz")):
            package.write(path, root / "data" / path.name)
        if (data_dir / "checksums.json").exists():
            package.write(data_dir / "checksums.json", root / "data/checksums.json")
        package.writestr(str(Path(outer) / "export_info.json"), json.dumps({
            "mode": mode, "checkpoints": 48, "raw_rows": 63,
            "checkpoint_root": "mnist_classification/outputs/local/checkpoints",
            "checkpoint_schema": "Original model/optimizer/history/RNG/config; no conversion",
            "read_only_audit": audit,
        }, indent=2))
    temporary.replace(destination)
    with zipfile.ZipFile(destination) as package:
        names = package.namelist()
        checkpoints = [name for name in names if "/outputs/local/checkpoints/" in name and name.endswith(".pt")]
        if len(checkpoints) != 48 or package.testzip() is not None or not all(name.startswith(outer + "/") for name in names):
            raise AssertionError("Archive CRC/layout/checkpoint verification failed")
    return {"path": str(destination), "sha256": sha256(destination), "checkpoint_files": 48,
            "raw_rows": 63, "archive_mode": mode, "outer_folder": outer}


class ColabRun:
    def __init__(self, work, persistent_root, allow_training):
        self.work = Path(work).resolve()
        self.persistent = Path(persistent_root).resolve()
        self.allow_training = bool(allow_training)
        if self.allow_training and (not in_colab() or os.environ.get("MNIST_COLAB_VALIDATE_ONLY") == "1"):
            raise RuntimeError("New training is forbidden during local preparation")
        if self.allow_training and not self.persistent.is_relative_to(DRIVE_BASE):
            raise ValueError("Use the dedicated persistent Drive run folder")
        self.persistent.mkdir(parents=True, exist_ok=True)
        self.result = self.persistent / "mnist_classification/outputs/local"
        self.data_dir = self.work / "data"
        self.config, self.signature = load_reference(self.work)
        self.report = {"status": "prepared", "scientific_signature": self.signature,
                       "config": self.config, "training_enabled": self.allow_training,
                       "real_colab_training_completed": False, "new_training_executed": False,
                       "result_directory": str(self.result)}
        session = self.persistent / "colab_session.json"
        if session.exists() and json.loads(session.read_text(encoding="utf-8"))["scientific_signature"] != self.signature:
            raise ValueError("Drive folder belongs to a different scientific experiment; use a new RUN_ID")
        self._save()

    def _save(self):
        atomic_json(self.persistent / "colab_session.json", self.report)

    def run_no_training_tests(self):
        output = self.work / "preflight_test_results.json"
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        subprocess.run([sys.executable, "-B", "-u", str(self.work / "test_colab_preflight.py"),
                        "--output", str(output)], cwd=self.work, env=environment, check=True)
        self.report["no_training_tests"] = json.loads(output.read_text(encoding="utf-8"))
        if not self.report["no_training_tests"]["successful"]:
            raise AssertionError("No-training preflight tests failed")
        self._save()

    def prepare_data(self, local_cache=None):
        import torch
        import data
        torch.set_num_threads(self.config["threads"])
        self.data_dir.mkdir(exist_ok=True)
        if not self.allow_training:
            if local_cache is None:
                raise RuntimeError("Local validation requires the existing read-only MNIST cache; it never downloads data")
            local_cache = Path(local_cache)
            for name, expected in data.FILES.items():
                if data.md5(local_cache / name) != expected:
                    raise ValueError("Existing MNIST cache failed MD5 validation")
                shutil.copyfile(local_cache / name, self.data_dir / name)
        output = self.work / "preflight_data"
        output.mkdir(exist_ok=True)
        tensors = data.prepare(self.data_dir, self.config, output, torch.device("cpu"))
        reference = json.loads((self.work / "final_reference.json").read_text(encoding="utf-8"))
        with np.load(output / "splits.npz") as split:
            for key, expected in reference["split_array_sha256"].items():
                if hashlib.sha256(split[key].tobytes()).hexdigest() != expected:
                    raise AssertionError(f"Train/validation/test split differs from completed local experiment: {key}")
        shapes = [list(tensor.shape) for tensor in tensors]
        if shapes[:6] != [[55000, 784], [55000], [5000, 784], [5000], [10000, 784], [10000]]:
            raise AssertionError("Preprocessing changed dataset sizes or input features")
        if any(not torch.isfinite(tensor).all() for tensor in tensors):
            raise AssertionError("Non-finite preprocessed values")
        if any(tensors[index].dtype != torch.float32 or tensors[index].min() < 0 or tensors[index].max() > 1
               for index in (0, 2, 4)):
            raise AssertionError("Expected float32 MNIST pixels scaled to [0,1]")
        self.report["preprocessing"] = {"shapes": shapes, "dtype": "float32", "range": [0, 1],
                                        "split_seed": 4509, "split_arrays_match_local_sha256": True,
                                        "official_data_md5": "passed", "local_downloads": 0 if not self.allow_training else None}
        del tensors
        self._save()
        print("Preprocessing passed: 55000 / 5000 / 10000; original split hashes; float32 [0,1] / 784", flush=True)

    def run_final(self):
        if not self.report.get("preprocessing") or not self.report.get("no_training_tests"):
            raise RuntimeError("Run no-training tests and preprocessing first")
        if not self.allow_training:
            self.report.update(status="local_compatibility_only", new_training_executed=False)
            self._save()
            print("LOCAL VALIDATION: final 48-model training intentionally not executed", flush=True)
            return
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        command = [sys.executable, "-B", "-u", str(self.work / "colab_entrypoint.py"),
                   "--output", str(self.result), "--data-dir", str(self.data_dir)]
        self.report["status"] = "running"
        self._save()
        start = time.perf_counter()
        try:
            with (self.persistent / "training_progress.log").open("a", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=self.work, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            audit = read_only_audit(self.work, self.result, self.data_dir)
            atomic_json(self.result / "colab_export_verification.json", audit)
            manifest = json.loads((self.result / "manifest.json").read_text(encoding="utf-8"))
            self.report.update(status="complete", real_colab_training_completed=True,
                               new_training_executed=manifest["colab_resume_summary"]["trained_new"] > 0
                               or manifest["colab_resume_summary"]["resumed_partial"] > 0,
                               checkpoints=48, result_rows=63, invocation_seconds=time.perf_counter()-start,
                               training_epoch_seconds=manifest["total_training_epoch_seconds"],
                               environment=manifest["environment"], resume_summary=manifest["colab_resume_summary"])
        except BaseException as exc:
            self.report.update(status="interrupted_or_failed", error=repr(exc), invocation_seconds=time.perf_counter()-start)
            self._save()
            raise
        self._save()

    def export(self):
        if not self.allow_training:
            print("LOCAL VALIDATION: no fabricated Colab checkpoint ZIP is produced", flush=True)
            return None
        if not self.report["real_colab_training_completed"]:
            raise RuntimeError("Only a completed verified real run may be exported")
        run_id = self.persistent.name
        outer = f"mnist_classification_colab_{run_id}"
        destination = self.persistent / f"{outer}.zip"
        self.report["export"] = export_bundle(self.work, self.result, self.data_dir, destination, outer, "real_colab_completed")
        self._save()
        return destination
