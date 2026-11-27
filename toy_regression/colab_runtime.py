"""Colab orchestration only; experiment implementations/configurations stay in 01..08.

The notebook extracts byte-identical source copies into a NEW temporary run.
No training is permitted outside Colab or when validation-only mode is set.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import runpy
import subprocess
import sys
import time
import zipfile

SCRIPTS = (
    "01_generate_data.py", "02_train_mse.py", "03_train_mse_ensemble.py",
    "04_train_gaussian_nll.py", "05_train_gaussian_nll_at.py", "06_train_deep_ensemble.py",
    "07_multi_seed_experiments.py", "08_diagnose_uncertainty.py",
)
TRAINING_SCRIPTS = SCRIPTS[1:7]


def in_colab():
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class ColabRun:
    def __init__(self, run_root, source_manifest, allow_training=False):
        self.run_root = Path(run_root).resolve()
        self.work = self.run_root / "toy_regression"
        if not self.work.is_dir() or self.work.is_symlink():
            raise ValueError("Expected a freshly extracted sandbox")
        if allow_training and (not in_colab() or os.environ.get("TOY_COLAB_VALIDATE_ONLY") == "1"):
            raise RuntimeError("Training is enabled only in Colab, never in local validation")
        self.allow_training = bool(allow_training)
        self.source_manifest = source_manifest
        self.report = {"mode": "colab_training" if allow_training else "local_validation_only",
                       "training_enabled": self.allow_training, "steps": [],
                       "source_manifest": source_manifest, "run_root": str(self.run_root),
                       "status": "prepared", "real_colab_training_completed": False}
        self.started = time.perf_counter()
        self.logs = self.run_root / "logs"
        self.logs.mkdir(exist_ok=True)
        self._save()

    def _save(self):
        write_json(self.run_root / "run_manifest.json", self.report)

    def preflight(self):
        import torch
        import numpy as np
        import matplotlib
        if sys.version_info < (3, 10):
            raise RuntimeError("Python 3.10+ required by existing type annotations")
        if self.allow_training and not torch.cuda.is_available():
            raise RuntimeError("Select a Colab GPU runtime before Run all; GPU is not currently available")
        modules, checks = {}, []
        for name, expected in self.source_manifest.items():
            path = self.work / name
            if not path.resolve().is_relative_to(self.work) or sha256(path) != expected:
                raise AssertionError(f"Embedded source/data hash mismatch: {name}")
        for name in SCRIPTS:
            # __name__ is not '__main__': only imports/constants/definitions run.
            namespace = runpy.run_path(str(self.work / name), run_name=f"_colab_preflight_{name[:2]}")
            modules[name] = namespace
            for key in ("ROOT", "OUT", "OUTPUT_DIR", "INPUT_DIR", "DEFAULT_OUT", "DATA_FILE", "DATA_CSV", "INPUT_CSV"):
                value = namespace.get(key)
                if isinstance(value, Path) and not value.resolve().is_relative_to(self.work):
                    raise AssertionError(f"Output/input path escapes sandbox: {name}/{key}")
            checks.append({"script": name, "imported_without_main": True})
        step07 = modules[SCRIPTS[6]]
        if (tuple(step07["RUN_SEEDS"]) != (42, 123, 2026) or step07["EPOCHS_DEFAULT"] != 3000
                or step07["NUM_ENSEMBLE_MEMBERS"] != 5 or step07["MSE_LR"] != .01
                or step07["NLL_LR"] != .001 or step07["HIDDEN"] != 50
                or step07["TEST_SEED"] != 20261122):
            raise AssertionError("Unexpected experiment configuration")
        if any(modules[name]["EPOCHS"] != 3000 for name in SCRIPTS[1:6]):
            raise AssertionError("An introductory experiment changed epoch count")
        if modules[SCRIPTS[4]]["NLL_BASELINE_PRED_CSV"].name != "gaussian_nll_single_predictions.csv":
            raise AssertionError("Step 05 must consume the file actually produced by Step 04")
        if modules[SCRIPTS[5]]["PRIOR_PANELS"][1][0] != "gaussian_nll_single_predictions.csv":
            raise AssertionError("Step 06 must consume the file actually produced by Step 04")
        data = self.work / "outputs/toy_training_data.csv"
        x, y, data_hash = step07["read_train"](data)
        if len(x) != 20 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise AssertionError("Expected the original 20 fixed finite training examples")
        self.report.update(status="preflight_passed", import_checks=checks,
                           dataset_sha256=data_hash, training_samples=20,
                           scientific_config={"run_seeds": [42, 123, 2026], "epochs": 3000,
                                              "members": 5, "hidden": 50, "mse_lr": .01, "nll_lr": .001,
                                              "eval_seed": 20261122, "legacy_model_seeds": [123, 124, 125, 126, 127]},
                           environment={"python": platform.python_version(), "torch": torch.__version__,
                                        "numpy": np.__version__, "matplotlib": matplotlib.__version__,
                                        "platform": platform.platform(), "cuda": torch.version.cuda,
                                        "gpu_available": torch.cuda.is_available(),
                                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
        self._save()
        print(json.dumps(self.report["environment"], indent=2))
        print(f"Preflight passed: 8 imports; original dataset SHA-256 {data_hash}")
        print("Mode:", self.report["mode"], "| isolated output:", self.work / "outputs")
        return self.report

    def run_step(self, name):
        if name not in SCRIPTS[1:]:
            raise ValueError("Only existing steps 02..08 may be dispatched; input data are fixed")
        if not self.report.get("import_checks"):
            raise RuntimeError("Run preflight first")
        path = self.work / name
        if sha256(path) != self.source_manifest[name]:
            raise AssertionError("Experiment source changed after preflight")
        command = [sys.executable, "-B", "-u", str(path)]
        record = {"script": name, "command": command, "cwd": str(self.work)}
        if not self.allow_training:
            record.update(status="not_executed_local_validation", seconds=0.0)
            self.report["steps"].append(record)
            self._save()
            print(f"LOCAL VALIDATION: {name} not executed; no training or fabricated results")
            return record
        if not in_colab() or os.environ.get("TOY_COLAB_VALIDATE_ONLY") == "1":
            raise RuntimeError("Local training gate rejected execution")
        environment = dict(os.environ, CUBLAS_WORKSPACE_CONFIG=":4096:8", MPLBACKEND="Agg",
                           PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8")
        started = time.perf_counter()
        log_path = self.logs / f"{name}.log"
        try:
            # shell=False and the kernel's Python: no Windows .venv/shell commands.
            with log_path.open("w", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=self.work, env=environment, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            record.update(status="completed", seconds=time.perf_counter() - started, log=str(log_path))
        except BaseException as exc:
            record.update(status="failed", error=repr(exc), seconds=time.perf_counter() - started)
            self.report["steps"].append(record)
            self.report["status"] = "failed"
            self._save()
            raise
        self.report["steps"].append(record)
        self._save()
        return record

    def finish(self):
        expected = list(SCRIPTS[1:])
        actual = [row["script"] for row in self.report["steps"]]
        if actual != expected:
            raise AssertionError("Execute notebook cells in order (Run all)")
        if self.allow_training:
            if any(row["status"] != "completed" for row in self.report["steps"]):
                raise AssertionError("A Colab experiment step did not complete")
            output = self.work / "outputs"
            required = ["toy_figure1_all_4_panels.png", "toy_figure1_panels_1_to_3.png",
                        "07_seed_experiments/aggregate/metrics/summary_mean_sd.csv",
                        "07_seed_experiments/aggregate/metrics/metrics_by_seed.csv",
                        "07_seed_experiments/aggregate/figures/summary_metrics.png",
                        "08_uncertainty_diagnostics/figures/left_ood_intervals.png"]
            for name in required:
                if not (output / name).is_file():
                    raise AssertionError(f"Run All output missing: {name}")
            with (output / required[2]).open(newline="", encoding="utf-8") as stream:
                summary = list(csv.DictReader(stream))
            with (output / required[3]).open(newline="", encoding="utf-8") as stream:
                by_seed = list(csv.DictReader(stream))
            if len(summary) != 16 or len(by_seed) != 48 or len(list(output.rglob("*.pt"))) != 49:
                raise AssertionError("Expected 16 region summaries, 48 seed/region rows, 49 total models")
            config = json.loads((output / "07_seed_experiments/experiment_config.json").read_text(encoding="utf-8"))
            if (config["run_seeds"] != [42, 123, 2026] or config["epochs_per_model"] != 3000
                    or config["data_sha256"] != self.report["dataset_sha256"]):
                raise AssertionError("Finished experiment configuration/data changed")
            self.report.update(status="colab_training_complete", real_colab_training_completed=True,
                               trained_model_files=49, seed_region_rows=48, region_summary_rows=16)
        else:
            self.report.update(status="local_compatibility_only_no_training", trained_model_files=0,
                               real_colab_training_completed=False)
        self.report["elapsed_seconds"] = time.perf_counter() - self.started
        self._save()
        archive = self.run_root / f"toy_regression_colab_{self.run_root.name}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as package:
            for folder in (self.work, self.logs):
                for path in sorted(folder.rglob("*")):
                    if path.is_file() and "__pycache__" not in path.parts:
                        package.write(path, Path(self.run_root.name) / path.relative_to(self.run_root))
            package.write(self.run_root / "run_manifest.json", f"{self.run_root.name}/run_manifest.json")
        print("Finished:", self.report["status"])
        print("Archive:", archive)
        return archive
