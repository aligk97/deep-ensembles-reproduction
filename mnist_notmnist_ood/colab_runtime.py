"""Colab-only source/data orchestration and results export; training always forbidden."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

from colab_source import drive_source, zip_source, install_bridge, verify_source_unchanged, sha256
from data import HERE, write_json, EXPECTED_SHA256

DRIVE_OOD = Path("/content/drive/MyDrive/DeepEnsembles_MNIST_NotMNIST_OOD")


def in_colab():
    try:
        return importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def pack_outputs(project, output, destination, outer):
    """Byte-preserving packaging only; callers establish real-run completion first."""
    project, output, destination = map(Path, (project, output, destination))
    if not re.fullmatch(r"[A-Za-z0-9_-]+", outer) or destination.resolve().is_relative_to(output.resolve()):
        raise ValueError("ZIP must have a safe outer name and stay outside results")
    assets = json.loads((project / "bundle_manifest.json").read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as package:
        for relative, expected in assets.items():
            path = project / relative
            if sha256(path) != expected:
                raise ValueError(f"Source bundle changed before ZIP export: {relative}")
            package.write(path, Path(outer) / relative)
        package.write(project / "bundle_manifest.json", Path(outer) / "bundle_manifest.json")
        for path in sorted(output.rglob("*")):
            if path.is_file():
                package.write(path, Path(outer) / "mnist_notmnist_ood/outputs" / path.relative_to(output))
    temporary.replace(destination)
    with zipfile.ZipFile(destination) as package:
        if package.testzip() is not None or not all(name.startswith(outer + "/") for name in package.namelist()):
            raise AssertionError("OOD result ZIP CRC/path verification failed")
    return {"path": str(destination), "sha256": sha256(destination)}


class ColabOOD:
    def __init__(self, project, persistent, source_mode, mnist_run_id, zip_path, execute_colab):
        self.project = Path(project).resolve()
        self.here = self.project / "mnist_notmnist_ood"
        self.persistent = Path(persistent).resolve()
        self.execute_colab = bool(execute_colab)
        if self.execute_colab and (not in_colab() or os.environ.get("OOD_COLAB_VALIDATE_ONLY") == "1"):
            raise RuntimeError("Actual checkpoint inference is blocked outside real Colab")
        if self.execute_colab and not self.persistent.is_relative_to(DRIVE_OOD):
            raise ValueError("Use the separate OOD Drive directory")
        if source_mode not in ("drive", "zip"):
            raise ValueError("Choose SOURCE_MODE='drive' or 'zip'; sources are never guessed")
        self.source_mode, self.mnist_run_id, self.zip_path = source_mode, mnist_run_id, Path(zip_path)
        self.persistent.mkdir(parents=True, exist_ok=True)
        self.output = self.persistent / "outputs"
        self.output.mkdir(exist_ok=True)
        self.reference = json.loads((self.here / "colab_reference.json").read_text(encoding="utf-8"))
        self.source = None
        self.report = {"status": "prepared", "real_colab_inference_completed": False, "new_training": False,
                       "source_mode": source_mode, "mnist_run_id": mnist_run_id,
                       "execute_colab": self.execute_colab, "local_model_fallback": False}
        self._save()

    def _save(self):
        write_json(self.persistent / "colab_session.json", self.report)

    def connect_source(self):
        if not self.execute_colab:
            self.report["source_validation"] = "not_attempted_locally; policy tested with explicitly synthetic fixtures"
            self._save()
            print("LOCAL PREPARATION: no Drive/ZIP checkpoint source or Windows model is loaded", flush=True)
            return
        selected = (drive_source(self.mnist_run_id, self.reference) if self.source_mode == "drive" else
                    zip_source(self.zip_path, Path("/content/mnist_ood_zip_imports"), self.reference))
        install_bridge(self.project / "mnist_classification", selected)
        from inference import load_config, checkpoint_inventory, load_model
        import torch
        config = load_config()
        inventory, manifest = checkpoint_inventory(config)
        if inventory != selected["inventory"]:
            raise ValueError("Checkpoint bridge resolved to an unexpected source")
        for record in inventory:
            try:
                model = load_model(record, manifest, torch.device("cpu"))
                if any(not torch.isfinite(value).all() for value in model.state_dict().values()):
                    raise ValueError("Non-finite checkpoint model state")
                del model
            except Exception as exc:
                raise ValueError(f"Invalid Colab model seed={record['run_seed']} method={record['method']} member={record['member']}: {exc}") from exc
        self.source = selected
        write_json(self.output / "checkpoint_source.json", selected)
        reference_dir = self.output / "classifier_reference"
        reference_dir.mkdir(exist_ok=True)
        for name in ("manifest.json", "config.json", "raw_results.csv", "summary.csv", "colab_resume_audit.json"):
            path = Path(selected["result_root"]) / name
            if not path.is_file():
                raise ValueError(f"Required Colab Figure 2(a) reference is missing: {path}")
            shutil.copyfile(path, reference_dir / name)
        self.report.update(source_validation="48 hashes + real Colab provenance + strict CPU model loads passed",
                           checkpoint_source_manifest_sha256=selected["manifest_sha256"])
        self._save()
        print("Verified real Colab source: 48 checkpoints; no local fallback; no training", flush=True)

    def run_tests(self):
        environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8", OOD_COLAB_VALIDATE_ONLY="1")
        for script, output, extra in (
            ("test_colab_source.py", self.here / "preflight_source_results.json", []),
            ("test_metrics.py", self.here / "preflight_metrics_results.json", []),
        ):
            subprocess.run([sys.executable, "-B", "-u", str(self.here / script), "--output", str(output), *extra],
                           cwd=self.here, env=environment, check=True)
            info = json.loads(output.read_text(encoding="utf-8"))
            if not info["successful"]:
                raise AssertionError(f"Preflight tests failed: {script}")
            shutil.copyfile(output, self.output / ("colab_source_test_results.json" if script.startswith("test_colab") else "metrics_test_results.json"))
            self.report[script] = info
        self._save()

    def prepare_data(self, local_ood_cache=None, local_mnist_cache=None):
        from data import download_notmnist, ensure_mnist_test, MNIST_FILES
        from prepare_data import prepare
        cache = self.persistent / "dataset_archives"
        cache.mkdir(exist_ok=True)
        if not self.execute_colab:
            if local_ood_cache is None or local_mnist_cache is None:
                raise RuntimeError("Local compatibility requires explicit data-only caches; never guesses checkpoint sources")
            for name in ("notMNIST_small.tar.gz", "download_provenance.json"):
                shutil.copyfile(Path(local_ood_cache) / name, cache / name)
        archive, _ = download_notmnist(cache)
        if sha256(archive) != self.reference["notmnist_archive_sha256"] or sha256(archive) != EXPECTED_SHA256:
            raise ValueError("NotMNIST small archive does not match the validated local version")
        data_root = self.here / "data"
        data_root.mkdir(exist_ok=True)
        for name in ("notMNIST_small.tar.gz", "download_provenance.json"):
            shutil.copyfile(cache / name, data_root / name)
        mnist = ensure_mnist_test(Path(local_mnist_cache) if not self.execute_colab else None, cache / "mnist_test")
        # Always copy data-only test IDX files into the fresh bridge workspace;
        # never write into the selected classifier's checkpoint/result tree.
        bridge_data = self.project / "mnist_classification/data"
        bridge_data.mkdir(exist_ok=True)
        for name in MNIST_FILES:
            shutil.copyfile(mnist / name, bridge_data / name)
        info = prepare(data_root, self.output, bridge_data, batch_size=self.reference["ood_config"]["batch_size"])
        if info["notmnist"]["valid_images"] != 18724 or info["mnist_test"]["valid_images"] != 10000:
            raise AssertionError("The full prepared dataset must be used")
        if [row["path"] for row in info["notmnist"]["corrupt_files"]] != self.reference["corrupt_files"]:
            raise AssertionError("Corrupt-image elimination changed")
        subprocess.run([sys.executable, "-B", "-u", str(self.here / "test_data.py"),
                        "--data-root", str(data_root), "--output-root", str(self.output)],
                       cwd=self.here, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"), check=True)
        from inference import load_config
        from colab_data import load_reference_datasets
        datasets = load_reference_datasets(load_config(), self.reference, self.output)
        del datasets
        subprocess.run([sys.executable, "-B", "-u", str(self.here / "test_colab_order.py"),
                        "--data-root", str(data_root / "notMNIST_small"),
                        "--output", str(self.output / "colab_order_test_results.json")],
                       cwd=self.here, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"), check=True)
        self.report.update(data_validated=True, mnist_examples=10000, notmnist_examples=18724,
                           notmnist_archive_sha256=sha256(archive), local_data_downloads=0 if not self.execute_colab else None,
                           dataset_tensor_hashes_match_local=True)
        self._save()

    def run_inference(self):
        if not self.report.get("data_validated"):
            raise RuntimeError("Prepare and validate datasets first")
        if not self.execute_colab:
            self.report.update(status="local_compatibility_only_no_checkpoint_inference", checkpoint_model_forwards=0)
            self._save()
            print("LOCAL PREPARATION: real smoke/full checkpoint inference intentionally not executed", flush=True)
            return
        if self.source is None:
            raise RuntimeError("No verified real Colab checkpoint source")
        command = [sys.executable, "-B", "-u", str(self.here / "colab_entrypoint.py"),
                   "--output", str(self.output), "--source-info", str(self.output / "checkpoint_source.json")]
        start = time.perf_counter()
        try:
            with (self.persistent / "inference_progress.log").open("a", encoding="utf-8") as log:
                with subprocess.Popen(command, cwd=self.here, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONIOENCODING="utf-8"),
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace") as process:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line); log.flush()
                    code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            manifest = json.loads((self.output / "experiment_manifest.json").read_text(encoding="utf-8"))
            if not manifest.get("real_colab_inference_completed") or manifest["status"] != "complete":
                raise AssertionError("OOD evaluation did not complete")
            self.report.update(status="complete", real_colab_inference_completed=True,
                               invocation_seconds=time.perf_counter()-start, new_training=False)
        except BaseException as exc:
            self.report.update(status="failed_or_interrupted", error=repr(exc))
            self._save()
            raise
        self._save()

    def export(self):
        if not self.execute_colab:
            print("LOCAL PREPARATION: no completed Colab result ZIP is claimed", flush=True)
            return None
        if not self.report["real_colab_inference_completed"]:
            raise RuntimeError("Cannot export incomplete OOD results")
        manifest = json.loads((self.output / "experiment_manifest.json").read_text(encoding="utf-8"))
        if manifest["result_rows"] != 36 or manifest["summary_rows"] != 12:
            raise AssertionError("Incomplete result grid")
        if len(list((self.output / "probabilities").glob("*.npz"))) != 24 or len(list((self.output / "entropy").glob("*.npy"))) != 72:
            raise AssertionError("Probability or per-image entropy caches are missing")
        verify_source_unchanged(self.source)
        outer = "mnist_notmnist_ood_colab_" + self.persistent.name
        path = self.persistent / f"{outer}.zip"
        self.report["export"] = pack_outputs(self.project, self.output, path, outer)
        self._save()
        return path
