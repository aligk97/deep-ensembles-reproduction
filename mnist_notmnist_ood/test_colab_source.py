"""Colab-source policy tests with explicitly synthetic fixtures, never trained results."""
from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch

import colab_source
from colab_source import inspect_result, expected_models, sha256, extract_mnist_zip, zip_source
from colab_runtime import pack_outputs
from inference import load_model, original_api

HERE = Path(__file__).resolve().parent


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def metadata_fixture(root, reference):
    """Dummy checkpoint BYTES test file/hash policy; not loadable/trained models."""
    result = Path(root) / "mnist_classification/outputs/local"
    result.mkdir(parents=True)
    config = reference["mnist_config"]
    manifest = {"status": "complete", "trained_models": 48, "result_rows": 63, "config": config,
                "signature": reference["mnist_signature"], "source_sha256": reference["mnist_source_sha256"],
                "environment": {"device": "cuda", "gpu": "SYNTHETIC_TEST_T4_LABEL_NOT_MEASURED", "platform": "Linux_FIXTURE"}}
    write_json(result / "manifest.json", manifest)
    write_json(result / "config.json", config)
    events = []
    for seed, method, member in expected_models(config):
        path = result / "checkpoints" / f"seed_{seed}" / method / f"member_{member:02d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"UNTRAINED_METADATA_FIXTURE {seed} {method} {member}".encode())
        events.append({"seed": seed, "method": method, "member": member, "epoch": 10,
                       "action": "trained_new", "sha256": sha256(path)})
    write_json(result / "colab_resume_audit.json", events)
    rows = []
    for seed in config["seeds"]:
        for method in colab_source.METHODS:
            for m in range(1, 6):
                row = {"run_seed": seed, "method": method, "M": m, "n_test": 10000,
                       "accuracy": .9, "classification_error": .1, "nll": .2, "brier": .01}
                rows.append(row)
                if method == "ensemble" and m == 1:
                    rows.append({**row, "method": "single"})
    with (result / "raw_results.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    (result / "summary.csv").write_text("method,M,n_seeds\nensemble,1,3\n", encoding="utf-8")
    attestation = {"kind": "drive_session", "run_id": "SYNTHETIC_FIXTURE", "session": {
        "status": "complete", "real_colab_training_completed": True,
        "scientific_signature": reference["mnist_signature"], "config": config,
        "fixture_only_not_a_real_session": True}}
    return result, attestation


class SourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = json.loads((HERE / "colab_reference.json").read_text(encoding="utf-8"))
        torch.set_num_threads(2)

    def test_all_48_independent_hash_anchors_cover_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, attestation = metadata_fixture(tmp, self.reference)
            source = inspect_result(result, attestation, self.reference)
            self.assertEqual(len(source["inventory"]), 48)
            self.assertEqual(sum(row["method"] == "mc_dropout" for row in source["inventory"]), 3)

    def test_windows_source_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, attestation = metadata_fixture(tmp, self.reference)
            manifest = json.loads((result / "manifest.json").read_text())
            manifest["environment"]["platform"] = "Windows-11"
            write_json(result / "manifest.json", manifest)
            with self.assertRaisesRegex(ValueError, "Windows/local"):
                inspect_result(result, attestation, self.reference)

    def test_missing_checkpoint_and_hash_tamper_rejected(self):
        for kind in ("missing", "changed"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                result, attestation = metadata_fixture(tmp, self.reference)
                target = result / "checkpoints/seed_42/ensemble/member_00.pt"
                if kind == "missing":
                    target.unlink()
                else:
                    target.write_bytes(b"tampered")
                with self.assertRaisesRegex(ValueError, "checkpoint|SHA-256"):
                    inspect_result(result, attestation, self.reference)

    def test_wrong_manifest_and_incomplete_source_rejected(self):
        for key, value in (("signature", "wrong"), ("status", "running"), ("trained_models", 47), ("result_rows", 62)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                result, attestation = metadata_fixture(tmp, self.reference)
                manifest = json.loads((result / "manifest.json").read_text())
                manifest[key] = value
                write_json(result / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    inspect_result(result, attestation, self.reference)

    def test_missing_or_duplicate_audit_anchor_rejected(self):
        for duplicate in (False, True):
            with tempfile.TemporaryDirectory() as tmp:
                result, attestation = metadata_fixture(tmp, self.reference)
                events = json.loads((result / "colab_resume_audit.json").read_text())
                events.pop()
                if duplicate: events.append(events[0])
                write_json(result / "colab_resume_audit.json", events)
                with self.assertRaises(ValueError): inspect_result(result, attestation, self.reference)

    def test_no_real_colab_attestation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, attestation = metadata_fixture(tmp, self.reference)
            attestation["session"]["real_colab_training_completed"] = False
            with self.assertRaisesRegex(ValueError, "attest"):
                inspect_result(result, attestation, self.reference)

    def test_explicit_missing_drive_run_never_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(colab_source, "DRIVE_CLASSIFICATION", Path(tmp)):
            with self.assertRaisesRegex(ValueError, "No fallback"):
                colab_source.drive_source("missing_source", self.reference)

    def test_complete_zip_layout_and_hash_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result, _ = metadata_fixture(root / "source", self.reference)
            archive = root / "source.zip"
            with zipfile.ZipFile(archive, "w") as package:
                for path in (root / "source").rglob("*"):
                    if path.is_file(): package.write(path, Path("SYNTHETIC_FIXTURE") / path.relative_to(root / "source"))
                package.writestr("SYNTHETIC_FIXTURE/export_info.json", json.dumps({"mode": "real_colab_completed", "checkpoints": 48, "raw_rows": 63}))
            selected = zip_source(archive, root / "imports", self.reference)
            self.assertEqual(len(selected["inventory"]), 48)
            self.assertEqual(selected["attestation"]["archive_sha256"], sha256(archive))

    def test_unsafe_and_local_validation_zips_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("../escape.pt", "/escape.pt", "C:\\escape.pt", "safe/link"):
                archive = root / "bad.zip"
                with zipfile.ZipFile(archive, "w") as package:
                    if name == "safe/link":
                        member = zipfile.ZipInfo(name); member.create_system = 3
                        member.external_attr = (stat.S_IFLNK | 0o777) << 16
                        package.writestr(member, "../../escape.pt")
                    else: package.writestr(name, b"x")
                with self.assertRaises(ValueError): extract_mnist_zip(archive, root / "output")
            with zipfile.ZipFile(root / "local.zip", "w") as package:
                package.writestr("outer/export_info.json", json.dumps({"mode": "LOCAL_FORMAT_FIXTURE_NOT_COLAB_RESULTS", "checkpoints": 48, "raw_rows": 63}))
            with self.assertRaisesRegex(ValueError, "local-validation"):
                extract_mnist_zip(root / "local.zip", root / "local_output")

    def test_figure2_reference_must_be_matching_complete_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            result, attestation = metadata_fixture(tmp, self.reference)
            (result / "raw_results.csv").write_text("run_seed,method,M,n_test,accuracy,classification_error,nll,brier\n42,ensemble,1,10000,.9,.1,.2,.01\n")
            with self.assertRaisesRegex(ValueError, "63-row"):
                inspect_result(result, attestation, self.reference)

    def test_original_strict_loader_and_dropout_modes_with_untrained_fixture(self):
        config = self.reference["mnist_config"]
        api = original_api()
        for method in ("ensemble", "mc_dropout"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "private_untrained_format_fixture.pt"
                model = api.MLP(.1 if method == "mc_dropout" else 0)
                state = {"config": config, "signature": self.reference["mnist_signature"], "epoch": 10,
                         "member_seed": api.member_seed(42, 0, method), "model": model.state_dict(),
                         "history": [{"run_seed": 42, "method": method, "member": 0, "epoch": 10}],
                         "synthetic_untrained_fixture_only": True}
                torch.save(state, path)
                record = {"path": str(path), "run_seed": 42, "method": method, "member": 0}
                manifest = {"signature": self.reference["mnist_signature"], "config": config}
                restored = load_model(record, manifest, torch.device("cpu"))
                self.assertTrue(all(not layer.training for layer in restored.modules() if isinstance(layer, torch.nn.BatchNorm1d)))
                drops = [layer for layer in restored.modules() if isinstance(layer, torch.nn.Dropout)]
                self.assertEqual(len(drops), 3 if method == "mc_dropout" else 0)
                state["model"].pop(next(iter(state["model"])))
                torch.save(state, path)
                with self.assertRaises(RuntimeError): load_model(record, manifest, torch.device("cpu"))

    def test_output_zip_preserves_cache_files_and_unique_outer_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); project = root / "project"; output = root / "output"
            source = project / "mnist_notmnist_ood/data.py"
            source.parent.mkdir(parents=True); source.write_bytes(b"source fixture")
            write_json(project / "bundle_manifest.json", {"mnist_notmnist_ood/data.py": sha256(source)})
            for name in ("results_per_seed.csv", "experiment_manifest.json", "entropy/e.npy", "probabilities/p.npz", "figures/f.png", "figures/f.pdf"):
                path = output / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"FORMAT_ONLY_FIXTURE")
            report = pack_outputs(project, output, root / "result.zip", "SYNTHETIC_OUTPUT_FORMAT_FIXTURE")
            with zipfile.ZipFile(report["path"]) as package:
                names = package.namelist()
                self.assertTrue(all(name.startswith("SYNTHETIC_OUTPUT_FORMAT_FIXTURE/") for name in names))
                self.assertIn("SYNTHETIC_OUTPUT_FORMAT_FIXTURE/mnist_notmnist_ood/outputs/probabilities/p.npz", names)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    def forbidden(*a, **k): raise RuntimeError("No optimizer steps in OOD preparation")
    torch.optim.Adam.step = forbidden
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(SourceTests))
    report = {"successful": result.wasSuccessful(), "tests_run": result.testsRun,
              "failed": len(result.failures), "errors": len(result.errors), "optimizer_steps": 0,
              "real_colab_test": False, "real_colab_checkpoint_sources_tested": 0,
              "scope": "Explicitly synthetic metadata/untrained model/ZIP format fixtures only; no local trained checkpoint fallback",
              "failures": [{"test": str(t), "traceback": msg} for t, msg in result.failures],
              "error_details": [{"test": str(t), "traceback": msg} for t, msg in result.errors]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__": main()
