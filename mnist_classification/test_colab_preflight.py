"""No-training correctness tests; synthetic checkpoint fixtures are NOT models/results."""
from __future__ import annotations

import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import copy
import gzip
import json
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from colab_checkpoint import load_reference, checkpoint_records, validate_checkpoint, make_train_member, sha256
from data import read_idx
from engine import rng_state, member_seed, perturb
from model import MLP
from run_experiment import summarize
from test_correctness import Correctness

ROOT = Path(__file__).resolve().parent


def fixture(config, signature, method="ensemble", seed=42, member=0, epoch=10):
    """Serialize the expected FORMAT without a single gradient/optimizer step."""
    model = MLP(.1 if method == "mc_dropout" else 0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"]).state_dict()
    steps = epoch * 550
    for index, parameter in zip(optimizer["param_groups"][0]["params"], model.parameters()):
        optimizer["state"][index] = {"step": torch.tensor(float(steps)), "exp_avg": torch.zeros_like(parameter),
                                      "exp_avg_sq": torch.zeros_like(parameter)}
    history = [{"epoch": e, "run_seed": seed, "method": method, "member": member,
                "member_seed": member_seed(seed, member, method), "objective_loss": 0., "clean_loss": 0.,
                "perturbed_loss": 0., "train_accuracy": 0., "validation_loss": 0.,
                "validation_accuracy": 0., "epoch_seconds": 0.} for e in range(1, epoch + 1)]
    return {"signature": signature, "config": config, "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer, "history": history, "member_seed": member_seed(seed, member, method),
            "rng": rng_state(torch.Generator().manual_seed(4509)),
            "synthetic_untrained_format_fixture_not_experiment_result": True}


class Preflight(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config, cls.signature = load_reference(ROOT)

    def test_final_config_and_48_checkpoint_paths(self):
        self.assertEqual(self.config["seeds"], [42, 123, 2026])
        self.assertEqual(self.config["ensemble_sizes"], [1, 2, 3, 4, 5])
        self.assertEqual(self.config["learning_rate"], .1)
        self.assertEqual(self.config["batch_size"], 100)
        self.assertEqual(self.config["split_seed"], 4509)
        records = checkpoint_records(Path("unused"), self.config)
        self.assertEqual(len(records), 48)
        self.assertEqual(sum(method == "mc_dropout" for _, method, _, _ in records), 3)

    def test_checkpoint_format_strict_load_all_four_architectures(self):
        with tempfile.TemporaryDirectory() as tmp:
            for method in ("ensemble", "random", "adversarial", "mc_dropout"):
                path = Path(tmp) / f"{method}.pt"
                torch.save(fixture(self.config, self.signature, method), path)
                restored = validate_checkpoint(path, self.config, self.signature, method, 42, 0, True)
                self.assertEqual(restored["epoch"], 10)

    def test_invalid_checkpoints_fail_instead_of_retraining(self):
        state = fixture(self.config, self.signature)
        cases = []
        bad = copy.deepcopy(state); bad["signature"] = "wrong"; cases.append(bad)
        bad = copy.deepcopy(state); bad["config"]["learning_rate"] = .001; cases.append(bad)
        bad = copy.deepcopy(state); bad["epoch"] = 11; cases.append(bad)
        bad = copy.deepcopy(state); bad["history"].pop(); cases.append(bad)
        bad = copy.deepcopy(state); bad["model"].pop(next(iter(bad["model"]))); cases.append(bad)
        bad = copy.deepcopy(state); bad["model"]["network.0.weight"][0, 0] = float("nan"); cases.append(bad)
        bad = copy.deepcopy(state); bad["optimizer"]["state"][0]["step"] = torch.tensor(1.); cases.append(bad)
        bad = copy.deepcopy(state); bad["rng"]["shuffle"] = torch.tensor([1]); cases.append(bad)
        with tempfile.TemporaryDirectory() as tmp:
            for index, broken in enumerate(cases):
                with self.subTest(case=index):
                    path = Path(tmp) / "broken.pt"
                    torch.save(broken, path)
                    with self.assertRaises((ValueError, RuntimeError, TypeError, KeyError)):
                        validate_checkpoint(path, self.config, self.signature, "ensemble", 42, 0)

    def test_complete_checkpoint_skipped_and_bytes_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            path = output / "checkpoints/seed_42/ensemble/member_00.pt"
            path.parent.mkdir(parents=True)
            torch.save(fixture(self.config, self.signature), path)
            before = sha256(path)
            def forbidden(*args, **kwargs):
                raise AssertionError("Original training must not be called")
            events = []
            guarded = make_train_member(forbidden, output, False, events)
            result = guarded(self.config, self.signature, output, "ensemble", 42, 0,
                             None, None, None, None, None)
            self.assertEqual(result, path)
            self.assertEqual(before, sha256(path))
            self.assertEqual(events[0]["action"], "verified_complete_skipped")
            self.assertEqual(len(pd.read_csv(output / "histories/seed_42_ensemble_00.csv")), 10)

    def test_partial_checkpoint_validates_but_local_training_is_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            path = output / "checkpoints/seed_42/ensemble/member_00.pt"
            path.parent.mkdir(parents=True)
            torch.save(fixture(self.config, self.signature, epoch=5), path)
            state = validate_checkpoint(path, self.config, self.signature, "ensemble", 42, 0)
            self.assertEqual(state["epoch"], 5)
            guarded = make_train_member(lambda *a, **k: self.fail("Training called"), output, False, [])
            with self.assertRaisesRegex(RuntimeError, "prohibited"):
                guarded(self.config, self.signature, output, "ensemble", 42, 0, None, None, None, None, None)

    def test_original_summary_63_rows_21_configs_and_sample_sd(self):
        rows = []
        for index, seed in enumerate(self.config["seeds"]):
            for method in ("ensemble", "random", "adversarial", "mc_dropout"):
                for m in self.config["ensemble_sizes"]:
                    row = {"run_seed": seed, "method": method, "M": m, "n_test": 10000,
                           "accuracy": .8 + .01 * index, "classification_error": .2 - .01 * index,
                           "nll": .1 + .1 * index, "brier": .01 + .001 * index}
                    rows.append(row)
                    if method == "ensemble" and m == 1:
                        rows.append({**row, "method": "single"})
        with tempfile.TemporaryDirectory() as tmp:
            summarize(rows, Path(tmp))
            raw, summary = pd.read_csv(Path(tmp) / "raw_results.csv"), pd.read_csv(Path(tmp) / "summary.csv")
            self.assertEqual(len(raw), 63)
            self.assertEqual(len(summary), 21)
            self.assertTrue((summary.n_seeds == 3).all())
            np.testing.assert_allclose(summary.nll_std, .1, atol=1e-12)

    def test_idx_scaling_and_flattening_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "images.gz"
            pixels = np.zeros((2, 28, 28), dtype=np.uint8)
            pixels[1] = 255
            with gzip.open(path, "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 2, 28, 28) + pixels.tobytes())
            x = read_idx(path, True).float().div_(255)
            self.assertEqual(tuple(x.shape), (2, 784))
            self.assertEqual(x.dtype, torch.float32)
            self.assertEqual(float(x.min()), 0)
            self.assertEqual(float(x.max()), 1)

    def test_fgsm_and_random_perturbations_without_optimizer(self):
        x = torch.full((3, 784), .5)
        epsilon = torch.linspace(0, .01, 784)
        gradient = torch.linspace(-1, 1, 784).expand_as(x)
        adversarial = perturb(x, epsilon, "adversarial", gradient)
        random = perturb(x, epsilon, "random")
        self.assertTrue(torch.allclose(adversarial - x, epsilon * gradient.sign(), atol=3e-8))
        self.assertTrue(torch.allclose((random - x).abs(), epsilon.expand_as(x), atol=3e-8))
        self.assertFalse(adversarial.requires_grad)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Even accidental use of an existing training unit test must fail before updating weights.
    def no_step(*args, **kwargs):
        raise RuntimeError("Optimizer steps are prohibited during Colab preparation")
    torch.optim.Adam.step = no_step
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Preflight)
    for name in ("test_metrics_known_values", "test_probability_mixture_is_not_logit_average",
                 "test_dropout_stochastic_bn_frozen"):
        suite.addTest(Correctness(name))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"successful": result.wasSuccessful(), "tests_run": result.testsRun,
              "failed": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
              "optimizer_steps": 0, "new_training": False, "real_colab_test": False,
              "checkpoint_fixtures": "Synthetic untrained private format fixtures; never used as experiment results",
              "training_dependent_old_tests_not_run": ["test_fgsm_random_and_gradient_update", "test_exact_resume_all_methods_and_mismatch"],
              "failures": [{"test": str(test), "traceback": message} for test, message in result.failures],
              "error_details": [{"test": str(test), "traceback": message} for test, message in result.errors]}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
