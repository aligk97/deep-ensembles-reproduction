"""Mathematical tests using synthetic probabilities only; no data/checkpoints."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import unittest
from pathlib import Path

import numpy as np
import torch

if __package__:
    from .metrics import (predictive_entropy, mean_id_entropy, mean_ood_entropy,
                          entropy_gap, ood_auroc, summarize_seed_metrics)
else:
    from metrics import (predictive_entropy, mean_id_entropy, mean_ood_entropy,
                         entropy_gap, ood_auroc, summarize_seed_metrics)

HERE = Path(__file__).resolve().parent


class MetricTests(unittest.TestCase):
    def setUp(self):
        self.evidence = {}

    def check(self, name, expected, calculated, atol=1e-12):
        if isinstance(calculated, torch.Tensor):
            calculated = calculated.detach().cpu().numpy()
        expected_array, calculated_array = np.asarray(expected), np.asarray(calculated)
        self.evidence[name] = {"expected": expected_array.tolist(),
                               "calculated": calculated_array.tolist(), "absolute_tolerance": atol}
        np.testing.assert_allclose(calculated_array, expected_array, rtol=0, atol=atol)

    def test_one_hot_entropy_zero(self):
        p = torch.eye(10, dtype=torch.float64).unsqueeze(0)
        self.check("entropy_nat", [0.0] * 10, predictive_entropy(p), atol=0)

    def test_uniform_ten_class_entropy_ln10(self):
        p = torch.full((3, 4, 10), 0.1, dtype=torch.float64)
        self.check("entropy_nat", [math.log(10)] * 4, predictive_entropy(p))

    def test_two_equal_classes_entropy_ln2(self):
        p = torch.zeros(1, 1, 10, dtype=torch.float64)
        p[0, 0, :2] = 0.5
        self.check("entropy_nat", [math.log(2)], predictive_entropy(p))

    def test_disagreeing_certain_members_entropy_ln2(self):
        p = torch.zeros(2, 1, 10, dtype=torch.float64)
        p[0, 0, 2], p[1, 0, 7] = 1, 1
        self.check("member_entropies_nat", [0, 0], [float(predictive_entropy(p[i:i+1])[0]) for i in range(2)])
        self.check("ensemble_entropy_nat", [math.log(2)], predictive_entropy(p))
        # The wrong mean of member entropies would be 0, not ln(2).

    def test_identical_members_preserve_entropy(self):
        p = torch.tensor([[[0.7, 0.2, 0.1] + [0.0] * 7]], dtype=torch.float64)
        expected = -(0.7 * math.log(0.7) + 0.2 * math.log(0.2) + 0.1 * math.log(0.1))
        for m in (1, 3, 5):
            self.check(f"M={m}_entropy_nat", [expected], predictive_entropy(p.repeat(m, 1, 1)))

    def test_member_permutation_invariance(self):
        generator = torch.Generator().manual_seed(42)
        p = torch.softmax(torch.randn(5, 7, 10, generator=generator, dtype=torch.float64), dim=-1)
        self.check("permuted_entropy_nat", predictive_entropy(p).numpy(),
                   predictive_entropy(p[[4, 1, 3, 0, 2]]))

    def test_output_shape_dtype_and_input_unchanged(self):
        p = torch.full((5, 9, 10), 0.1, dtype=torch.float32, requires_grad=True)
        before = p.detach().clone()
        entropy = predictive_entropy(p)
        self.check("output_shape", [9], list(entropy.shape), atol=0)
        self.check("input_unchanged", True, torch.equal(p.detach(), before), atol=0)
        self.check("float64_output", True, entropy.dtype == torch.float64, atol=0)
        self.check("device_preserved", True, entropy.device == p.device, atol=0)
        self.check("no_autograd_graph", False, entropy.requires_grad, atol=0)

    def test_zero_and_tiny_probabilities_finite(self):
        p = torch.zeros(1, 3, 10, dtype=torch.float64)
        p[0, 0, 0] = 1
        p[0, 1, 0], p[0, 1, 1] = 1, 1e-300
        p[0, 2, :2] = 0.5
        entropy = predictive_entropy(p)
        tiny_expected = -1e-300 * math.log(1e-300)
        self.check("finite_outputs", [True] * 3, torch.isfinite(entropy), atol=0)
        self.check("exact_zero", 0, entropy[0], atol=0)
        self.check("tiny_probability_contribution", tiny_expected, entropy[1], atol=1e-310)
        self.check("two_class_entropy_nat", math.log(2), entropy[2])

    def test_invalid_probabilities_rejected(self):
        good = torch.full((1, 2, 10), 0.1, dtype=torch.float64)
        cases = {
            "negative": good.clone(), "above_one": good.clone(), "nan": good.clone(),
            "positive_inf": good.clone(), "negative_inf": good.clone(),
            "mass_too_low": good * 0.9, "mass_too_high": good * 1.1,
            "zero_mass": torch.zeros_like(good), "rank_two": good[0],
            "rank_four": good.unsqueeze(0), "empty_members": good[:0],
            "empty_examples": good[:, :0], "one_class": torch.ones(1, 2, 1),
            "integer": good.to(torch.int64), "complex": good.to(torch.complex128),
            "boolean": good.to(torch.bool), "numpy_not_tensor": good.numpy(),
            "logits": torch.tensor([[[2.0, -1.0] + [0.0] * 8]]),
        }
        for name, value in (("negative", -0.01), ("above_one", 1.01), ("nan", math.nan),
                            ("positive_inf", math.inf), ("negative_inf", -math.inf)):
            cases[name][0, 0, 0] = value
        rejected = 0
        for name, p in cases.items():
            with self.subTest(case=name):
                with self.assertRaises((TypeError, ValueError)):
                    predictive_entropy(p)
                rejected += 1
        self.check("invalid_cases_rejected", len(cases), rejected, atol=0)

    def test_auroc_perfect_separation(self):
        # Unequal sizes intentionally mirror the unbalanced ID/OOD experiment.
        self.check("auroc", 1, ood_auroc([0.0, 0.1], [1.0, 1.2, 2.0]), atol=0)
        self.check("reverse_direction_auroc", 0, ood_auroc([1.0, 1.2], [0.0, 0.1, 0.2]), atol=0)

    def test_auroc_identical_scores_half(self):
        self.check("auroc", 0.5, ood_auroc(torch.full((3,), 0.5), np.full(5, 0.5)), atol=0)

    def test_entropy_means_and_gap(self):
        id_h, ood_h = [0.1, 0.3], [0.7, 0.9, 1.1]
        self.check("mean_id_entropy_nat", 0.2, mean_id_entropy(id_h))
        self.check("mean_ood_entropy_nat", 0.9, mean_ood_entropy(ood_h))
        self.check("entropy_gap_nat", 0.7, entropy_gap(id_h, ood_h))
        self.check("negative_gap_nat", -0.7, entropy_gap(ood_h, id_h))

    def test_seed_aggregation_sample_std(self):
        rows = {
            42: {"mean_id_entropy": 0.1, "mean_ood_entropy": 1.1, "entropy_gap": 1, "ood_auroc": 0.5},
            123: {"mean_id_entropy": 0.2, "mean_ood_entropy": 1.2, "entropy_gap": 1, "ood_auroc": 0.75},
            2026: {"mean_id_entropy": 0.3, "mean_ood_entropy": 1.3, "entropy_gap": 1, "ood_auroc": 1},
        }
        result = summarize_seed_metrics(rows)
        self.check("n_seeds", 3, result["n_seeds"], atol=0)
        self.check("seeds", [42, 123, 2026], result["seeds"], atol=0)
        self.check("ddof", 1, result["ddof"], atol=0)
        for name, mean, std in (("mean_id_entropy", 0.2, 0.1), ("mean_ood_entropy", 1.2, 0.1),
                                ("entropy_gap", 1, 0), ("ood_auroc", 0.75, 0.25)):
            self.check(f"{name}_mean", mean, result["metrics"][name]["mean"])
            self.check(f"{name}_sample_std", std, result["metrics"][name]["std"])

    def test_probability_averaging_not_logit_averaging(self):
        logits = torch.zeros(2, 1, 10, dtype=torch.float64)
        logits[0, 0, 0], logits[1, 0, 1] = 5, 1
        probabilities = torch.softmax(logits, dim=-1)
        p_bar = probabilities.numpy().mean(axis=0)
        expected = -(p_bar * np.log(p_bar)).sum(axis=-1)
        actual = predictive_entropy(probabilities)
        wrong = torch.softmax(logits.mean(dim=0), dim=-1)
        wrong_entropy = -(wrong * wrong.log()).sum(dim=-1)
        self.check("entropy_of_mean_probabilities_nat", expected, actual)
        self.evidence["incorrect_logit_average_entropy_nat"] = float(wrong_entropy[0])
        self.evidence["differs_from_logit_average"] = not math.isclose(
            float(actual[0]), float(wrong_entropy[0]), rel_tol=1e-6)
        self.assertTrue(self.evidence["differs_from_logit_average"])

    def test_ten_class_bounds_for_all_planned_M(self):
        generator = torch.Generator().manual_seed(2026)
        for m in (1, 3, 5):
            p = torch.softmax(50 * torch.randn(m, 37, 10, generator=generator, dtype=torch.float64), dim=-1)
            h = predictive_entropy(p)
            self.check(f"M={m}_all_within_[0,ln10]", True,
                       bool(((h >= 0) & (h <= math.log(10))).all()), atol=0)
            self.evidence[f"M={m}_observed_range_nat"] = [float(h.min()), float(h.max())]
        self.check("theoretical_upper_bound_nat", 2.302585092994046, math.log(10))

    def test_float32_float16_bfloat16_rounding(self):
        for dtype in (torch.float64, torch.float32, torch.float16, torch.bfloat16):
            p = torch.full((5, 2, 10), 0.1, dtype=dtype)
            self.check(f"{dtype}_uniform_entropy_nat", [math.log(10)] * 2, predictive_entropy(p))

    def test_invalid_entropy_vectors_rejected(self):
        invalid = [[], [[0.1]], [math.nan], [math.inf], [-0.1], [True], [1j], ["0.1"]]
        rejected = 0
        for vector in invalid:
            for function in (mean_id_entropy, mean_ood_entropy,
                             lambda x: entropy_gap(x, [1]), lambda x: entropy_gap([1], x),
                             lambda x: ood_auroc(x, [1]), lambda x: ood_auroc([1], x)):
                with self.subTest(vector=repr(vector), function=repr(function)):
                    with self.assertRaises((TypeError, ValueError)):
                        function(vector)
                    rejected += 1
        self.check("invalid_score_calls_rejected", 48, rejected, atol=0)

    def test_invalid_seed_aggregation_rejected(self):
        cases = [
            {}, {42: {"x": 1}}, {42: {}, 123: {}},
            {42: {"x": 1}, 123: {"y": 2}}, {42: {"x": math.nan}, 123: {"x": 2}},
            {42: {"x": 1}, 123: {"x": math.inf}}, {42: {"x": [1]}, 123: {"x": 2}},
            {"42": {"x": 1}, "123": {"x": 2}}, {42: {"x": True}, 123: {"x": 2}},
            {42: {"x": -1e308}, 123: {"x": 1e308}},
        ]
        rejected = 0
        for rows in cases:
            with self.subTest(rows=repr(rows)):
                with self.assertRaises((TypeError, ValueError)):
                    summarize_seed_metrics(rows)
                rejected += 1
        self.check("invalid_seed_inputs_rejected", len(cases), rejected, atol=0)


class ReportingResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records, self.statuses = [], {}

    def addSuccess(self, test):
        super().addSuccess(test)
        self.statuses[test.id()] = "passed"

    def addFailure(self, test, err):
        super().addFailure(test, err)
        self.statuses[test.id()] = "failed"

    def addError(self, test, err):
        super().addError(test, err)
        self.statuses[test.id()] = "error"

    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        self.statuses[test.id()] = "skipped"

    def addSubTest(self, test, subtest, err):
        super().addSubTest(test, subtest, err)
        if err is not None:
            self.statuses[test.id()] = "failed" if issubclass(err[0], AssertionError) else "error"

    def stopTest(self, test):
        self.records.append({"test": test.id(), "status": self.statuses.get(test.id(), "error"),
                             "checks": getattr(test, "evidence", {})})
        super().stopTest(test)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "outputs" / "metrics_test_results.json")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(HERE):
        parser.error("Test report must remain inside mnist_notmnist_ood")
    torch.set_num_threads(2)
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(MetricTests)
    result = unittest.TextTestRunner(verbosity=2, resultclass=ReportingResult).run(suite)
    report = {
        "successful": result.wasSuccessful(), "tests_run": result.testsRun,
        **{status: sum(row["status"] == status for row in result.records)
           for status in ("passed", "failed", "error", "skipped")},
        "scope": "Synthetic mathematical tests only; no dataset access, inference or checkpoint loading",
        "formulas": {"p_bar": "mean(probabilities, axis=0)",
                     "predictive_entropy_nat": "-sum_k p_bar_k * ln(p_bar_k), with 0*ln(0)=0",
                     "entropy_gap_nat": "mean(H_OOD) - mean(H_ID)",
                     "auroc": "roc_auc_score([0]*N_ID + [1]*N_OOD, [H_ID, H_OOD])",
                     "sample_std": "sqrt(sum_s (metric_s - mean)^2 / (S - 1)); ddof=1"},
        "tests": result.records,
        "failures": [{"test": str(test), "traceback": message} for test, message in result.failures],
        "errors": [{"test": str(test), "traceback": message} for test, message in result.errors],
        "environment": {"python": platform.python_version(), "torch": torch.__version__,
                        "numpy": np.__version__, "sklearn": importlib.metadata.version("scikit-learn"), "device": "cpu"},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Saved {output}", flush=True)
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
