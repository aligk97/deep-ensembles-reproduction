from math import log, pi, sqrt
from statistics import NormalDist

import numpy as np
import pytest
import torch

from src.toy_regression.compare_regional_performance import aggregate_rows, collect_region, compare_v2
from src.toy_regression.compute_regional_metrics import LEVELS, evaluate_region, mixture_bounds, region_masks, sample_summary
from src.toy_regression.load_saved_predictions import Samples, discover, load_members, read_prediction
from src.toy_regression.v1_v2_regression_networks import GaussianRegressor
from src.toy_regression.train_v1_v2_models import CONDITIONS


def column(values):
    return torch.tensor(values, dtype=torch.float64).reshape(-1, 1)


def test_regions_include_only_the_requested_endpoints():
    x = column([-6.01, -6, -5, -4.01, -4, 0, 4, 4.01, 5, 6, 6.01])
    masks = region_masks(x)
    assert [x[mask].tolist() for mask in masks.values()] == [
        [-6, -5, -4.01], [-4, 0, 4], [4.01, 5, 6], [-6, -5, -4.01, 4.01, 5, 6],
    ]
    assert [int(mask.sum()) for mask in masks.values()] == [3, 3, 3, 6]


def test_rmse_to_noisy_and_true_targets():
    x, target, mu = column([0, 2]), column([3, 7]), column([0, 6])
    result = evaluate_region(x, target, {"mean": mu}, CONDITIONS[1], 1e-6, "in_domain")
    assert result["rmse_noisy"] == pytest.approx(sqrt(5))
    assert result["rmse_true"] == pytest.approx(sqrt(2))


def test_full_single_gaussian_nll_and_variance_bias():
    x, mu, target, var = column([0, 0]), column([1, 1]), column([1, 3]), column([4, 4])
    result = evaluate_region(x, target, {"mean": mu, "total_variance": var, "aleatoric": var},
                             CONDITIONS[1], 1e-6, "in_domain")
    assert result["predictive_nll"] == pytest.approx(.5 * (log(2 * pi * 4) + .5))
    assert result["aleatoric_signed_error_from_9"] == -5
    assert result["aleatoric_abs_error_from_9"] == 5


def test_gaussian_mixture_nll_stable_and_distinct_from_moment_nll():
    component_means = torch.tensor([[[-1.0], [0.0]], [[1.0], [2.0]]], dtype=torch.float64)
    component_vars = torch.ones_like(component_means)
    x, target = column([0, 1]), column([0, 1000])
    prediction = {"mean": component_means.mean(0), "total_variance": column([2, 2]),
                  "member_means": component_means, "member_variances": component_vars}
    values = evaluate_region(x, target, prediction, CONDITIONS[-1], 1e-6, "in_domain")
    expected = ((log(2 * pi) + 1) / 2 + (log(2 * pi) + 998**2) / 2 + log(2)) / 2
    assert values["predictive_nll"] == pytest.approx(expected)
    assert values["moment_matched_nll"] != pytest.approx(expected)


def test_gaussian_coverage_and_width_distinguish_95_from_3sigma():
    x = column([0, 0, 0, 0])
    mu, var = x, column([4, 4, 4, 4])
    target = column([0, 3.91, 4, 6])
    result = evaluate_region(x, target, {"mean": mu, "total_variance": var}, CONDITIONS[1], 1e-6, "in_domain")
    assert [result[f"coverage_{level}"] for level, _ in LEVELS] == [.25, .25, .25, .50, 1.0]
    assert result["true_function_containment_95"] == 1
    assert result["interval_width_95"] == pytest.approx(4 * NormalDist().inv_cdf(.975))
    assert result["interval_width_99_73"] == 12


def test_mixture_intervals_invert_cdf_at_all_five_nominal_levels():
    means = torch.tensor([[[-3.0]], [[2.0]]], dtype=torch.float64)
    variances = torch.tensor([[[.25]], [[4.0]]], dtype=torch.float64)
    for (_, nominal), (low, high) in zip(LEVELS, mixture_bounds(means, variances), strict=True):
        for endpoint, tail in ((low, (1 - nominal) / 2), (high, (1 + nominal) / 2)):
            attained = torch.special.ndtr((endpoint - means) / variances.sqrt()).mean().item()
            assert attained == pytest.approx(tail, abs=1e-10)


def test_saved_members_rebuild_aleatoric_epistemic_and_total(tmp_path):
    x = np.array([-5, 0, 5])
    arrays = {"x": x, "member_means": np.repeat(np.arange(5)[:, None], 3, axis=1),
              "member_variances": np.full((5, 3), 9.0)}
    targets = Samples(column([5, -5]), column([123, -123]), tmp_path)
    pred = read_prediction(arrays, targets, CONDITIONS[-1], 42)
    assert pred["mean"].flatten().tolist() == [2, 2]
    assert pred["aleatoric"].flatten().tolist() == [9, 9]
    assert pred["epistemic"].flatten().tolist() == [2, 2]
    assert torch.equal(pred["total_variance"], pred["aleatoric"] + pred["epistemic"])


def test_mean_and_sample_standard_deviation():
    assert sample_summary([1, 2, 3]) == (2, 1)
    assert sample_summary([3]) == (3, None)


def test_combined_ood_pools_individual_examples_with_unequal_counts(tmp_path):
    x = column([-5, 0, 5, 5.5, 6, 6.01])
    target = column([1, 50, 4, 4, 4, 100])
    zero = torch.zeros_like(x)
    part = (Samples(x, target, tmp_path), {"mean": zero, "total_variance": torch.ones_like(x)})
    combined = collect_region([part], "combined_ood")
    assert combined[0].flatten().tolist() == [-5, 5, 5.5, 6]
    result = evaluate_region(*combined, CONDITIONS[1], 1e-6, "combined_ood")
    assert result["rmse_noisy"] == pytest.approx(sqrt(49 / 4))
    assert result["predictive_nll"] == pytest.approx(.5 * (log(2 * pi) + 49 / 4))
    assert result["coverage_95"] == .25
    only_left = (Samples(x[:1], target[:1], tmp_path), {"mean": zero[:1]})
    assert collect_region([only_left], "combined_ood") is None


def test_v2_pairing_uses_same_seed_and_saved_test_fingerprint():
    rows = []
    for label, values, fingerprints in (
        ("V2-A", [1, 2, 3], ["x", "y", "z"]), ("V2-C", [4, 20, 6], ["x", "other", "z"]),
    ):
        for seed, value, fingerprint in zip((42, 123, 2026), values, fingerprints, strict=True):
            rows.append(dict(configuration=label, version="V2", seed=seed, model=CONDITIONS[-1].name,
                             model_label=CONDITIONS[-1].title, region="right_ood", x_bounds="(4,6]",
                             metric="rmse_true", metric_definition="test", distribution="point_or_raw_moments",
                             value=value, test_sha256=fingerprint, n_samples=1000, source_run=label))
    comparison = next(row for row in compare_v2(rows) if row["model"] == CONDITIONS[-1].name
                      and row["region"] == "right_ood" and row["metric"] == "rmse_true")
    assert comparison["paired_seeds"] == "42|2026"
    assert comparison["paired_delta_c_minus_a"] == 3
    assert comparison["paired_delta_sample_std_ddof1"] == 0
    assert aggregate_rows(rows)[0]["sample_std_ddof1"] == 1


def test_discovery_uses_saved_configuration_and_loads_checkpoint_to_cpu(tmp_path):
    run_dir = tmp_path / "arbitrary-directory"
    model_dir = run_dir / "seed_42" / CONDITIONS[1].name / "models"
    model_dir.mkdir(parents=True)
    (run_dir / "config_resolved.yaml").write_text("learning_rate: 0.1\nbatch_size: 5\n", encoding="utf-8")
    original = GaussianRegressor(hidden_dim=7)
    torch.save({"outer_seed": 42, "hidden_dim": 7, "model_state_dict": original.state_dict()},
               model_dir / "member_0.pt")
    detected, _, _ = discover([tmp_path])
    assert list(detected) == ["V2-A"]
    paths = []
    model, = load_members(detected["V2-A"], 42, CONDITIONS[1], paths.append)
    assert paths == [model_dir / "member_0.pt"]
    assert not model.training
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    assert torch.equal(model.network[0].weight, original.network[0].weight)
