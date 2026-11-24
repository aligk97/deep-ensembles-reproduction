import csv
import json

import numpy as np
import pytest
import torch

from src.toy_regression import experiment_commands, train_v1_v2_models as train
from src.toy_regression.experiment_config import load_config
from src.toy_regression.summarize_seed_results import load_completed, summarize


def test_cpu_smoke_archive_and_shared_data(tmp_path, monkeypatch):
    config = load_config(experiment_commands.DEFAULT_CONFIG, smoke=True, device="cpu")
    original = experiment_commands.train_condition
    observed = []

    def check_shared(condition, config, split, device, epsilon, seed, path, digest, log):
        observed.append(split)
        with np.load(path.parent / "data" / "train.npz") as data:
            np.testing.assert_array_equal(data["x"], split.x.numpy())
            np.testing.assert_array_equal(data["y_noisy"], split.y_noisy.numpy())
        return original(condition, config, split, device, epsilon, seed, path, digest, log)

    monkeypatch.setattr(experiment_commands, "train_condition", check_shared)
    run_dir = experiment_commands.run_experiment(config, tmp_path)
    manifest = load_completed(run_dir)
    assert manifest["smoke"] and manifest["completed_seeds"] == [42]
    assert all(split is observed[0] for split in observed) and len(observed) == 4
    assert load_config(run_dir / "config_resolved.yaml") == config
    for name in ("environment.json", "provenance.md", "logs/run.log"):
        assert (run_dir / name).stat().st_size > 0

    seed_dir = run_dir / "seed_42"
    for condition in train.CONDITIONS:
        path = seed_dir / condition.name
        count = 5 if condition.ensemble else 1
        assert len(list((path / "models").glob("*.pt"))) == count
        for index in range(count):
            history = path / "histories" / f"member_{index}.csv"
            with history.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            assert len(rows) == config.max_epochs
            assert all(row["finite"] == "True" for row in rows)
        checkpoint = torch.load(path / "models/member_0.pt", map_location="cpu", weights_only=True)
        assert checkpoint["final_epoch"] == config.max_epochs
        assert checkpoint["config_hash"] == manifest["config_hash"]
        assert checkpoint["model_state_dict"] and checkpoint["optimizer_state_dict"]
        for split in ("grid", "test_in", "test_ood"):
            with np.load(path / f"predictions_{split}.npz") as prediction, np.load(
                seed_dir / "data" / f"{split}.npz"
            ) as data:
                np.testing.assert_array_equal(prediction["x"], data["x"])
                np.testing.assert_allclose(prediction["std"] ** 2, prediction["total_variance"],
                                           rtol=1e-6, atol=1e-8)
                assert prediction["member_means"].shape == (count, len(data["x"]), 1)
                assert bool(prediction["aleatoric_available"]) == condition.gaussian
                assert bool(prediction["epistemic_available"]) == condition.ensemble
                assert all(np.isfinite(prediction[key]).all() for key in prediction.files
                           if prediction[key].dtype.kind in "fiu")
        result = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        expected_kind = ("mixture" if condition.ensemble else "gaussian") if condition.gaussian else "proxy_gaussian"
        assert result["metric_types"]["predictive_nll_in"] == expected_kind

    figure_names = ("figure1_four_panels", "loss_curves", "uncertainty_decomposition",
                    "coverage_comparison", "predictions_by_member")
    aggregate = run_dir / "aggregate"
    for directory, names in ((seed_dir / "figures", figure_names),
                             (aggregate / "figures", ("figure1_seed_comparison", "metrics_comparison"))):
        for name in names:
            for suffix in ("pdf", "png"):
                assert (directory / f"{name}.{suffix}").stat().st_size > 1000
    summary = json.loads((aggregate / "metrics_mean_std.json").read_text(encoding="utf-8"))
    assert summary["n_outer_seeds"] == 1
    assert summary["conditions"]["nll_single"]["rmse_noisy_in"]["sample_std"] is None


def test_sample_std_over_outer_seeds():
    rows = [{"condition": "model", "outer_seed": seed, "metric_types": {}, "smoke": False,
             "metrics": {"rmse": value, "epistemic": None}}
            for seed, value in ((42, 1.0), (123, 3.0), (2026, 5.0))]
    stats, _ = summarize(rows)
    assert stats["model"]["rmse"] == {"metric_type": "shared", "mean": 3, "sample_std": 2, "n": 3}
    assert stats["model"]["epistemic"]["mean"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        summarize(rows + rows[:1])


def test_failed_run_keeps_history_and_is_not_aggregated(tmp_path, monkeypatch):
    config = load_config(experiment_commands.DEFAULT_CONFIG, smoke=True, device="cpu")
    original = train.training_losses

    def nonfinite(condition, model, x, y, epsilon):
        clean, adv = original(condition, model, x, y, epsilon)
        return clean * float("nan"), adv

    monkeypatch.setattr(train, "training_losses", nonfinite)
    with pytest.raises(RuntimeError, match="Non-finite training"):
        experiment_commands.run_experiment(config, tmp_path)
    run_dir, = tmp_path.iterdir()
    failure = json.loads((run_dir / "failure.json").read_text(encoding="utf-8"))
    assert "epoch 1, batch 1" in failure["error"]
    history = run_dir / "seed_42/mse_ensemble_5/histories/member_0.csv"
    with history.open(encoding="utf-8", newline="") as stream:
        assert next(csv.DictReader(stream))["finite"] == "False"
    with pytest.raises(ValueError, match="Run status is failed"):
        load_completed(run_dir)
    assert not (run_dir / "aggregate").exists()
