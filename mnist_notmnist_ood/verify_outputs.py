"""Independently verify real resume caches, saved prefixes, tables and artifacts.

Reads saved probabilities; performs ZERO model forwards or dataset downloads.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

if __package__:
    from .data import HERE, digest, write_json
    from .inference import read_group, cache_path
    from .metrics import predictive_entropy, mean_id_entropy, mean_ood_entropy, entropy_gap, ood_auroc
    from .run_experiment import json_hash, group_metadata, make_summary, METRICS
else:
    from data import HERE, digest, write_json
    from inference import read_group, cache_path
    from metrics import predictive_entropy, mean_id_entropy, mean_ood_entropy, entropy_gap, ood_auroc
    from run_experiment import json_hash, group_metadata, make_summary, METRICS


def verify(output):
    output = Path(output)
    report = {"successful": False, "new_model_forward_passes": 0, "checks": []}
    def check(name, condition, **details):
        report["checks"].append({"name": name, "passed": bool(condition), **details})
        if not condition:
            raise AssertionError(name)
    try:
        manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
        config = manifest["config"]
        check("experiment_complete", manifest["status"] == "complete")
        with (output / "results_per_seed.csv").open(newline="", encoding="utf-8") as stream:
            saved_rows = list(csv.DictReader(stream))
        with (output / "results_summary.csv").open(newline="", encoding="utf-8") as stream:
            saved_summary = list(csv.DictReader(stream))
        check("36_seed_rows_and_12_summary_rows", len(saved_rows) == 36 and len(saved_summary) == 12)
        expected_keys = {(seed, method, m) for seed in config["seeds"]
                         for method in config["methods"] for m in config["ensemble_sizes"]}
        check("complete_distinct_seed_method_M_grid", expected_keys == {
            (int(row["run_seed"]), row["method"], int(row["M"])) for row in saved_rows})
        order = {}
        for name in ("mnist", "notmnist"):
            record = json.loads((output / "sample_order" / f"{name}.json").read_text(encoding="utf-8"))
            expected_hash = record.pop("order_sha256")
            check(f"{name}_sample_order_integrity", json_hash(record) == expected_hash)
            record["order_sha256"] = expected_hash
            check(f"{name}_all_examples", record["n_examples"] == (10000 if name == "mnist" else 18724))
            order[name] = record
        recomputed, cache_checks, entropy_checks = [], 0, 0
        for seed in config["seeds"]:
            for method in config["methods"]:
                entropies = {}
                for name in order:
                    metadata = group_metadata(seed, method, name, order[name], manifest["checkpoints"], config)
                    restored = read_group(cache_path(output, seed, method, name), manifest["inference_signature"], metadata, config)
                    check(f"resume_cache/{seed}/{method}/{name}", restored is not None)
                    stack, receipt = restored
                    cache_checks += 1
                    check(f"frozen_BN/{seed}/{method}/{name}",
                          all(row["bn_buffers_unchanged"] for row in receipt["validation"]["models"]))
                    if method == "mc_dropout":
                        check(f"distinct_MC_passes/{seed}/{name}", receipt["validation"]["mc_passes_differ"])
                    for m in config["ensemble_sizes"]:
                        h = predictive_entropy(stack[:m]).numpy()
                        stored = np.load(output / "entropy" / f"seed_{seed}_{method}_{name}_M{m}.npy", allow_pickle=False)
                        check(f"entropy_prefix/{seed}/{method}/{name}/M{m}", np.array_equal(h, stored))
                        entropy_checks += 1
                        entropies[name, m] = h
                for m in config["ensemble_sizes"]:
                    h_id, h_ood = entropies["mnist", m], entropies["notmnist", m]
                    row = {"run_seed": seed, "method": method, "M": m,
                           "mean_id_entropy": mean_id_entropy(h_id), "mean_ood_entropy": mean_ood_entropy(h_ood),
                           "entropy_gap": entropy_gap(h_id, h_ood), "ood_auroc": ood_auroc(h_id, h_ood)}
                    old_row = next(r for r in saved_rows if int(r["run_seed"]) == seed and r["method"] == method and int(r["M"]) == m)
                    check(f"metrics_recomputed/{seed}/{method}/M{m}", all(
                        abs(row[key] - float(old_row[key])) <= 1e-12 for key in METRICS))
                    recomputed.append(row)
        for row in make_summary(recomputed, config):
            saved = next(r for r in saved_summary if r["method"] == row["method"] and int(r["M"]) == row["M"])
            check(f"seed_summary_recomputed/{row['method']}/M{row['M']}", all(
                abs(row[f"{metric}_{stat}"] - float(saved[f"{metric}_{stat}"])) <= 1e-12
                for metric in METRICS for stat in ("mean", "std")))
        for name, sha256 in manifest["artifact_sha256"].items():
            check(f"artifact_hash/{name}", digest(output / name) == sha256)
        check("24_complete_probability_files", len(list((output / "probabilities").glob("*.npz"))) == 24)
        check("72_per_image_entropy_files", len(list((output / "entropy").glob("*.npy"))) == 72)
        report.update(successful=True, resume_cache_checks=cache_checks, entropy_prefix_checks=entropy_checks,
                      seed_metric_rows_recomputed=36, summary_rows_recomputed=12,
                      passed=len(report["checks"]), failed=0)
    except BaseException as exc:
        report.update(error=repr(exc), failed=1, passed=sum(row["passed"] for row in report["checks"]))
        raise
    finally:
        write_json(output / "output_verification.json", report)
    print(f"OUTPUT VERIFICATION PASSED: {report['passed']} checks; 24 real resume caches, 72 exact entropy prefixes; zero model forwards")
    return report


if __name__ == "__main__":
    verify(HERE / "outputs")
