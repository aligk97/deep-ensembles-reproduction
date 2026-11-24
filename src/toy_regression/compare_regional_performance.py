"""Evaluate saved toy regression models without invoking the training entry point."""

import argparse
from collections import defaultdict
import logging
from os.path import relpath
from pathlib import Path
import pickle
import sys

import torch

from .compute_regional_metrics import METRICS, REGIONS, distribution_for, evaluate_region, region_masks, sample_summary
from .load_saved_predictions import (
    FIXED_CONFIGS, SEEDS, SPLITS, Samples, discover, file_hash, first_file, infer, load_members,
    native_samples, read_arrays, read_prediction, read_samples,
)
from .train_v1_v2_models import CONDITIONS
from .archive_project_versions import write_snapshot
from .experiment_io import DEFAULT_RUN_ROOT, environment, git_metadata, new_run_dir, utc_now, write_csv, write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results"
DEEP = "nll_ensemble_5"
DATA_ERRORS = (OSError, ValueError, TypeError, KeyError, IndexError, RuntimeError, pickle.UnpicklingError)


def collect_region(splits, region):
    selected = []
    seen_sides = set()
    for samples, prediction in splits:
        masks = region_masks(samples.x.flatten())
        mask = masks["in_domain"] | masks["combined_ood"] if region == "overall" else masks[region]
        if mask.any():
            seen_sides.update(side for side in ("left_ood", "right_ood") if (masks[side] & mask).any())
            selected.append((samples.x[mask], samples.y[mask], {
                name: values[:, mask] if name.startswith("member_") else values[mask]
                for name, values in prediction.items()
            }))
    if not selected or (region == "combined_ood" and len(seen_sides) != 2):
        return None
    x, y, prediction = zip(*selected)
    shared_fields = [name for name in prediction[0] if all(name in fields for fields in prediction)]
    return torch.cat(x), torch.cat(y), {
        name: torch.cat([fields[name] for fields in prediction], dim=1 if name.startswith("member_") else 0)
        for name in shared_fields
    }


def available_prediction(run, seed, condition, split, reference, native, read, checkpoints, issue):
    path = first_file(run.seed_path(seed) / condition.name, f"predictions_{split}")
    arrays = None
    if path is not None:
        try:
            arrays = read(path)
        except DATA_ERRORS as error:
            issue(f"{path}: {error}")
    targets = [(reference, "common_test" if split != "grid" else "saved_clean_grid")]
    if native is not None and (reference is None or split != "grid"):
        targets.append((native, "native_test" if split != "grid" else "saved_clean_grid"))
    partial = None
    for samples, mode in targets:
        if samples is None:
            continue
        if arrays is not None:
            try:
                prediction = read_prediction(arrays, samples, condition, seed)
                saved = (samples, prediction, mode, "saved_predictions", path)
                needs_members = condition.gaussian and condition.ensemble and split != "grid"
                if "total_variance" in prediction and (not needs_members or "member_variances" in prediction):
                    return saved
                partial = partial or saved
            except DATA_ERRORS as error:
                issue(f"{path} on {mode}: {error}")
        try:
            models = checkpoints()
            if models is not None:
                return samples, infer(models, samples, condition), mode, "cpu_checkpoint_inference", None
        except DATA_ERRORS as error:
            issue(f"Inference on {split}: {error}")
    return partial


def metric_rows(runs, read, record, validation, log):
    results, grids = [], {}
    saved_data, references = {}, {}
    for run in runs.values():
        for name in ("config_resolved.yaml", "manifest.json"):
            if (path := run.path / name).is_file():
                record(path)
        for seed in SEEDS:
            for split in SPLITS:
                issue = validation["source_issues"].append
                samples = native_samples(run.seed_path(seed), split, CONDITIONS, read, issue)
                saved_data[run.label, seed, split] = samples
                if samples is not None:
                    if samples.excluded:
                        issue(f"{samples.source}: omitted {samples.excluded} x outside [-6,6]")
                    references.setdefault((seed, split), samples)

    validation["reference_datasets"] = [
        {"seed": seed, "split": split, "source": str(samples.source), "n": len(samples.x),
         "sha256_xy": samples.fingerprint(torch.ones(len(samples.x), dtype=torch.bool))}
        for (seed, split), samples in references.items()
    ]
    for run in runs.values():
        for seed in SEEDS:
            for condition in CONDITIONS:
                context = {"configuration": run.label, "seed": seed, "model": condition.name}
                log.info("%s | seed %s | %s", run.label, seed, condition.name)
                issues = []
                splits = []
                models, checkpoint_attempted = None, False

                def checkpoints():
                    nonlocal models, checkpoint_attempted
                    if not checkpoint_attempted:
                        checkpoint_attempted = True
                        try:
                            models = load_members(run, seed, condition, record)
                        except DATA_ERRORS as error:
                            issues.append(f"Checkpoint fallback: {error}")
                    return models

                for split in (*SPLITS, "grid"):
                    if split == "grid":
                        if run.label not in ("V2-A", "V2-C"):
                            continue
                        native = native_samples(run.seed_path(seed), split, (condition,), read, issues.append)
                        reference = native
                    else:
                        native = saved_data[run.label, seed, split]
                        reference = references.get((seed, split))
                        model_path = first_file(run.seed_path(seed) / condition.name, f"predictions_{split}")
                        if model_path is not None:
                            try:
                                embedded = read_samples(model_path, read(model_path), require_noisy=True)
                                if native is None or not torch.equal(native.x, embedded.x):
                                    native = embedded
                            except DATA_ERRORS:
                                pass  # The independent split can provide the targets.
                    loaded = available_prediction(run, seed, condition, split, reference, native, read,
                                                  checkpoints, issues.append)
                    if loaded is None:
                        validation["missing_splits"].append({**context, "split": split})
                        continue
                    samples, prediction, mode, origin, path = loaded
                    validation["prediction_sources"].append({
                        **context, "split": split, "mode": mode, "prediction_origin": origin,
                        "data_source": str(samples.source), "prediction_source": str(path) if path else None,
                        "n": len(samples.x),
                    })
                    if split == "grid":
                        grids[run.label, seed, condition.name] = (samples, prediction)
                    else:
                        splits.append((samples, prediction))
                if issues:
                    validation["source_issues"].append({**context, "details": issues})
                for region, bounds in REGIONS.items():
                    collected = collect_region(splits, region)
                    if collected is None:
                        validation["missing_regions"].append({**context, "region": region})
                        continue
                    x, y, prediction = collected
                    values = evaluate_region(x, y, prediction, condition, run.floor, region)
                    if "predictive_nll" not in values or "coverage_95" not in values:
                        validation["missing_metrics"].append({
                            **context, "region": region,
                            "missing": [name for name in ("predictive_nll", "coverage_95") if name not in values],
                        })
                    if condition.name == DEEP:
                        fields = ("aleatoric", "epistemic", "total_variance")
                        if all(name in prediction for name in fields):
                            a, e, t = (prediction[name] for name in fields)
                            difference = (a + e - t).abs().max().item()
                            validation["variance_decomposition_checks"].append({
                                **context, "region": region, "max_absolute_residual": difference,
                                "passed": bool(torch.allclose(a + e, t, rtol=3e-5, atol=1e-5)),
                            })
                        elif "total_variance" in prediction:
                            validation["missing_metrics"].append({
                                **context, "region": region, "missing": [name for name in fields if name not in prediction],
                            })
                    fingerprint = Samples(x, y, run.path).fingerprint(torch.ones(len(x), dtype=torch.bool))
                    validation["sample_counts"].append({
                        **context, "region": region, "n_samples": len(x), "test_sha256": fingerprint,
                    })
                    for metric, value in values.items():
                        results.append({
                            **context, "version": run.version, "model_label": condition.title,
                            "source_run": run.path.name, "region": region, "x_bounds": bounds,
                            "n_samples": len(x), "test_sha256": fingerprint, "metric": metric,
                            "metric_definition": METRICS[metric], "distribution": distribution_for(metric, condition),
                            "value": value,
                        })
    return results, grids


SUMMARY_FIELDS = ("configuration", "version", "model", "model_label", "region", "x_bounds",
                  "metric", "metric_definition", "distribution")


def aggregate_rows(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[field] for field in SUMMARY_FIELDS)].append(row)
    summary = []
    for key, items in sorted(groups.items()):
        items = sorted(items, key=lambda row: row["seed"])
        average, deviation = sample_summary([row["value"] for row in items])
        summary.append({
            **dict(zip(SUMMARY_FIELDS, key)), "source_run": items[0]["source_run"],
            "n_seeds": len(items), "seeds": "|".join(str(row["seed"]) for row in items),
            "n_samples_per_seed": "|".join(str(row["n_samples"]) for row in items),
            "test_sha256_per_seed": "|".join(row["test_sha256"] for row in items),
            "mean": average, "sample_std_ddof1": deviation,
        })
    return summary


def compare_v2(rows):
    by_key = {(row["configuration"], row["model"], row["region"], row["metric"], row["seed"]): row
              for row in rows if row["configuration"] in ("V2-A", "V2-C")}
    output = []
    for condition in CONDITIONS:
        for region, bounds in REGIONS.items():
            for metric, definition in METRICS.items():
                if metric.startswith("aleatoric") and (not condition.gaussian or
                                                       "error_from_9" in metric and region != "in_domain"):
                    continue
                if (metric.startswith("mixture_") or metric == "moment_matched_nll") and condition.name != DEEP:
                    continue
                if metric == "epistemic_variance" and not condition.ensemble:
                    continue
                sides = {label: [by_key.get((label, condition.name, region, metric, seed)) for seed in SEEDS]
                         for label in ("V2-A", "V2-C")}
                left, right = sides.values()
                matched = [(a, c) for a, c in zip(left, right) if a is not None and c is not None
                           and a["test_sha256"] == c["test_sha256"]]
                deltas = [c["value"] - a["value"] for a, c in matched]
                delta, delta_std = sample_summary(deltas) if deltas else (None, None)
                line = {
                    "configuration_a": "V2-A", "configuration_c": "V2-C", "model": condition.name,
                    "model_label": condition.title, "region": region, "x_bounds": bounds,
                    "metric": metric, "metric_definition": definition, "distribution": distribution_for(metric, condition),
                    "paired_seeds": "|".join(str(a["seed"]) for a, _ in matched), "n_paired_seeds": len(matched),
                    "paired_delta_c_minus_a": delta, "paired_delta_sample_std_ddof1": delta_std,
                    "paired_a_mean": sample_summary([a["value"] for a, _ in matched])[0] if matched else None,
                    "paired_c_mean": sample_summary([c["value"] for _, c in matched])[0] if matched else None,
                }
                for name, side in zip(("a", "c"), (left, right)):
                    available = [row for row in side if row is not None]
                    values = [row["value"] for row in available]
                    avg, std = sample_summary(values) if values else (None, None)
                    line.update({f"{name}_seeds": "|".join(str(row["seed"]) for row in available),
                                 f"{name}_mean": avg, f"{name}_sample_std_ddof1": std})
                line["status"] = ("paired_3_seeds" if len(matched) == len(SEEDS) else
                                  "partially_paired" if matched else "no_common_seed_data")
                output.append(line)
    return output


def manifest_path(path, metrics_dir):
    try:
        return Path(relpath(path, metrics_dir)).as_posix()
    except ValueError:  # Windows paths on separate drives have no relative path.
        return path.resolve().as_posix()


def artifact_index(metrics_dir, *other_dirs):
    return {manifest_path(path, metrics_dir): file_hash(path)
            for directory in (metrics_dir, *other_dirs)
            for path in sorted(directory.glob("*"))
            if path.is_file() and path.name != "manifest.json"}


def run_analysis(inputs, output, command, *, report_output=None):
    from .plot_regional_comparisons import make_figures
    from .write_regional_report import write_report, write_latex

    torch.set_num_threads(1)
    output = Path(output).resolve()
    run_dir = new_run_dir(output / "metrics", "regional")
    figures_dir = output / "figures" / run_dir.name
    report_dir = run_dir if report_output is None else Path(report_output).resolve() / run_dir.name
    artifact_dirs = []
    log = logging.Logger("regional_analysis")
    handler = logging.FileHandler(run_dir / "analysis.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    manifest = {
        "run_id": run_dir.name, "started_at_utc": utc_now().isoformat(), "status": "running",
        "purpose": "Kayıtlı Toy Regression tahminlerinin bölgesel doğruluğunu ve kalibrasyonunu ölçmek.",
        "command": command, "input_roots": [str(root) for root in inputs], "git": git_metadata(),
        "environment": environment(torch.device("cpu")), "training": False, "seeds": SEEDS,
        "regions": REGIONS, "paper": "https://arxiv.org/pdf/1612.01474v3#page=6",
        "output_layout": {
            "metrics": ".", "figures": manifest_path(figures_dir, run_dir),
            "report": manifest_path(report_dir, run_dir),
        },
    }
    source_hashes = {}

    def record(path):
        if str(path) not in source_hashes:
            source_hashes[str(path)] = file_hash(path)

    def read(path):
        record(path)
        return read_arrays(path)

    try:
        write_json(run_dir / "manifest.json", manifest)
        manifest["source_snapshot"] = write_snapshot(run_dir, run_dir.name)
        runs, audit, archives = discover(inputs)
        v3_count = sum(run.version == "V3" for run in runs.values())
        validation = {
            "expected_configuration_count": len(FIXED_CONFIGS) + 14,
            "detected_configuration_count": len(runs),
            "missing_named_configurations": [label for label in FIXED_CONFIGS if label not in runs],
            "expected_v3_configurations": 14, "detected_v3_configurations": v3_count,
            "missing_v3_configuration_count": max(0, 14 - v3_count),
            "discovery": audit, "archive_inventory": archives, "source_issues": [], "missing_splits": [],
            "missing_regions": [], "missing_metrics": [], "sample_counts": [], "prediction_sources": [],
            "variance_decomposition_checks": [],
        }
        manifest["selected_runs"] = [
            {"configuration": run.label, "version": run.version, "path": str(run.path), "parameters": run.config,
             "original_manifest": run.manifest}
            for run in runs.values()
        ]
        log.info("Selected configurations: %s", ", ".join(runs))
        log.info("Missing configurations: %s | V3 present: %d of 14", validation["missing_named_configurations"],
                 v3_count)
        rows, grids = metric_rows(runs, read, record, validation, log)
        summary = aggregate_rows(rows)
        v2 = compare_v2(rows)
        comparison = [{**row, "status": "available"} for row in summary]
        for label in validation["missing_named_configurations"]:
            comparison.append({"configuration": label, "version": label.split("-")[0], "n_seeds": 0,
                               "status": "missing_run"})
        if v3_count < 14:
            comparison.append({"configuration": "V3", "version": "V3", "n_seeds": 0,
                               "status": "missing_run" if not v3_count else "partial_configuration_inventory",
                               "missing_configurations": 14 - v3_count})
        write_csv(run_dir / "regional_metrics_per_seed.csv", rows or [
            {key: None for key in (*SUMMARY_FIELDS, "seed", "n_samples", "test_sha256", "value")}])
        write_csv(run_dir / "regional_metrics_mean_std.csv", summary or [
            {key: None for key in (*SUMMARY_FIELDS, "n_seeds", "mean", "sample_std_ddof1")}])
        write_csv(run_dir / "v2a_vs_v2c.csv", v2)
        write_csv(run_dir / "v1_v2_v3_comparison.csv", comparison)
        if report_dir != run_dir:
            report_dir.mkdir(parents=True)
            artifact_dirs.append(report_dir)
        write_latex(report_dir / "regional_results.tex", summary)
        figures_dir.mkdir(parents=True)
        artifact_dirs.append(figures_dir)
        make_figures(figures_dir, summary, grids)
        validation["unchanged_read_inputs"] = all(file_hash(Path(path)) == digest
                                                   for path, digest in source_hashes.items())
        validation["variance_decomposition_passed"] = all(
            row["passed"] for row in validation["variance_decomposition_checks"])
        validation["per_seed_metric_rows"] = len(rows)
        validation["mean_std_metric_rows"] = len(summary)
        complete = (not validation["missing_named_configurations"] and v3_count == 14 and
                    not any(validation[key] for key in ("missing_splits", "missing_regions", "missing_metrics")))
        if not validation["unchanged_read_inputs"] or not validation["variance_decomposition_passed"]:
            raise ValueError("Input integrity or variance decomposition validation failed")
        write_report(report_dir / "analysis_report_tr.md", summary, v2, validation, manifest)
        manifest["status"] = "completed" if complete else "completed_with_missing_data"
        log.info("Wrote %d per-seed and %d summary metrics | %s", len(rows), len(summary), manifest["status"])
        write_json(run_dir / "validation.json", validation)
    except Exception:
        manifest["status"] = "failed"
        log.exception("Analysis failed")
        raise
    finally:
        manifest["finished_at_utc"] = utc_now().isoformat()
        manifest["read_input_sha256"] = source_hashes
        handler.close()
        manifest["artifacts"] = artifact_index(run_dir, *artifact_dirs)
        write_json(run_dir / "manifest.json", manifest)
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", type=Path,
                        help="Directory with saved toy runs (default: results/raw/toy_regression). Repeatable.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Base directory for metrics and figures (default: results).")
    parser.add_argument("--report-dir", type=Path,
                        help="Optional base directory for report and LaTeX (default: the metrics run directory).")
    args = parser.parse_args(argv)
    inputs = [path.resolve() for path in (args.input_dir or [DEFAULT_RUN_ROOT])]
    for path in inputs:
        if not path.is_dir():
            parser.error(f"Input directory does not exist: {path}")
    command = ["python", "-m", "src.toy_regression.compare_regional_performance",
               *(sys.argv[1:] if argv is None else argv)]
    print(run_analysis(inputs, args.output_dir, command, report_output=args.report_dir))


if __name__ == "__main__":
    main()
