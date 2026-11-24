"""Run or resume the V3 matrix. Relative output paths always resolve from the Git repository."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import re
import shutil
import sys
from uuid import uuid4

import torch
import yaml

from .archive_project_versions import write_snapshot
from .experiment_config import load_config
from .experiment_io import DEFAULT_RUN_ROOT, configure_torch, environment, git_metadata, utc_now, write_json
from .generate_cubic_data import generate_data, fgsm_epsilon, save_data
from .train_v1_v2_models import CONDITIONS
from .train_v3_members import train_group

ROOT = Path(__file__).resolve().parents[2]
CONFIG = Path(__file__).with_name("configs") / "toy_v3.yaml"


def digest(path):
    with path.open("rb") as f:
        return sha256(f.read()).hexdigest()


def verify_v4_reference(spec, base):
    reference = DEFAULT_RUN_ROOT / spec["baseline_run_id"]
    folder = reference / "arms" / spec["baseline_arm"]
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if (manifest["experiment_version"], manifest["status"], manifest["verified_models"]) != ("V2-A", "completed", 36):
        raise ValueError("A verified 36-member V2-A baseline is required")
    resolved = load_config(folder / "config_resolved.yaml", smoke=base.smoke)
    if resolved.learning_rate != 0.1 or replace(resolved, learning_rate=base.learning_rate) != base:
        raise ValueError("V2-A and V4 must differ only in learning rate (including device)")
    expected_arm = {0.3: "v4_lr_0p3", 0.03: "v4_lr_0p03"}[base.learning_rate]
    if spec["arms"] != {expected_arm: {"batch_size": 5, "max_epochs": 3000}} or spec["seeds"] != [42, 123, 2026]:
        raise ValueError("Invalid V4 arm or seeds")
    with (reference / "run_files.csv").open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["path"].startswith("arms/v2_a/")]
    if sum(r["path"].endswith(".pt") and "/models/" in r["path"] for r in rows) != 36:
        raise ValueError("Incomplete V2-A checkpoint index")
    for row in rows:
        if digest(reference / row["path"]) != row["sha256"]:
            raise ValueError(f"V2-A reference hash mismatch: {row['path']}")
    return folder


def train_arm(raw, arm, settings, base, device_name):
    device = configure_torch(device_name, False)
    torch.set_num_threads(1)
    folder = raw / "arms" / arm
    folder.mkdir(parents=True, exist_ok=True)
    options = settings.copy()
    mse_lr = options.pop("mse_learning_rate", base.learning_rate)
    config = replace(base, **options)
    if base.smoke:
        config = replace(config, max_epochs=min(3, config.max_epochs))
    write_json(folder / "parameters.json", dict(**config.to_dict(), mse_learning_rate=mse_lr))
    splits = {seed: generate_data(config, seed) for seed in config.outer_seeds}
    for seed in config.outer_seeds:
        target = folder / f"seed_{seed}" / "data"
        target.mkdir(parents=True, exist_ok=True)
        for source in (raw / "data" / f"seed_{seed}").iterdir():
            dest = target / source.name
            if not dest.exists():
                shutil.copyfile(source, dest)
            if digest(source) != digest(dest):
                raise ValueError(f"Data mismatch: {dest}")
    errors = {}
    for condition in CONDITIONS:
        flag = folder / (condition.name + "_finished.json")
        if flag.exists():
            saved = json.loads(flag.read_text(encoding="utf-8")).get("files", {})
            if saved and all((folder / name).is_file() and digest(folder / name) == expected
                             for name, expected in saved.items()):
                continue
        try:
            resolved = config if condition.gaussian else replace(config, learning_rate=mse_lr)
            train_group(folder, arm, condition, resolved, splits, device)
            outputs = {f.relative_to(folder).as_posix(): digest(f)
                       for seed in config.outer_seeds
                       for f in (folder / f"seed_{seed}" / condition.name).rglob("*") if f.is_file()}
            write_json(flag, {"epoch": resolved.max_epochs, "completed_utc": utc_now().isoformat(),
                              "files": outputs})
        except Exception:
            import traceback
            errors[condition.name] = traceback.format_exc()
    write_json(folder / "training_status.json", {"errors": errors, "status": "failed" if errors else "trained"})
    return arm, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--all", action="store_true", help="Train the full matrix (default)")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run-id")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    spec_path = (ROOT / args.config).resolve()
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    version = spec["experiment_version"]
    base = replace(load_config(spec_path.parent / spec["base_config"], seeds=spec["seeds"],
                               device=args.device, smoke=args.smoke), learning_rate=spec["learning_rate"])
    baseline = verify_v4_reference(spec, base) if version == "V4" else None
    run_id = args.run_id or f"{utc_now():%Y%m%d_%H%M%S}_{version.lower()}_{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id) or args.workers < 1 or (args.resume and not args.run_id):
        parser.error("Invalid run ID, resume arguments or worker count")
    raw = DEFAULT_RUN_ROOT / run_id
    if args.smoke:
        if not args.resume and raw.exists():
            raise FileExistsError(f"Choose a fresh smoke run ID: {run_id}")
        raw = raw / "preflight"
    file = raw / "manifest.json"
    if args.resume:
        meta = json.loads(file.read_text(encoding="utf-8"))
        if meta["config_sha256"] != digest(spec_path) or meta["base"] != base.to_dict():
            raise ValueError("Resume config mismatch")
        if meta["status"] == "completed":
            print("Already validated", run_id)
            return
    else:
        raw.mkdir(parents=True, exist_ok=False)
        shutil.copyfile(spec_path, raw / "config_resolved.yaml")
        active = {p.parent for p in DEFAULT_RUN_ROOT.glob("*/manifest.json")
                  if p.parent.name.startswith(("v3_", "v2_")) and
                  json.loads(p.read_text(encoding="utf-8")).get("status") == "running"}
        prior = [p for parent in ("results/raw/toy_regression", "results/figures", "results/metrics")
                 for p in (ROOT / parent).rglob("*") if p.is_file() and raw not in p.parents
                 and not active.intersection(p.parents)]
        write_json(raw / "prior_files_sha256.json", {p.relative_to(ROOT).as_posix(): digest(p) for p in prior})
        meta = dict(run_id=run_id, source_run_id=run_id, experiment_version=version, smoke=args.smoke,
                    status="running", training_performed=True, imported_from=None,
                    base=base.to_dict(), config_sha256=digest(spec_path), git=git_metadata(),
                    root_layout=sorted(p.name for p in ROOT.iterdir() if p.name != ".git"),
                    code_paths=sorted(p.relative_to(ROOT).as_posix() for p in CONFIG.parents[1].rglob("*.py")),
                    checked_docs=["README.md","results/README.md",".gitignore"],
                    command=["python", "-m", "src.toy_regression.run_v3_sweep", *sys.argv[1:]],
                    started_utc=utc_now().isoformat(), environment=environment(configure_torch(args.device, False)))
        meta["source"] = write_snapshot(raw, run_id, experiment=version)
        if baseline:
            meta.update(baseline_version="V2", baseline_arm="V2-A", baseline_run_id=spec["baseline_run_id"],
                        baseline_path=baseline.relative_to(ROOT).as_posix(), arm_id=next(iter(spec["arms"])),
                        learning_rate=base.learning_rate)
        write_json(file, meta)
        (raw / "data").mkdir()
        for seed in base.outer_seeds:
            splits = generate_data(base, seed)
            save_data(raw / "data" / f"seed_{seed}", splits, base, seed,
                      fgsm_epsilon(splits["train"].x, base.fgsm_epsilon_fraction))
    if baseline:
        for seed in base.outer_seeds:
            for name in (("train", "grid") if base.smoke else ("train", "grid", "test_in", "test_ood")):
                f = f"{name}.npz"
                if digest(raw / "data" / f"seed_{seed}" / f) != digest(baseline / f"seed_{seed}" / "data" / f):
                    raise ValueError(f"Unpaired V2-A/V4 input: {seed}/{name}")
    print("Run", run_id, flush=True)
    # Start the longest arm early so other workers can complete the shorter arms.
    arms = sorted(spec["arms"].items(), key=lambda item: item[1]["max_epochs"] / item[1]["batch_size"],
                  reverse=True)
    failures = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(train_arm, raw, arm, options, base, args.device): arm for arm, options in arms}
        for job in as_completed(jobs):
            arm = jobs[job]
            try:
                _, errors = job.result()
            except Exception as e:
                errors = {"arm": repr(e)}
            if errors:
                failures[arm] = errors
            print(arm, "FAILED" if errors else "trained", flush=True)
    meta.update(status="partial" if failures else "trained", failures=failures, ended_utc=utc_now().isoformat())
    write_json(file, meta)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

