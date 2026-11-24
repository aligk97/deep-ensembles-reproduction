"""Read-only discovery of toy regression runs, test arrays and checkpoints."""

import csv
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from zipfile import BadZipFile, ZipFile

import numpy as np
import torch
import yaml

from .v1_v2_regression_networks import GaussianRegressor, MSERegressor
from .ensemble_predictions import aggregate_predictions, predict

SEEDS = (42, 123, 2026)
SPLITS = ("test_in", "test_ood")
FIXED_CONFIGS = {
    "V1": (.001, 5), "V2-A": (.1, 5), "V2-B": (.001, 100), "V2-C": (.1, 100),
}
PREDICTION_FIELDS = ("mean", "total_variance", "aleatoric", "epistemic", "member_means", "member_variances")


@dataclass(frozen=True)
class Run:
    path: Path
    label: str
    version: str
    config: dict
    manifest: dict

    @property
    def floor(self):
        return float(self.config.get("variance_floor", 1e-6))

    def seed_path(self, seed):
        return self.path / f"seed_{seed}"


def identify_run(path, config, manifest):
    if manifest.get("version_basis") == "explicit" and manifest.get("experiment_version"):
        label = manifest["experiment_version"]
        if label in FIXED_CONFIGS and (config.get("learning_rate"), config.get("batch_size")) != FIXED_CONFIGS[label]:
            raise ValueError(f"{label} conflicts with saved learning rate/batch")
        return label, label.split("-")[0]
    labels = " ".join(str(source.get(key, "")) for source in (manifest, config)
                      for key in ("configuration", "config_name", "label", "version", "variant", "preset"))
    command = manifest.get("command", [])
    labels = " ".join((labels, path.name, path.parent.name,
                       command if isinstance(command, str) else " ".join(command)))
    settings = (config.get("learning_rate"), config.get("batch_size"))
    explicit = re.search(r"(?<![a-z0-9])v2[ _-]*([abc])(?![a-z0-9])", labels, re.I)
    version = str(manifest.get("version", config.get("version", ""))).lower()
    if version == "3" or re.search(r"(?<![a-z0-9])v3(?![a-z])", labels, re.I):
        # A manifest label or preset can name the configuration, even if the run directory is a timestamp.
        for source in (manifest, config):
            for key in ("configuration", "config_name", "label", "variant", "preset"):
                name = str(source.get(key, "")).strip()
                if name and name.lower() not in {"v3", "3", "practical", "paper_default_probe"}:
                    return (name if re.match(r"v3", name, re.I) else f"V3-{name}"), "V3"
        name = path.name if re.match(r"v3[ _-]?\w", path.name, re.I) else path.parent.name
        suffix = re.sub(r"^v3[ _-]*", "", name, flags=re.I)
        return f"V3-{suffix or path.name}", "V3"
    if explicit:
        label = f"V2-{explicit[1].upper()}"
        if all(value is not None for value in settings) and settings != FIXED_CONFIGS[label]:
            raise ValueError(f"{label} conflicts with saved learning rate/batch {settings}")
        return label, "V2"
    for label, pair in FIXED_CONFIGS.items():
        if pair == settings:
            return label, label.split("-")[0]
    if version == "1" or re.search(r"(?<![a-z0-9])v1(?![a-z0-9])", labels, re.I):
        return "V1", "V1"
    return f"Unclassified-{path.name}", "Unclassified"


def discover(roots):
    runs, audit, archives = {}, [], []
    paths = set()
    for root in roots:
        for candidate in (*root.rglob("config_resolved.yaml"), *root.rglob("manifest.json")):
            path = candidate.parent
            if any(path.glob("seed_*")) or (path / "config_resolved.yaml").is_file():
                paths.add(path.resolve())
    for path in sorted(paths):
        manifest_path = path / "manifest.json"
        config_path = path / "config_resolved.yaml"
        entry = {"path": str(path)}
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.is_file() else {}
            entry.update(status=manifest.get("status", "unknown"), smoke=bool(
                manifest.get("smoke") or config.get("smoke")))
            if "arms" in config and (path / "arms").is_dir():
                entry["decision"] = "matrix_container"
            elif entry["smoke"] or entry["status"] in ("running", "cancelled", "failed"):
                entry["decision"] = "excluded_smoke_or_unfinished"
            else:
                label, version = identify_run(path, config, manifest)
                run = Run(path, label, version, config, manifest)
                previous = runs.get(label)
                if previous is not None:
                    previous_hash = previous.manifest.get("config_hash")
                    current_hash = manifest.get("config_hash")
                    if label not in FIXED_CONFIGS and previous_hash != current_hash:
                        label = f"{label}-{path.name}"
                        run = Run(path, label, version, config, manifest)
                        previous = None
                entry.update(label=label, decision="selected")
                if previous is not None:
                    def priority(candidate):
                        return (sum(candidate.seed_path(seed).is_dir() for seed in SEEDS),
                                candidate.manifest.get("status") == "completed",
                                candidate.manifest.get("finished_at_utc", ""), str(candidate.path))

                    if priority(previous) >= priority(run):
                        entry["decision"] = "alternate_run"
                    else:
                        next(item for item in audit if item["path"] == str(previous.path))["decision"] = "alternate_run"
                if entry["decision"] == "selected":
                    runs[label] = run
        except (OSError, TypeError, ValueError, AttributeError, yaml.YAMLError) as error:
            entry.update(decision="invalid_metadata", reason=str(error))
        audit.append(entry)
    for archive in sorted({path.resolve() for root in roots for path in root.rglob("*.zip")}):
        try:
            with ZipFile(archive) as handle:
                paths = [name for name in handle.namelist() if Path(name).name == "config_resolved.yaml"]
            archives.append({"path": str(archive), "run_config_entries": paths})
        except (OSError, BadZipFile, ValueError, RuntimeError) as error:
            archives.append({"path": str(archive), "error": str(error)})
    return dict(sorted(runs.items())), audit, archives


def file_hash(path):
    with path.open("rb") as stream:
        return sha256(stream.read()).hexdigest()


def first_file(directory, base):
    return next((path for suffix in (".npz", ".csv")
                 if (path := directory / (base + suffix)).is_file()), None)


def read_arrays(path):
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            keys = ("x", "y_true", "y_noisy", "condition", "outer_seed", *PREDICTION_FIELDS)
            return {key: archive[key] for key in keys if key in archive.files}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Empty sample CSV")
    columns = {key: np.asarray([float(row[key]) for row in rows])
               for key in ("x", "y_true", "y_noisy", *PREDICTION_FIELDS[:4]) if key in rows[0]}
    for key in PREDICTION_FIELDS[4:]:
        names = sorted((name for name in rows[0] if re.fullmatch(rf"{key}_\d+", name)),
                       key=lambda name: int(name.rsplit("_", 1)[-1]))
        if names:
            columns[key] = np.asarray([[float(row[name]) for row in rows] for name in names])
    return columns


def column(value, name):
    result = torch.as_tensor(value, dtype=torch.float64)
    if result.ndim == 1:
        result = result[:, None]
    if result.ndim != 2 or result.shape[1] != 1 or not result.isfinite().all():
        raise ValueError(f"Invalid {name} shape or values: {tuple(result.shape)}")
    return result


@dataclass(frozen=True)
class Samples:
    x: torch.Tensor
    y: torch.Tensor | None
    source: Path
    excluded: int = 0

    def fingerprint(self, mask):
        values = torch.cat((self.x[mask], self.y[mask]), dim=1).numpy()
        values = values[np.lexsort((values[:, 1], values[:, 0]))].astype("<f8")
        return sha256(values.tobytes()).hexdigest()


def read_samples(path, arrays, require_noisy):
    x = column(arrays["x"], "x")
    y = column(arrays["y_noisy"], "y_noisy") if "y_noisy" in arrays else None
    if (require_noisy and y is None) or (y is not None and len(x) != len(y)):
        raise ValueError("Saved noisy targets are missing or misaligned")
    mask = ((x >= -6) & (x <= 6)).flatten()
    if not mask.any():
        raise ValueError("No saved x in [-6,6]")
    return Samples(x[mask], y[mask] if y is not None else None, path, int((~mask).sum()))


def native_samples(seed_path, split, conditions, read, issue):
    candidates = [first_file(seed_path / "data", split)]
    candidates.extend(first_file(seed_path / condition.name, f"predictions_{split}") for condition in conditions)
    for path in filter(None, candidates):
        try:
            return read_samples(path, read(path), require_noisy=split != "grid")
        except (KeyError, OSError, ValueError, TypeError) as error:
            issue(f"{path}: {error}")
    return None


def aligned_indices(source_x, target_x):
    if torch.equal(source_x, target_x):
        return torch.arange(len(source_x))
    source, target = source_x.flatten(), target_x.flatten()
    ordered, permutation = source.sort()
    locations = torch.searchsorted(ordered, target)
    if len(ordered) and (locations < len(ordered)).all() and torch.equal(ordered[locations], target):
        return permutation[locations]
    return None


def read_prediction(arrays, target, condition, seed):
    if ("condition" in arrays and str(arrays["condition"].item()) != condition.name
            or "outer_seed" in arrays and int(arrays["outer_seed"].item()) != seed):
        raise ValueError("Prediction metadata disagrees with model or seed")
    x = column(arrays["x"], "prediction x")
    indices = aligned_indices(x, target.x)
    if indices is None:
        raise ValueError("Saved predictions do not cover the evaluation inputs")
    fields = {}
    for key in PREDICTION_FIELDS:
        if key not in arrays:
            continue
        if key.startswith("member_"):
            members = torch.as_tensor(arrays[key], dtype=torch.float64)
            if members.ndim == 2:
                members = members[..., None]
            if members.shape != (5 if condition.ensemble else 1, len(x), 1):
                raise ValueError(f"Unexpected {key} shape: {tuple(members.shape)}")
            values = members[:, indices]
        else:
            values = column(arrays[key], key)
            if len(values) != len(x):
                raise ValueError(f"{key} and x have different sample counts")
            values = values[indices]
        if not values.isfinite().all() or (key == "member_variances" and (values <= 0).any()):
            raise ValueError(f"Invalid {key}")
        fields[key] = values
    if "member_means" in fields and (not condition.gaussian or "member_variances" in fields):
        moments = aggregate_predictions(fields["member_means"], fields.get("member_variances"))
        for key, values in vars(moments).items():
            if values is not None:
                if key in fields and not torch.allclose(fields[key], values, rtol=3e-5, atol=1e-5):
                    raise ValueError(f"Saved {key} disagrees with member predictions")
                fields[key] = values
    if "mean" not in fields:
        raise ValueError("Prediction mean is missing")
    if "total_variance" in fields:
        if not condition.gaussian:
            fields.setdefault("epistemic", fields["total_variance"])
        elif not condition.ensemble:
            fields.setdefault("aleatoric", fields["total_variance"])
    elif "aleatoric" in fields and "epistemic" in fields:
        fields["total_variance"] = fields["aleatoric"] + fields["epistemic"]
    for key in ("total_variance", "aleatoric", "epistemic"):
        if key in fields and ((fields[key] < 0).any() or
                              key == "total_variance" and condition.gaussian and (fields[key] == 0).any()):
            raise ValueError(f"Invalid {key}")
    if "member_means" not in fields:
        fields.pop("member_variances", None)
    return fields


def load_members(run, seed, condition, record):
    paths = [run.seed_path(seed) / condition.name / "models" / f"member_{index}.pt"
             for index in range(5 if condition.ensemble else 1)]
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Missing {', '.join(path.name for path in paths if not path.is_file())}")
    models = []
    for index, path in enumerate(paths):
        record(path)
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        for key, expected in (("condition", condition.name), ("outer_seed", seed), ("member", index),
                              ("config_hash", run.manifest.get("config_hash"))):
            if expected is not None and key in checkpoint and checkpoint[key] != expected:
                raise ValueError(f"{path.name}: invalid {key}")
        hidden = checkpoint.get("hidden_dim", run.config.get("hidden_dim", 100))
        if condition.gaussian:
            model = GaussianRegressor(hidden, checkpoint.get("variance_floor") or run.floor)
        else:
            model = MSERegressor(hidden)
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint), strict=True)
        models.append(model.eval())
    return models


def infer(models, target, condition):
    output = predict(models, target.x.float(), condition.gaussian, torch.device("cpu"))
    members = output.member_means.double()
    variances = output.member_variances.double() if condition.gaussian else None
    return {name: values for name, values in vars(aggregate_predictions(members, variances)).items()
            if values is not None}
