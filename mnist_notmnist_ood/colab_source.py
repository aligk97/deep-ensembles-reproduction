"""Explicit Colab Drive/ZIP checkpoint selection; NEVER fall back to local models."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import stat
import zipfile

DRIVE_CLASSIFICATION = Path("/content/drive/MyDrive/DeepEnsembles_MNIST_Figure2a")
METHODS = ("ensemble", "random", "adversarial", "mc_dropout")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"Required Colab provenance file is missing: {path}. No local fallback or training is permitted.")
    return json.loads(path.read_text(encoding="utf-8"))


def expected_models(config):
    return [(seed, method, member) for seed in config["seeds"] for method in METHODS
            for member in range(1 if method == "mc_dropout" else 5)]


def inspect_result(result, attestation, reference):
    """Verify provenance + all 48 SHA-256 anchors before loading pickle tensors.

    The anchor is the classifier's completed colab_resume_audit.json, recorded
    by the successful MNIST Colab wrapper; hashes are never invented from the
    candidate checkpoints and then treated as an independent reference.
    """
    result = Path(result).resolve()
    manifest = read_json(result / "manifest.json")
    config = reference["mnist_config"]
    if (manifest.get("status") != "complete" or manifest.get("trained_models") != 48
            or manifest.get("result_rows") != 63 or manifest.get("config") != config
            or read_json(result / "config.json") != config
            or manifest.get("signature") != reference["mnist_signature"]
            or manifest.get("source_sha256") != reference["mnist_source_sha256"]):
        raise ValueError("Selected checkpoints do not match the completed final 48-model MNIST experiment")
    environment = manifest.get("environment", {})
    if (environment.get("device") != "cuda" or not environment.get("gpu")
            or not str(environment.get("platform", "")).lower().startswith("linux")):
        raise ValueError("A completed Linux/CUDA Colab run is required; Windows/local checkpoints are rejected")
    if attestation.get("kind") == "drive_session":
        session = attestation["session"]
        if (not session.get("real_colab_training_completed") or session.get("status") != "complete"
                or session.get("scientific_signature") != manifest["signature"]
                or session.get("config") != config):
            raise ValueError("Drive session does not attest a completed real Colab MNIST run")
    elif attestation.get("kind") == "zip_export":
        info = attestation["export_info"]
        if info.get("mode") != "real_colab_completed" or info.get("checkpoints") != 48 or info.get("raw_rows") != 63:
            raise ValueError("Use the complete real MNIST Colab ZIP, not a local validation/report-only archive")
    else:
        raise ValueError("Missing explicit real Colab Drive/ZIP provenance")
    events_path = result / "colab_resume_audit.json"
    events = read_json(events_path)
    expected = expected_models(config)
    indexed = {}
    for row in events:
        key = row.get("seed"), row.get("method"), row.get("member")
        if key in indexed or row.get("epoch") != 10 or row.get("action") not in (
                "trained_new", "resumed_partial", "verified_complete_skipped"):
            raise ValueError("Invalid/duplicated MNIST checkpoint SHA-256 audit entry")
        indexed[key] = row
    if set(indexed) != set(expected):
        raise ValueError("MNIST Colab SHA-256 audit must cover all 48 seed/method/member checkpoints")
    inventory = []
    for seed, method, member in expected:
        path = result / "checkpoints" / f"seed_{seed}" / method / f"member_{member:02d}.pt"
        if not path.is_file():
            raise ValueError(f"Missing Colab checkpoint: seed={seed}, method={method}, member={member}: {path}")
        expected_hash = indexed[seed, method, member].get("sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64 or sha256(path) != expected_hash:
            raise ValueError(f"Checkpoint SHA-256 mismatch: seed={seed}, method={method}, member={member}: {path}")
        inventory.append({"run_seed": seed, "method": method, "member": member,
                          "path": str(path), "sha256": expected_hash})
    if {p.resolve() for p in (result / "checkpoints").rglob("*.pt")} != {Path(row["path"]) for row in inventory}:
        raise ValueError("The selected source must contain exactly the expected 48 checkpoints")
    raw_path = result / "raw_results.csv"
    if not raw_path.is_file():
        raise ValueError("Colab Figure 2(a) raw_results.csv is missing; cannot compare against this checkpoint run")
    with raw_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(int(row["run_seed"]), row["method"], int(row["M"])) for row in rows}
    expected_keys = {(s, method, m) for s in config["seeds"] for method in METHODS for m in range(1, 6)}
    expected_keys |= {(s, "single", 1) for s in config["seeds"]}
    if len(rows) != 63 or keys != expected_keys or any(int(row["n_test"]) != 10000 for row in rows):
        raise ValueError("Colab Figure 2(a) results must contain the full matching 63-row test grid")
    if any(not math.isfinite(float(row[key])) for row in rows for key in ("accuracy", "classification_error", "nll", "brier")):
        raise ValueError("Non-finite values in the Colab Figure 2(a) reference")
    return {"result_root": str(result), "manifest": manifest, "inventory": inventory,
            "manifest_sha256": sha256(result / "manifest.json"),
            "checkpoint_audit_sha256": sha256(events_path), "figure2a_raw_sha256": sha256(raw_path),
            "attestation": attestation, "source_is_real_colab": True}


def drive_source(run_id, reference):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid MNIST RUN_ID")
    folder = DRIVE_CLASSIFICATION / run_id
    if not folder.is_dir():
        raise ValueError(f"Selected MNIST Colab Drive RUN_ID does not exist: {folder}. No fallback to local models or ZIP.")
    session_path = folder / "colab_session.json"
    attestation = {"kind": "drive_session", "run_id": run_id, "session": read_json(session_path),
                   "session_sha256": sha256(session_path)}
    return inspect_result(folder / "mnist_classification/outputs/local", attestation, reference)


def extract_mnist_zip(archive, destination):
    """Bounded allowlisted extraction; no traversal, links, duplicate paths or devices."""
    archive, destination = Path(archive), Path(destination).resolve()
    if not archive.is_file():
        raise ValueError(f"Selected MNIST Colab ZIP not found: {archive}; no alternative source will be chosen")
    with zipfile.ZipFile(archive) as package:
        entries, seen, total = [], set(), 0
        for item in package.infolist():
            name = item.filename.rstrip("/")
            parts = PurePosixPath(name).parts
            mode = item.external_attr >> 16
            if (not parts or name.startswith("/") or "\\" in name or ":" in name
                    or any(p in ("", ".", "..") for p in name.split("/")) or name.casefold() in seen
                    or stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR))):
                raise ValueError(f"Unsafe MNIST ZIP member: {item.filename}")
            if item.file_size > 128 * 1024**2:
                raise ValueError("MNIST ZIP member exceeds checkpoint-size safety bound")
            total += item.file_size
            if total > 4 * 1024**3 or len(seen) >= 10000:
                raise ValueError("MNIST ZIP exceeds size/member limits")
            seen.add(name.casefold())
            entries.append(item)
        infos = [item for item in entries if not item.is_dir() and PurePosixPath(item.filename).name == "export_info.json"]
        if len(infos) != 1:
            raise ValueError("ZIP must be the complete MNIST Colab export with one export_info.json")
        info = json.loads(package.read(infos[0]))
        if info.get("mode") != "real_colab_completed" or info.get("checkpoints") != 48 or info.get("raw_rows") != 63:
            raise ValueError("Rejected report-only/local-validation ZIP; real Colab 48-checkpoint ZIP required")
        prefix = PurePosixPath(infos[0].filename).parent / "mnist_classification"
        for item in entries:
            target = destination / item.filename
            if not target.resolve().is_relative_to(destination):
                raise ValueError("ZIP path escaped extraction root")
            if any(parent.is_symlink() for parent in [target, *target.parents]):
                raise ValueError("Symlink found in ZIP extraction destination")
        # Validate all entries before writing. Existing extracted bytes are verified,
        # never silently replaced by another archive/source.
        for item in entries:
            target = destination / item.filename
            if item.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            content = package.read(item)
            if target.exists():
                if not target.is_file() or hashlib.sha256(content).hexdigest() != sha256(target):
                    raise ValueError(f"Existing ZIP import changed: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
    return destination.joinpath(*prefix.parts), info


def zip_source(archive, import_base, reference):
    archive = Path(archive)
    if not archive.is_file():
        raise ValueError(f"Explicitly selected ZIP is missing: {archive}")
    archive_sha = sha256(archive)
    root, info = extract_mnist_zip(archive, Path(import_base) / archive_sha)
    attestation = {"kind": "zip_export", "archive_path": str(archive), "archive_sha256": archive_sha,
                   "export_info": info}
    return inspect_result(root / "outputs/local", attestation, reference)


def install_bridge(classification_dir, source):
    classification_dir = Path(classification_dir)
    link = classification_dir / "outputs/local"
    if link.exists() or link.is_symlink():
        raise ValueError("Refusing to overwrite a checkpoint bridge")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(Path(source["result_root"]), target_is_directory=True)


def verify_source_unchanged(source):
    for row in source["inventory"]:
        if sha256(row["path"]) != row["sha256"]:
            raise ValueError(f"Checkpoint source changed during OOD inference: {row['path']}")
    result = Path(source["result_root"])
    if (sha256(result / "manifest.json") != source["manifest_sha256"]
            or sha256(result / "colab_resume_audit.json") != source["checkpoint_audit_sha256"]
            or sha256(result / "raw_results.csv") != source["figure2a_raw_sha256"]):
        raise ValueError("MNIST Colab provenance/reference changed during OOD inference")
