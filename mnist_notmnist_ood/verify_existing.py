"""Record/check SHA-256 for every file in the two protected experiment trees.

Includes ignored files, environments, datasets, and checkpoints. Reads only;
the baseline and final report are written exclusively in this new stage.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    from .data import HERE, digest, write_json
except ImportError:
    from data import HERE, digest, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="Create a baseline BEFORE starting work")
    args = parser.parse_args()
    project = HERE.parent
    baseline_path = HERE / "data" / "protected_before.json"
    if args.record and baseline_path.exists():
        raise RuntimeError("Refusing to overwrite the existing baseline")
    before = None if args.record else json.loads(baseline_path.read_text(encoding="utf-8"))
    current, start = {}, time.monotonic()
    for name in ("toy_regression", "mnist_classification"):
        for path in sorted((project / name).rglob("*")):
            if path.is_file():
                current[path.relative_to(project).as_posix()] = {
                    "size": path.stat().st_size, "sha256": digest(path)}
        print(f"Hashed {name}; {len(current)} files so far", flush=True)
    if args.record:
        write_json(baseline_path, current)
        print(f"Recorded {len(current)} protected files", flush=True)
        return
    added = sorted(set(current) - set(before))
    removed = sorted(set(before) - set(current))
    modified = sorted(p for p in set(current) & set(before) if current[p] != before[p])
    report = {"unchanged": not (added or removed or modified),
              "algorithm": "SHA-256 + byte size + complete file-path set",
              "scope": ["toy_regression/**", "mnist_classification/**"],
              "includes": "All files including .venv, ignored data, outputs and checkpoints",
              "files_before": len(before), "files_after": len(current),
              "bytes_checked": sum(x["size"] for x in current.values()),
              "local_mnist_checkpoint_files": sum(
                  p.startswith("mnist_classification/outputs/local/checkpoints/")
                  and p.endswith(".pt") for p in current),
              "added": added, "removed": removed, "modified": modified,
              "elapsed_seconds": round(time.monotonic() - start, 2)}
    write_json(HERE / "outputs" / "protected_integrity.json", report)
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["unchanged"] else 1)


if __name__ == "__main__":
    main()
