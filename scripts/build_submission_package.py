"""Build and verify the outer delivery ZIP; never run experiment code."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from verify_submission_package import verify


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root):
    root = root.resolve(strict=True)
    if root.name != "FINAL_SUBMISSION" or not root.is_dir():
        raise ValueError("Pass the FINAL_SUBMISSION directory explicitly")
    for required in ("01_REPORT", "02_OVERLEAF_SOURCE", "03_PRESENTATION",
                     "04_GITHUB_REPOSITORY", "05_STUDY_AND_DEFENSE", "06_EXPERIMENT_EVIDENCE"):
        if not (root / required).is_dir():
            raise ValueError("Missing delivery directory: " + required)
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Symlink in delivery")
        if not path.is_file() or path.name == "MANIFEST_SHA256.json":
            continue
        relative = path.relative_to(root)
        if any(part in {".git", ".venv", "__pycache__", ".deps", "deps"} for part in relative.parts):
            raise ValueError("Unwanted build/dependency directory: " + str(relative))
        if path.suffix.lower() in {".pt", ".pth", ".npz", ".npy"}:
            raise ValueError("Unwanted model/bulk cache copy: " + str(relative))
        records.append({"file": relative.as_posix(), "bytes": path.stat().st_size,
                        "sha256": digest(path)})
    manifest = root / "MANIFEST_SHA256.json"
    manifest.write_text(json.dumps({
        "algorithm": "SHA-256", "manifest_self_hash_excluded": True,
        "scope": "Every delivered file except this manifest; no original experiment is rerun",
        "files": records,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    folder_result = verify(root)
    target = root.parent / "FINAL_SUBMISSION.zip"
    temporary = root.parent / "FINAL_SUBMISSION.zip.tmp"
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, "FINAL_SUBMISSION/" + path.relative_to(root).as_posix())
    zip_result = verify(temporary)
    temporary.replace(target)
    archive_hash = digest(target)
    target.with_suffix(".zip.sha256").write_text(archive_hash + "  " + target.name + "\n", encoding="ascii")
    return {"successful": True, "zip_crc_passed": True, "zip_sha256": archive_hash,
            "archive_bytes": target.stat().st_size, "archive_files": len(records) + 1,
            "directory_verification": folder_result, "archive_verification": zip_result,
            "model_forwards": 0, "new_training": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, help="Optional external receipt, outside FINAL_SUBMISSION")
    args = parser.parse_args()
    if args.output and args.output.resolve().is_relative_to(args.root.resolve()):
        parser.error("Receipt must be outside package to avoid self-reference")
    result = build(args.root)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
