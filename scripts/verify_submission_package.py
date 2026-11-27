"""Verify a final submission directory or ZIP without executing experiments."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("Unsafe member path: " + name)
    return path


def verify(path):
    count = 0
    total = 0
    if path.is_dir():
        manifest = json.loads((path / "MANIFEST_SHA256.json").read_text(encoding="utf-8"))
        expected = {item["file"] for item in manifest["files"]}
        actual = {p.relative_to(path).as_posix() for p in path.rglob("*") if p.is_file()}
        if actual != expected | {"MANIFEST_SHA256.json"}:
            raise AssertionError("Directory file list differs from manifest")
        items = []
        for item in manifest["files"]:
            safe_name(item["file"])
            target = path / item["file"]
            if target.is_symlink():
                raise AssertionError("Symlink in package")
            items.append((item, target.read_bytes()))
    else:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise AssertionError("Duplicate ZIP members")
            for name in names:
                safe_name(name)
            if archive.testzip() is not None:
                raise AssertionError("ZIP CRC mismatch")
            prefix = "FINAL_SUBMISSION/"
            manifest = json.loads(archive.read(prefix + "MANIFEST_SHA256.json"))
            expected = {prefix + item["file"] for item in manifest["files"]}
            if set(names) != expected | {prefix + "MANIFEST_SHA256.json"}:
                raise AssertionError("ZIP file list differs from manifest")
            items = []
            for item in manifest["files"]:
                safe_name(item["file"])
                items.append((item, archive.read(prefix + item["file"])))
    for item, data in items:
        if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
            raise AssertionError("File hash mismatch: " + item["file"])
        count += 1
        total += len(data)
    if manifest["algorithm"] != "SHA-256":
        raise AssertionError("Unexpected hash algorithm")
    return {"successful": True, "verified_files": count, "verified_bytes": total,
            "manifest_self_hash_excluded": True, "model_forwards": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
