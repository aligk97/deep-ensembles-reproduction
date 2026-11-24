"""Archive the working source or the local project with a verified file catalogue."""

import csv
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import yaml

from .load_saved_predictions import identify_run
from .experiment_io import new_run_dir

ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = ROOT / "results/raw/project_versions"
FIELDS = ("version", "path", "purpose", "bytes", "sha256", "experiment")
PURPOSES = {
    "run_v3_sweep": "V3 matrisini veya eşleşmiş V2 referanslarını başlatır ve sürdürür",
    "train_v3_members": "Aynı mimariyi kullanan bağımsız ağları eğitir ve Adam durumlarını kaydeder",
    "verify_v3_checkpoints": "Checkpoint, veri, kayıp ve test tahminlerinin tutarlılığını denetler",
    "evaluate_v3_sweep": "Bölgesel ölçümleri, doğrulamayı ve rapor çıktısını kaydeder",
    "report_v3_comparisons": "Seed eşlemeli deney farklarını hesaplar ve Türkçe bulguları yazar",
    "plot_v3_results": "V3 seed panellerini ve bölgesel karşılaştırma grafiklerini oluşturur",
    "train_v1_v2_models": "Model parametrelerini eğitir ve üye geçmişini kaydeder",
    "ensemble_predictions": "Üye tahminlerini ve ensemble momentlerini hesaplar",
    "evaluate_predictions": "Tahminlerin hata ve belirsizlik metriklerini hesaplar",
    "v1_v2_regression_networks": "MSE ve Gaussian regresyon ağlarını tanımlar",
    "regression_losses": "MSE, Gaussian NLL ve mixture NLL hesaplar",
    "generate_cubic_data": "Eğitim, test ve çizim verilerini üretir",
    "experiment_config": "Deney ayarlarını okur ve doğrular",
    "experiment_commands": "Deney ve raporlama komutlarını yürütür",
    "run_training_experiment": "Eğitim komutunu başlatır",
    "export_run_results": "Tamamlanmış koşunun toplu sonuçlarını açar",
    "compare_regional_performance": "Kayıtlı tahminlerden bölgesel değerlendirmeyi yürütür",
    "compute_regional_metrics": "Bölgelere göre metrikleri hesaplar",
    "load_saved_predictions": "Kaydedilmiş koşuları, tahminleri ve modelleri okur",
    "plot_regional_comparisons": "Bölgesel sonuç grafiklerini oluşturur",
    "write_regional_report": "Türkçe değerlendirmeyi ve LaTeX tablosunu oluşturur",
    "plot_experiment_results": "Eğitim ve belirsizlik grafiklerini oluşturur",
    "summarize_seed_results": "Seed sonuçlarını toplar ve rapora aktarır",
    "archive_project_versions": "Dosyaların sürümünü, amacını ve doğrulama özetini kaydeder",
    "experiment_io": "Koşu dizinlerini, ortam ve dosya kayıtlarını yönetir",
}


def purpose(path):
    name = path.stem
    if "tests" in path.parts:
        return f"Otomatik doğrulama · {name.removeprefix('test_')}"
    if path.suffix == ".py":
        return PURPOSES.get(name, f"Python paketi veya modülü · {name}")
    if path.parent.name == "data" and path.suffix == ".npz":
        return f"Kaydedilmiş veri kümesi · {name}"
    if name.startswith("predictions_") and path.suffix == ".npz":
        return f"Model tahminleri · {name.removeprefix('predictions_')}"
    if "histories" in path.parts:
        return "Model üyesinin eğitim geçmişi veya seed kaydı"
    types = {".yaml": "Deney parametreleri", ".pt": "Eğitilmiş model checkpoint’i",
             ".ipynb": "Deney notebook’u", ".md": "Proje belgesi veya raporu",
             ".tex": "LaTeX raporu veya tablosu", ".bib": "Kaynakça",
             ".csv": "Tablo", ".json": "Yapılandırılmış kayıt", ".log": "Çalıştırma günlüğü",
             ".png": "Raster grafik", ".pdf": "PDF grafik veya belge", ".zip": "Dosya arşivi",
             ".txt": "Metin kaydı"}
    return f"{types.get(path.suffix, 'Depo dosyası')} · {path.name}"


def experiment_directories():
    labels = {}
    for config in (ROOT / "results/raw/toy_regression").glob("*/config_resolved.yaml"):
        manifest = config.parent / "manifest.json"
        metadata = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
        if (config.parent / "arms").is_dir() and metadata.get("experiment_version"):
            label = metadata["experiment_version"]
        else:
            label, _ = identify_run(config.parent, yaml.safe_load(config.read_text(encoding="utf-8")), metadata)
        labels[config.parent] = label
    for path in (ROOT / "results/metrics").glob("*/manifest.json"):
        metadata = json.loads(path.read_text(encoding="utf-8"))
        labels[path.parent] = labels[ROOT / "results/figures" / path.parent.name] = ", ".join(
            run["configuration"] for run in metadata.get("selected_runs", [])) or metadata.get(
                "experiment_version", "belirtilmemiş")
    return labels


def write_snapshot(destination, version, *, full=False, experiment="ortak"):
    """Publish archive and index only after every archived byte passes hash verification."""
    roots = ["src", "notebooks", "requirements.txt"]
    if full:
        roots += ["results", "report", "docs", "README.md", ".gitignore"]
    archive = destination / ("project.zip" if full else "source.zip")
    index = archive.with_suffix(".csv")
    if archive.exists() or index.exists():
        raise FileExistsError(f"Existing snapshot in {destination}")
    rows = []
    labels = experiment_directories() if full else {}
    presets = {"v1_baseline.yaml": "V1", "v2_a_high_learning_rate.yaml": "V2-A", "v2_b_full_batch.yaml": "V2-B",
               "v2_c_high_learning_rate_full_batch.yaml": "V2-C"}
    with ZipFile(archive.with_suffix(".partial"), "x", compression=ZIP_DEFLATED) as output:
        for name in roots:
            root = ROOT / name
            for path in sorted(root.rglob("*")) if root.is_dir() else [root]:
                if ARCHIVES in path.parents or {"__pycache__", ".pytest_cache", ".ipynb_checkpoints"}.intersection(path.parts):
                    continue
                if not path.is_file() or (not full and path.suffix not in {".py", ".yaml", ".yml", ".ipynb", ".txt"}):
                    continue
                relative = path.relative_to(ROOT).as_posix()
                data = path.read_bytes()
                output.writestr(relative, data)
                label = next((labels[parent] for parent in path.parents if parent in labels),
                             presets.get(path.name, experiment))
                rows.append((version, relative, purpose(path), len(data), sha256(data).hexdigest(), label))
    with ZipFile(archive.with_suffix(".partial")) as saved:
        for _, name, _, _, digest, _ in rows:
            if sha256(saved.read(name)).hexdigest() != digest:
                raise ValueError(f"Archive verification failed for {name}")
    with index.with_suffix(".tmp").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        writer.writerows(rows)
    archive.with_suffix(".partial").rename(archive)
    index.with_suffix(".tmp").rename(index)
    return {"version": version, "archive": archive.name, "index": index.name, "files": len(rows)}


def index_run_files(directory, version):
    """Index the raw input and output files of a finished or partial run."""
    index = directory / "run_files.csv"
    with index.with_suffix(".tmp").open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(FIELDS)
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.name not in {"manifest.json", "run_files.tmp", "run_files.csv"}:
                writer.writerow((directory.name, path.relative_to(directory).as_posix(), purpose(path),
                                 path.stat().st_size, sha256(path.read_bytes()).hexdigest(), version))
    index.with_suffix(".tmp").rename(index)
    return index.name


def main():
    directory = new_run_dir(ARCHIVES, "project")
    write_snapshot(directory, directory.name, full=True)
    print(directory)


if __name__ == "__main__":
    main()
