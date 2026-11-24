import argparse
import logging
from pathlib import Path
import sys
import traceback

import yaml

from .experiment_config import load_config
from .generate_cubic_data import fgsm_epsilon, generate_data, save_data
from .evaluate_predictions import evaluate_condition, save_metrics_csv
from .plot_experiment_results import plot_seed
from .ensemble_predictions import predict, save_predictions
from .summarize_seed_results import build_aggregate, export_report, load_completed
from .train_v1_v2_models import CONDITIONS, train_condition
from .load_saved_predictions import FIXED_CONFIGS
from .archive_project_versions import index_run_files, write_snapshot
from .experiment_io import (
    DEFAULT_RUN_ROOT, config_hash, configure_torch, environment, git_metadata, new_run_dir, utc_now, write_json,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "v1_baseline.yaml"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT


def provenance(config):
    rows = [
        ("Eğitim verisi", "DE2017 §3.2", "20, x³ + Normal(0, 3²)", "Hayır", "Toy DGP"),
        ("x örneklemesi", "HLA2015 §5.3", "Uniform(-4,4)", "Referanstan", "DE2017 §3.2 aralığı açıkça vermez"),
        ("Gizli katman", "DE2017 §3.2 → HLA2015 §5.3 / §4", "1 × 100, ReLU", "Referanstan",
         "Toy mimari ve referansın ReLU tanımı"),
        ("Kayıp ve varyans", "DE2017 §2.2.1, Şekil 1", "MSE / Gaussian NLL, Softplus + 1e-6",
         "Sayısal seçim", "Pozitif varyans ve normalizasyon sabiti"),
        ("Koşullar", "DE2017 Şekil 1, §2.4", "5 MSE, 1 NLL, 1 NLL+AT, 5 NLL",
         "Hayır", "Eğitim kümesi ortak, bootstrap yok; AT sadece üçüncü panelde"),
        ("Başlatma", "DE2017 §3.1", "PyTorch varsayılan Linear", "Evet",
         "Makalede Torch (Lua) varsayılan başlatması"),
        ("FGSM", "DE2017 §2.3, §3.1, Algoritma 1", "0.01 × gözlenen x aralığı, clean + adv NLL",
         "Hayır", "Hedef aynı, girdiye kırpma yok"),
        ("Optimizasyon", "DE2017 §3.1 genel varsayılan + proje", f"Adam, LR {config.learning_rate:g}; "
         f"batch {config.batch_size} / etkin {min(config.batch_size, config.train_size)}; shuffle={config.shuffle}; "
         f"weight decay={config.weight_decay:g}; gradient clip={config.gradient_clip_norm}",
         "Toy ayarı doğrulanmadı", "Pratik preset küçük batch seçer; genel varsayılan probe LR 0.1 ve batch 100 kullanır"),
        ("Eğitim süresi", "Proje", f"{config.max_epochs} epoch; preset={config.preset}; smoke={config.smoke}",
         "Evet", "Sabit eğitim üst sınırı; toy süresi verilmemiş, erken durdurma yok"),
        ("Seed ve karıştırma", "DE2017 §2.4 + proje", f"{list(config.outer_seeds)}, eşleşmiş üye seed’leri",
         "Seed takvimi eklendi", "Üyeler bağımsız; tam batch sıralaması gradyan dizisini değiştirmez "
         "(kayan nokta yuvarlaması farklı olabilir)"),
        ("Şekil", "DE2017 Şekil 1 + proje", f"[-6,6], {config.plot_points} nokta, ortak y ekseni, ±3σ",
         "Görsel seçim", "Panel düzeni ve işaretler korunur, bant momentlerden hesaplanır"),
        ("Testler", "Proje", f"{config.n_test_in} iç, {config.n_test_ood} OOD; ayrı RNG",
         "Ek analiz", "Eşit sayıda iki OOD tarafı; test etiketleri hiperparametre seçiminde kullanılmaz"),
        ("Olasılık skorları", "DE2017 §2.4 + proje", "Mixture NLL ve CDF kuantilleri; moment Gaussian ayrı; "
         "MSE proxy skoru varyans tabanı 1e-6", "Ek analiz", "MSE bandı ve kayıtları ham anlaşmazlığı korur"),
        ("Yazılım determinizmi", "Proje", f"deterministic={config.deterministic}, CPU tek PyTorch iş parçacığı",
         "Evet", "CPU/GPU arasında bit düzeyinde eşitlik beklenmez"),
    ]
    lines = [
        "# Yöntem kaynağı ve sapmalar", "",
        "DE2017, Lakshminarayanan ve ark. [arXiv v3](https://arxiv.org/abs/1612.01474).",
        "HLA2015, Hernández-Lobato ve Adams [PMLR §5.3](https://proceedings.mlr.press/v37/hernandez-lobatoc15.html).", "",
        "| Seçim | Kaynak | Gerçek değer | Sapma | Gerekçe |", "|---|---|---|---|---|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def run_seed(run_dir, config, device, seed, digest, log):
    seed_dir = run_dir / f"seed_{seed}"
    seed_dir.mkdir()
    splits = generate_data(config, seed)
    train = splits["train"]
    epsilon = fgsm_epsilon(train.x, config.fgsm_epsilon_fraction)
    log.info("Seed %s | dataset FGSM epsilon %.8f | degenerate=%s", seed, epsilon, epsilon == 0)
    save_data(seed_dir / "data", splits, config, seed, epsilon)
    grids, results = {}, []
    for condition in CONDITIONS:
        path = seed_dir / condition.name
        try:
            models = train_condition(condition, config, train, device, epsilon, seed, path, digest, log)
            predictions = {}
            for split_name in ("grid", "test_in", "test_ood"):
                split = splits[split_name]
                prediction = predict(models, split.x, condition.gaussian, device)
                predictions[split_name] = prediction
                save_predictions(path / f"predictions_{split_name}.npz", prediction, split, condition,
                                 seed, config.save_per_member_predictions)
            result = evaluate_condition(condition, predictions, splits, config, seed)
            write_json(path / "metrics.json", result)
            save_metrics_csv(path / "metrics.csv", [result])
            grids[condition.name] = predictions["grid"]
            results.append(result)
        except Exception as error:
            raise RuntimeError(f"Seed {seed}, condition {condition.name}: {error}") from error
    save_metrics_csv(seed_dir / "seed_summary.csv", results)
    plot_seed(seed_dir, config, seed, splits, grids, results)


def command_text(arguments):
    # Produce a copyable command for the platform where this run was launched.
    if sys.platform == "win32":
        return " ".join("'" + argument.replace("'", "''") + "'" if any(
            char.isspace() or char in "'&|;<>()$" for char in argument) else argument for argument in arguments)
    from shlex import join
    return join(arguments)


def run_experiment(config, output_dir=DEFAULT_OUTPUT, *, command=None, experiment_version=None, purpose=None):
    digest = config_hash(config)
    run_dir = new_run_dir(output_dir, digest)
    (run_dir / "logs").mkdir()
    log = logging.Logger("toy_regression")
    handlers = [logging.StreamHandler(sys.stdout),
                logging.FileHandler(run_dir / "logs" / "run.log", encoding="utf-8")]
    for handler in handlers:
        log.addHandler(handler)
    command = command or ["python", "-m", "src.toy_regression.experiment_commands",
                          "--config", str(run_dir / "config_resolved.yaml")]
    version = experiment_version or next((label for label, pair in FIXED_CONFIGS.items()
                                         if pair == (config.learning_rate, config.batch_size)), "custom")
    manifest = {"run_id": run_dir.name, "timestamp_utc": utc_now().isoformat(), "git": git_metadata(),
                "experiment_version": version, "version_basis": "explicit" if experiment_version else "lr_batch",
                "purpose": purpose or f"Şekil 1 için dört regresyon modelini karşılaştırmak ({version}).",
                "command": command, "config_hash": digest, "status": "running", "smoke": config.smoke,
                "outer_seeds": list(config.outer_seeds), "completed_seeds": []}
    manifest_path = run_dir / "manifest.json"
    write_json(manifest_path, manifest)
    log.info("Run directory: %s", run_dir)
    try:
        manifest["source_snapshot"] = write_snapshot(run_dir, run_dir.name, experiment=version)
        write_json(manifest_path, manifest)
        (run_dir / "config_resolved.yaml").write_text(yaml.safe_dump(config.to_dict(), sort_keys=False),
                                                      encoding="utf-8")
        (run_dir / "provenance.md").write_text(provenance(config), encoding="utf-8")
        device = configure_torch(config.device, config.deterministic)
        write_json(run_dir / "environment.json", environment(device))
        log.info("Device: %s | seeds: %s | smoke: %s", device, config.outer_seeds, config.smoke)
        for seed in config.outer_seeds:
            run_seed(run_dir, config, device, seed, digest, log)
            manifest["completed_seeds"].append(seed)
            write_json(manifest_path, manifest)
        build_aggregate(run_dir, config, command_text(command))
        manifest["status"] = "completed"
        log.info("Completed: %s", run_dir)
    except KeyboardInterrupt:
        manifest["status"] = "cancelled"
        log.warning("Run cancelled. Partial artifacts preserved at %s", run_dir)
        raise
    except Exception as error:
        manifest["status"] = "failed"
        write_json(run_dir / "failure.json", {"type": type(error).__name__, "error": str(error),
                                             "traceback": traceback.format_exc()})
        log.exception("Run failed. Partial artifacts preserved at %s", run_dir)
        raise
    finally:
        manifest["finished_at_utc"] = utc_now().isoformat()
        try:
            manifest["file_index"] = index_run_files(run_dir, version)
        finally:
            write_json(manifest_path, manifest)
            for handler in handlers:
                handler.close()
    return run_dir


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train and evaluate the four Figure 1 conditions.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--experiment-version", help="Experiment label, e.g. V2-A")
    parser.add_argument("--purpose", help="What this run is intended to measure")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true", help="Use one seed, at most 3 epochs and 80 points per test split.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Override the distinct outer seeds.")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config, seeds=args.seeds, device=args.device, smoke=args.smoke)
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))
    arguments = sys.argv[1:] if argv is None else argv
    run_experiment(config, args.output_dir, command=["python", "-m", "src.toy_regression.run_training_experiment", *arguments],
                   experiment_version=args.experiment_version, purpose=args.purpose)


def aggregate_main(argv=None):
    parser = argparse.ArgumentParser(description="Read completed run metrics and optionally regenerate or export them.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--rebuild-to", type=Path, help="Write rebuilt aggregate to a new directory.")
    parser.add_argument("--export-report", type=Path, help="Copy report assets beneath this directory.")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    try:
        manifest = load_completed(run_dir)
        if manifest.get("experiment_version") == "V4" and (run_dir / "arms").is_dir():
            from .evaluate_v4_lr_comparison import export_verified_report
            if args.rebuild_to:
                raise ValueError("For V4 regional analysis use evaluate_v4_lr_comparison")
            print(export_verified_report(run_dir, args.export_report))
            return
        if manifest.get("experiment_version") == "V3" and (run_dir / "arms").is_dir():
            from .evaluate_v3_sweep import export_verified_report
            if args.rebuild_to:
                raise ValueError("For V3 regional analysis use evaluate_v3_sweep")
            print(export_verified_report(run_dir, args.export_report))
            return
        aggregate = run_dir / "aggregate"
        command = command_text(manifest["command"])
        if args.rebuild_to:
            config = load_config(run_dir / "config_resolved.yaml")
            if config_hash(config) != manifest["config_hash"]:
                raise ValueError("Resolved configuration does not match the run manifest")
            aggregate = build_aggregate(run_dir, config, command,
                                        destination=args.rebuild_to.resolve())
        if args.export_report:
            target = export_report(run_dir, args.export_report, manifest, aggregate, command)
            print(f"Report assets: {target}")
        print((aggregate / "report_table.csv").read_text(encoding="utf-8"))
        print(f"Aggregate: {aggregate}")
    except (ValueError, OSError, yaml.YAMLError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
