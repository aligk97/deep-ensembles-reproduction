"""Independently verify V3 checkpoints, compute regional metrics and publish the reports."""
import argparse
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch
import yaml

from .archive_project_versions import index_run_files, write_snapshot
from .experiment_io import DEFAULT_RUN_ROOT, write_json
from .plot_v3_results import render
from .report_v3_comparisons import abbreviate, paired_comparisons, report
from .run_v3_sweep import ROOT, digest
from .verify_v3_checkpoints import require, verify_run


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def export_verified_report(raw, destination):
    metrics = ROOT / "results/metrics" / raw.name
    validation = read(metrics / "validation.json")
    require(validation["status"] == "completed", "Incomplete V3 analysis")
    if destination is None:
        return metrics
    target = Path(destination).resolve() / raw.name
    target.mkdir(parents=True, exist_ok=False)
    for name in ("report_table.tex", "findings_tr.md", "provenance.md", "validation.json"):
        shutil.copy2(metrics / name, target / name)
    figs = ROOT / "results/figures" / raw.name
    for source in [figs / "arm_comparison_rmse_true.pdf", figs / "calibration.pdf",
                   figs / validation["arms"][0]["arm_id"] / "figure1_seed_42.pdf"]:
        shutil.copy2(source, target / source.name)
    return target


def summary(frame):
    keys = [k for k in frame.columns if k not in ("seed","value")]
    return frame.groupby(keys,sort=False,as_index=False).value.agg(value="mean",sample_std="std",seed_count="size")


def save_scores(frame, metrics):
    stats = summary(frame)
    require((stats.seed_count==3).all() and np.isfinite(stats.sample_std).all(), "Incomplete seed groups")
    stats.insert(stats.columns.get_loc("model"), "seeds", "42,123,2026")
    frame.to_csv(metrics/"regional_metrics_per_seed.csv",index=False)
    stats.to_csv(metrics/"regional_metrics_mean_std.csv",index=False)
    frame[frame.region.isin(["in_domain","combined_ood"])].to_csv(metrics/"metrics_by_seed.csv",index=False)
    stats[stats.region.isin(["in_domain","combined_ood"])].to_csv(metrics/"metrics_mean_std.csv",index=False)
    return stats


def check(raw):
    manifest=read(raw/"manifest.json")
    require(manifest["status"] in ("trained","partial","completed"), "Training is still running")
    spec=yaml.safe_load((raw/"config_resolved.yaml").read_text(encoding="utf-8"))
    v3=manifest["experiment_version"]=="V3"
    expected=14 if v3 else 2
    require(len(spec["arms"])==expected and spec["seeds"]==[42,123,2026], "Incomplete matrix")
    frame,arms,hashes,errors=verify_run(raw)
    prior=read(raw/"prior_files_sha256.json")
    changed=[path for path,h in prior.items() if not (ROOT/path).is_file() or digest(ROOT/path)!=h]
    verified=sum(a["verified_models"] for a in arms)
    completed=sum(a["verified_models"]==36 for a in arms)
    result=dict(run_id=raw.name, source_run_id=raw.name, experiment_version=manifest["experiment_version"],
                expected_arms=expected, completed_arms=completed, expected_models=36*expected,
                verified_models=verified, missing_or_failed_models=36*expected-verified,
                trained_models_verified=verified, historical_models_reused=0, training_performed=True,
                imported_from=None, source_sha256=None, seeds=spec["seeds"], arms=arms, data_sha256=hashes,
                previous_files_checked=len(prior), previous_files_changed=changed, errors=errors,
                status="completed" if completed==expected and not changed and not errors else "partial")
    if errors or changed or completed!=expected:
        directory=ROOT/"results/metrics"/raw.name
        directory.mkdir(parents=True,exist_ok=True)
        write_json(directory/"validation.json",result)
        raise ValueError(f"Validation incomplete: {verified} models, {errors[:2]}, {changed[:2]}")
    return frame,result


def equal_inputs(raw, v2, v1, seeds):
    records=[]
    for seed in seeds:
        for name in ("train","test_in","test_ood","grid"):
            path=Path(f"seed_{seed}")/(name+".npz")
            v3_hash=digest(raw/"data"/path)
            v2_hash=digest(v2/"data"/path)
            v1_hash=digest(v1/path.parent/"data"/path.name)
            require(v3_hash==v2_hash==v1_hash, f"Reference data mismatch: {path}")
            records.append(dict(seed=seed, split=name, v1=v1_hash, v2=v2_hash, v3=v3_hash))
    return records


def finish(raw,validation,metrics):
    file=raw/"manifest.json"
    manifest=read(file)
    manifest.update(status=validation["status"], verified_models=validation["verified_models"],
                    validation=(metrics/"validation.json").relative_to(ROOT).as_posix())
    write_json(file,manifest)
    write_json(metrics/"validation.json",validation)
    write_json(metrics/"manifest.json",dict(**manifest, raw_path=raw.relative_to(ROOT).as_posix(),
               analysis_command=["python","-m","src.toy_regression.evaluate_v3_sweep",*sys.argv[1:]],
               analysis_files={f.relative_to(ROOT).as_posix():digest(f) for f in metrics.iterdir()
                               if f.is_file() and f.name!="manifest.json"}))


def register_verified_arms(raw, validation):
    for record in validation["arms"]:
        arm = record["arm_id"]
        directory = raw / "arms" / arm
        params = read(directory / "parameters.json")
        mse_lr = params.pop("mse_learning_rate")
        config = directory / "config_resolved.yaml"
        config.write_text(yaml.safe_dump(params, sort_keys=False), encoding="utf-8")
        write_json(directory / "manifest.json", {
            "run_id": raw.name, "arm_id": arm, "source_run_id": raw.name,
            "version_basis": "explicit", "experiment_version": abbreviate(arm),
            "status": "completed", "smoke": False, "verified_models": record["verified_models"],
            "learning_rate_by_model": {"mse_ensemble_5": mse_lr, "nll_single": params["learning_rate"],
                "nll_ensemble_5": params["learning_rate"], "nll_single_at": params["learning_rate"]},
        })


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--v2-run-id",required=True)
    parser.add_argument("--v1-metrics",type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    raw,v2=[DEFAULT_RUN_ROOT/name for name in (args.run_id,args.v2_run_id)]
    require(all(p.parent==DEFAULT_RUN_ROOT for p in (raw,v2)),"Invalid run id")
    metrics=ROOT/"results/metrics"/raw.name
    figs=ROOT/"results/figures"/raw.name
    previous=metrics/"validation.json"
    if previous.exists() and read(previous)["status"]=="completed":
        raise FileExistsError("A validated analysis already exists")
    df,validation=check(raw)
    ref,v2_validation=check(v2)
    for p in (metrics,figs):
        p.mkdir(parents=True,exist_ok=True)
    v2_metrics=ROOT/"results/metrics"/v2.name
    v2_metrics.mkdir(parents=True,exist_ok=True)
    v1_table=(ROOT/args.v1_metrics).resolve()
    historical=pd.read_csv(v1_table)
    historical=historical[historical.configuration.eq("V1")].copy()
    source_ids=historical.source_run.unique()
    require(len(source_ids)==1,"Ambiguous V1 reference")
    v1=DEFAULT_RUN_ROOT/source_ids[0]
    historical=historical.assign(arm_id="v1",source_run_id=source_ids[0],experiment_version="V1")
    v1_config=yaml.safe_load((v1/"config_resolved.yaml").read_text(encoding="utf-8"))
    for name in ("batch_size","learning_rate","weight_decay"):
        historical[name]=v1_config[name]
    historical["epochs"]=v1_config["max_epochs"]
    historical["optimizer_steps"]=historical.epochs*4
    validation["reference_input_hashes"]=equal_inputs(raw,v2,v1,validation["seeds"])
    stats=save_scores(df,metrics)
    save_scores(ref,v2_metrics)
    comparisons=paired_comparisons(df,pd.concat([ref,historical],ignore_index=True))
    comparisons.to_csv(metrics/"comparisons.csv",index=False)
    table=stats[(stats.model=="nll_ensemble_5") & stats.metric.isin(["rmse_true","coverage_95","predictive_nll"])]
    printable=table.pivot(index=["arm_id","region"],columns="metric",values="value").reset_index()
    printable.to_latex(metrics/"report_table.tex",index=False,escape=True,float_format="%.3f")
    report(raw,metrics,stats,comparisons)
    render(raw,figs,df)
    validation["figures"]={p.relative_to(ROOT).as_posix():digest(p) for p in sorted(figs.rglob("*")) if p.is_file()}
    require(len(validation["figures"])==2*(14*(3+1)+5),"Incomplete figure set")
    for p in figs.rglob("*"):
        if p.is_file():
            with p.open("rb") as stream:
                signature=stream.read(8)
            require(p.stat().st_size>1000 and (signature.startswith(b"%PDF") if p.suffix==".pdf" else
                    signature.startswith(b"\x89PNG")), f"Invalid figure {p}")
    validation.update(regional_rows=len(df), comparison_rows=len(comparisons),summary_rows=len(stats),
                      regions=sorted(df.region.unique()), all_three_seeds_verified=True,
                      trained_models=validation["verified_models"], missing_arms=0,
                      methods=sorted(df.model.unique()),
                      csv_paths=sorted(f.relative_to(ROOT).as_posix() for f in metrics.glob("*.csv")),
                      reference_runs={"V1":v1.name,"V2":v2.name})
    provenance=[
        "# V3 kaynak ve koşu kaydı","",
        f"Özgün görev CODEX_V3_GITHUB_DUZENI.md. Aranan tarihsel V3 koşusu 20261118_203659_bb6b3a0d. Yerel depo ve erişilebilir arşivlerde bu kimliğin checkpoint, tahmin ve doğrulama dosyaları bulunamadı. Yeni koşu {raw.name}.",
        f"V1 yerel kaynak koşu {v1.name}. Kaydedilmiş bölgesel metrikler {v1_table.relative_to(ROOT).as_posix()}. Bu kaynağın SHA-256 değeri {digest(v1_table)}.",
        f"V2-A ve V2-C için geçmiş koşu 20261118_133543_5d8d6017 bulunamadı. 72 referans model ayrı V2 koşusunda yeniden üretildi ({v2.name}). Batch 100 fiilen 20 örnektir.",
        "V3 verileri proje üreticisiyle yeniden oluşturuldu. Üç seed için V1/V2/V3 eğitim, iç test, dış test ve grid NPZ dosyalarının SHA-256 değerleri eşleşti. Özgün tarihsel V3 ile bit düzeyinde eşdeğerlik ölçülemedi.",
        "Aynı mimari, kayıp ve Adam kullanıldı. Bağımsız üye parametreleri CPU üzerinde PyTorch vmap ile birlikte işlendi. Minibatch sırası üye ve seed’e özel PyTorch DataLoader üretecinden gelir. Tarihsel cihaz ve aritmetik aynı olmayabilir.",
        "Git revizyonu, ortam ve eğitim komutu ham manifestte; kaynak dosyalarının yolları ve hash’leri source.csv dosyasında kayıtlıdır. validation.json önceki sonuç dosyalarının kontrolünü içerir.",
        "Bölgesel ölçümler için iki OOD tarafında 1.000’er ve eğitim aralığında 2.000 bağımsız gürültülü test noktası kullanıldı. Üç seed ortalaması ve örneklem standart sapması raporlandı."
    ]
    (metrics/"provenance.md").write_text("\n\n".join(provenance)+"\n",encoding="utf-8")
    for directory, result in ((raw, validation), (v2, v2_validation)):
        register_verified_arms(directory, result)
        index_run_files(directory,read(directory/"manifest.json")["experiment_version"])
    if not (metrics/"source.zip").exists():
        write_snapshot(metrics,raw.name,experiment="V3")
    finish(v2,v2_validation,v2_metrics)
    finish(raw,validation,metrics)
    print(metrics,flush=True)


if __name__=="__main__":
    main()

