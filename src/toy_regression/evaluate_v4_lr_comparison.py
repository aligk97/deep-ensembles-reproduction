"""Verify both sets of checkpoints and compare the V4 Adam LR with the saved V2-A reference."""
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
from .evaluate_v3_sweep import read, summary
from .experiment_io import DEFAULT_RUN_ROOT, write_json
from .plot_v4_lr_comparison import render_v4
from .report_v4_lr_comparison import MAIN, write_report
from .run_v3_sweep import ROOT, digest
from .verify_v3_checkpoints import require, verify_run

KEYS = ["seed", "model", "region", "metric", "distribution", "n_samples"]


def pair_scores(reference, current):
    selected = KEYS + ["value"]
    pairs = reference[selected].merge(current[selected], on=KEYS, suffixes=("_v2a","_v4"),
                                      how="outer", validate="one_to_one", indicator=True)
    require((pairs["_merge"]=="both").all(), "Unpaired seed, model, region or metric")
    pairs = pairs.drop(columns="_merge")
    pairs["delta"] = pairs.value_v4 - pairs.value_v2a
    table = pairs.groupby(KEYS[1:], sort=False, as_index=False).agg(
        v2a_mean=("value_v2a","mean"), v2a_std=("value_v2a","std"),
        v4_mean=("value_v4","mean"), v4_std=("value_v4","std"),
        paired_mean_delta=("delta","mean"), paired_std_delta=("delta","std"), n_seeds=("seed","size"))
    require(table.n_seeds.eq(3).all(), "Every summary needs three matched seeds")
    return pairs, table


def verify_controlled_settings(reference, arm, seeds):
    base, current = [read(p / "parameters.json") for p in (reference, arm)]
    diff = {k: [base.get(k),current.get(k)] for k in set(base)|set(current) if base.get(k)!=current.get(k)}
    require(set(diff)=={"learning_rate","mse_learning_rate"}, "Settings other than LR changed")
    checked = 0
    for path in arm.glob("seed_*/*/models/member_*.pt"):
        checkpoints = [torch.load(p,map_location="cpu",weights_only=True) for p in
                       (reference/path.relative_to(arm),path)]
        a,b = checkpoints
        require(all(a[k]==b[k] for k in ("outer_seed","member","model_seed","shuffle_seed","condition",
                                         "global_step","final_epoch","weight_decay")), "Member pairing mismatch")
        require(torch.equal(a["shuffle_rng_state"],b["shuffle_rng_state"]), "Shuffle stream mismatch")
        groups = [c["optimizer_state_dict"]["param_groups"][0].copy() for c in checkpoints]
        for g in groups:
            g.pop("lr")
        require(groups[0]==groups[1], "Adam settings differ")
        checked += 1
    require(checked==36, "Incomplete member pairing")
    hashes = []
    for seed in seeds:
        for split in ("train","grid","test_in","test_ood"):
            suffix = Path(f"seed_{seed}")/"data"/(split+".npz")
            a,b = [digest(p/suffix) for p in (reference,arm)]
            require(a==b, f"V2-A and V4 inputs differ: {suffix}")
            hashes.append(dict(seed=seed,split=split,v2a_sha256=a,v4_sha256=b))
    return diff, hashes


def export_verified_report(raw, destination):
    metrics = ROOT/"results/metrics"/raw.name
    require(read(metrics/"validation.json")["status"]=="completed", "Incomplete V4 evaluation")
    if destination is None:
        return metrics
    target = Path(destination).resolve()/raw.name
    target.mkdir(parents=True, exist_ok=False)
    for name in ("v4_results.tex","findings_tr.md","provenance.md","validation.json"):
        shutil.copy2(metrics/name,target/name)
    figs = ROOT/"results/figures"/raw.name
    for name in ("figure1_v2a_vs_v4_seed_42.pdf","coverage_width.pdf","rmse_nll.pdf"):
        shutil.copy2(figs/name,target/name)
    return target


def evaluate(run_id):
    torch.set_num_threads(1)
    raw = DEFAULT_RUN_ROOT/run_id
    require(raw.parent==DEFAULT_RUN_ROOT and raw.is_dir(), "Invalid run ID")
    meta = read(raw/"manifest.json")
    require(meta["experiment_version"]=="V4" and not meta["smoke"], "Expected a full V4 run")
    require(meta["status"] in ("trained","partial"), "Run is already validated or still training")
    reference = ROOT/meta["baseline_path"]
    ref_raw = reference.parents[1]
    arm = raw/"arms"/meta["arm_id"]
    metrics, figs = [ROOT/"results"/kind/raw.name for kind in ("metrics","figures")]
    metrics.mkdir(exist_ok=True,parents=True)
    figs.mkdir(exist_ok=True,parents=True)
    ref, ref_records, _, ref_errors = verify_run(ref_raw, selected_arm=reference.name, overall=True)
    current, records, _, errors = verify_run(raw, overall=True)
    prior = read(raw/"prior_files_sha256.json")
    changed = [f for f,h in prior.items() if not (ROOT/f).is_file() or digest(ROOT/f)!=h]
    root_layout = sorted(p.name for p in ROOT.iterdir() if p.name!=".git")
    validation = dict(run_id=raw.name, experiment_version="V4", arm_id=arm.name,
                      learning_rate=meta["learning_rate"], baseline_version="V2", baseline_arm="V2-A",
                      baseline_run_id=ref_raw.name, git_commit=meta["git"]["commit"], expected_new_models=36,
                      verified_new_models=records[0]["verified_models"], records=records,
                      baseline_models_reused=ref_records[0]["verified_models"], baseline_retrained=0,
                      errors=errors+ref_errors, training_failures=meta.get("failures",{}),
                      seeds=meta["base"]["outer_seeds"], previous_files_checked=len(prior),
                      previous_files_changed=changed, root_layout_matches=root_layout==meta["root_layout"])
    status_ok = (validation["verified_new_models"]==validation["baseline_models_reused"]==36
                 and not validation["errors"] and not changed and not validation["training_failures"]
                 and validation["root_layout_matches"])
    if not status_ok:
        validation["status"]="partial"
        for path in (metrics,raw):
            write_json(path/"validation.json",validation)
        raise ValueError("Partial run; see validation.json")
    diff, hashes = verify_controlled_settings(reference, arm, validation["seeds"])
    archived_path = ROOT/"results/metrics"/ref_raw.name/"regional_metrics_per_seed.csv"
    baseline_csv = pd.read_csv(archived_path)
    archived = baseline_csv[baseline_csv.arm_id==reference.name].set_index(KEYS).value.sort_index()
    checked = ref[ref.region!="overall"].set_index(KEYS).value.sort_index()
    require(archived.index.equals(checked.index), "Archived reference metric keys disagree")
    np.testing.assert_allclose(archived,checked,rtol=1e-12,atol=1e-12)
    pairs, table = pair_scores(ref,current)
    context = dict(baseline_version="V2",baseline_arm="V2-A",baseline_source_run_id=ref_raw.name,
                   git_commit=meta["git"]["commit"])
    current = current.assign(**context)
    stats = summary(current)
    common = dict(run_id=raw.name,source_run_id=raw.name,experiment_version="V4",arm_id=arm.name,
                  learning_rate=meta["learning_rate"],baseline_learning_rate=.1,**context)
    pairs = pairs.assign(**common)
    table = table.assign(**common)
    region_table = table[table.region!="overall"]
    for filename, frame in (("v4_per_seed",current), ("v4_mean_std",stats),
                            ("v2a_vs_v4_paired_seeds",pairs), ("v2a_vs_v4_regional",region_table),
                            ("v2a_vs_v4_overall",table[table.region=="overall"])):
        frame.to_csv(metrics/(filename+".csv"),index=False)
    latex = region_table[region_table.metric.isin(MAIN)].copy()
    for key in ("v2a","v4"):
        latex[key] = [f"{m:.3f} ± {s:.3f}" for m,s in zip(latex[key+"_mean"],latex[key+"_std"])]
    latex[["model","region","metric","v2a","v4"]].to_latex(metrics/"v4_results.tex",index=False,escape=True,
        longtable=True,caption=f"V2-A ve V4, üç seed ortalaması ± SS. Kapsama 0–1 ölçeğindedir. "
                              f"{raw.name}, {arm.name}, LR {meta['learning_rate']}, V2-A {ref_raw.name}, Git {meta['git']['commit'][:8]}.")
    write_report(metrics,meta,region_table,pairs)
    render_v4(reference,arm,region_table,figs,meta["learning_rate"])
    pictures = sorted(figs.glob("*"))
    expected = {f"figure1_v2a_vs_v4_seed_{s}" for s in validation["seeds"]} | {
                "rmse_nll","coverage_width","variance_decomposition"}
    require({p.name for p in pictures}=={name+ext for name in expected for ext in (".pdf",".png")},
            "Incomplete figures")
    for p in pictures:
        with p.open("rb") as stream:
            require(stream.read(4)==(b"%PDF" if p.suffix==".pdf" else b"\x89PNG") and p.stat().st_size>1000,
                    f"Invalid figure {p.name}")
    provenance = ["# V4 kaynak kaydı", "",
        f"Yeni eğitim `{raw.relative_to(ROOT).as_posix()}`. Referans `{reference.relative_to(ROOT).as_posix()}`.",
        f"Referans CSV `{archived_path.relative_to(ROOT).as_posix()}`. SHA-256 `{digest(archived_path)}`. "
        "Checkpoint’ten tekrar hesaplanan metrikler bu CSV ile karşılaştırıldı.",
        f"Eğitim komutu: `{' '.join(meta['command'])}`.",
        f"Değerlendirme komutu: `python -m src.toy_regression.evaluate_v4_lr_comparison --run-id {raw.name}`.",
        f"Git revizyonu `{meta['git']['commit']}`. Eğitim ve değerlendirme kaynak kopyaları ayrı source.zip ve source.csv dosyalarındadır.",
        "CSV’de kapsama 0–1 ölçeğindedir. Paired delta = aynı seed için V4 eksi V2-A. Genel metriklerde 4.000 test noktası birlikte değerlendirilir."]
    (metrics/"provenance.md").write_text("\n\n".join(provenance)+"\n",encoding="utf-8")
    validation.update(status="completed", controlled_parameter_differences=diff, paired_data=hashes,
                      verified_member_pairs=36, finite_losses_gradients_and_predictions=True,
                      verified_final_epoch=3000, verified_adam_steps=12000, baseline_metric_rows_checked=len(checked),
                      archive_reference_sha256=digest(archived_path), figures={p.relative_to(ROOT).as_posix():digest(p) for p in pictures})
    arm_params = read(arm/"parameters.json")
    arm_params.pop("mse_learning_rate")
    (arm/"config_resolved.yaml").write_text(yaml.safe_dump(arm_params,sort_keys=False),encoding="utf-8")
    write_json(arm/"manifest.json",dict(**common, version_basis="explicit",status="completed", verified_models=36))
    for folder in (metrics, raw):
        write_json(folder/"validation.json",validation)
    if not (metrics/"source.zip").exists():
        write_snapshot(metrics,raw.name,experiment="V4")
    index_run_files(raw,"V4")
    meta.update(status="completed",verified_models=36,validation=(metrics/"validation.json").relative_to(ROOT).as_posix())
    write_json(raw/"manifest.json",meta)
    write_json(metrics/"manifest.json", dict(**meta, analysis_files={p.relative_to(ROOT).as_posix():digest(p)
               for p in metrics.iterdir() if p.is_file() and p.name!="manifest.json"}))
    print(metrics)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id",required=True)
    evaluate(parser.parse_args().run_id)

