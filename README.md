# Deep Ensembles Reproduction

CMP4509 Introduction to Deep Learning, Fall 2026. A PyTorch reproduction of selected central uncertainty-estimation experiments from Lakshminarayanan, Pritzel and Blundell (NeurIPS 2017), [Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles](https://proceedings.neurips.cc/paper_files/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html).

Three local runs and three actual Google Colab Tesla T4 experiment deliveries are complete. This repository packages their saved results; final-delivery preparation did **not** retrain models or rerun experiments. The primary report uses **Colab for every experiment**. Local runs provide supplementary reproducibility evidence, with no per-method environment selection.

## Objectives and findings

1. Figure 1: learned Gaussian variance and five-member ensembles on 20 noisy cubic observations. **Partial reproduction**: all methods have zero left-OOD 95% noisy-observation coverage in every seed, locally and on Colab. Learned variance does not guarantee reliable extrapolation.
2. Figure 2(a): Ensemble, Ensemble + R, Ensemble + AT and MC Dropout, M=1,2,3,4,5. Ensembles perform better than the sampled dropout baseline in this setting. Improvement is not universally monotonic in M and no significance tests were performed.
3. Figure 3(a): MNIST versus NotMNIST small entropy histograms, M=1,3,5, reusing the **same 48 Colab MNIST checkpoints** with no new OOD training. AUROC and entropy gap are additional analyses, not original Figure 3(a) metrics.

All values below are mean ± sample SD over seeds 42,123,2026 (ddof=1).

| Method (M=5) | MNIST accuracy (%) | NLL | Brier (1/K) | OOD AUROC |
|---|---:|---:|---:|---:|
| Ensemble | 98.54 ± 0.11 | 0.0558 ± 0.0022 | 0.00247 ± 0.00006 | 0.9653 ± 0.0038 |
| Ensemble + R | 98.47 ± 0.21 | 0.0567 ± 0.0038 | 0.00250 ± 0.00018 | 0.9346 ± 0.0118 |
| Ensemble + AT | 98.61 ± 0.09 | 0.0498 ± 0.0039 | 0.00225 ± 0.00015 | 0.9668 ± 0.0023 |
| MC Dropout | 97.32 ± 0.22 | 0.0986 ± 0.0070 | 0.00420 ± 0.00037 | 0.7779 ± 0.0736 |

The negative local Ensemble M=1, seed=123 AUROC=0.3790 is preserved. Full tables, including all M values, high toy NLLs and failures, are under [results](results/) and [final analysis](docs/FINAL_RESULTS_ANALYSIS.md).

## Repository structure

```text
toy_regression/        original experiment sources and fixed 20-point CSV
mnist_classification/  original MNIST sources, final config, Colab support
mnist_notmnist_ood/    original OOD sources, final config, ordering adapter
notebooks/             three self-contained runnable Colab notebooks
notebooks/executed/    actual executed T4 notebook evidence
results/local/         saved local tables, metadata and figures
results/colab/         saved Colab tables, metadata and figures
report/                instructor-template LaTeX, figures and compiled PDF
presentation/          nine-slide PPTX/PDF and rehearsal scripts
docs/                  audit, deviations, study and defense material
scripts/               delivery-only numeric/link/source checks
```

## Installation and local execution

Use Python 3.10+ and an appropriate CUDA-enabled PyTorch installation. Install the remaining dependencies with `python -m pip install -r requirements.txt`. On Colab, keep its preinstalled CUDA PyTorch. Recorded T4 software: Python 3.13.16, PyTorch 2.14.1+cu130, CUDA 13.0, NumPy 2.4.6. Exact available metadata stays in each saved manifest.

The original source code is preserved. Running the commands below is a **new reproduction**, not a required part of viewing the completed results. New outputs go to the experiment `outputs/` directories; published tables remain separately under `results/`.

```bash
# The fixed training CSV is already included. Do not regenerate it for the recorded protocol.
python toy_regression/07_multi_seed_experiments.py
python toy_regression/08_diagnose_uncertainty.py
python mnist_classification/run_experiment.py --config mnist_classification/configs/local.json --device cuda
python mnist_notmnist_ood/prepare_data.py
python mnist_notmnist_ood/verify_existing.py --record
python mnist_notmnist_ood/run_experiment.py --device cuda
```

The local OOD entry point expects completed MNIST checkpoints at `mnist_classification/outputs/local`, the final profile name `local`, and a fresh `verify_existing.py --record` baseline. Never copy a Windows virtual environment. If the integrity baseline already exists, it is intentionally not overwritten. The fully audited cross-platform execution path is the Colab notebook chain below. The legacy direct local loader uses host-native filename ordering; the Colab adapter pins the validated local reference order and records it.

## Google Colab: execution order and dependencies

Open and Run all, in this order, on a free GPU runtime:

1. [Toy Regression](notebooks/Toy_Regression_Colab.ipynb). Self-contained, fixed 20-point CSV, no Drive mount required. Saves a uniquely named complete run ZIP.
2. [MNIST Classification](notebooks/MNIST_Deep_Ensembles_Colab.ipynb). Set `RUN_ID` (default `figure2a_final_local_48`), authorize Drive, train/save 48 models, export the complete MNIST ZIP. No hyperparameter search is required.
3. [MNIST + NotMNIST OOD](notebooks/MNIST_NotMNIST_OOD_Colab.ipynb). Set `MNIST_RUN_ID` to the **same** classifier run and choose an independent `OOD_RUN_ID`. OOD performs inference only.

Default checkpoint source:

```text
/content/drive/MyDrive/DeepEnsembles_MNIST_Figure2a/<MNIST_RUN_ID>/
  colab_session.json
  mnist_classification/outputs/local/
    manifest.json, config.json, colab_resume_audit.json
    raw_results.csv, summary.csv, checkpoints/...
```

OOD writes to `MyDrive/DeepEnsembles_MNIST_NotMNIST_OOD/<OOD_RUN_ID>/`. Use `SOURCE_MODE='drive'`, or explicitly choose `'zip'` and give `ZIP_PATH` for the complete real MNIST Colab results ZIP. Empty ZIP_PATH uses the previous MNIST notebook's default Drive ZIP location. Local Windows models, missing hashes or incomplete exports fail; there is no fallback or automatic training. Preserve the original complete MNIST export outside GitHub. Probability/entropy caches are omitted here to avoid bulk duplication, but the OOD notebook creates and exports them.

Completed probability groups are reusable only when checkpoint/config/data/runtime signatures match. Start a fresh Colab session when changing the notebook version or source. Mount/download interaction may require browser permission. The saved executed notebooks and manifests are evidence of completed T4 runs, not a guarantee of identical future hardware allocation or bitwise training results.

## Datasets and preprocessing

MNIST downloads automatically, verifies official file checksums, uses 55,000 training / 5,000 validation images (split seed 4509) and the untouched 10,000-image test set. Inputs are float32 uint8/255 with 784 features; no mean/std normalization or augmentation beyond the explicit training perturbations.

NotMNIST small/test uses all 18,724 readable grayscale 28×28 images; two known corrupt files are excluded. Original URL is attempted before a pinned mirror of the **same** archive. SHA-256: `76b2ef7ccc56452964780569db37b279c9cbe1bf442c3b38be784752683b3cec`. No replacement dataset or subsampling. Native light-on-dark polarity is preserved. The Windows/Linux `sorted(Path)` difference moved 16,719 positions; `colab_data.py` jointly reorders images/labels/IDs to the validated reference and checks both tensor and order SHA-256, without changing pixel values.

## Reproducibility and runtime

Three run seeds are 42,123,2026. Five members of an ensemble count as one configuration run. M=1,3,5 OOD uses prefixes of the same M=5 probability stack. Predictive entropy is calculated **after probability averaging**, with natural logarithms. MC Dropout activates only dropout; BatchNorm statistics stay frozen. Mean ± sample SD uses ddof=1.

Recorded T4 durations: Toy complete notebook workload 333.2 s; MNIST invocation 1228.2 s; OOD inference groups 15.91 s, complete inference invocation 52.42 s. These scopes differ and are not throughput benchmarks. Local hardware was RTX 4070 Ti SUPER with PyTorch 2.11.0+cu128. Cross-version/GPU results need not match bitwise.

## Differences and limitations

See [DIFFERENCES.md](docs/DIFFERENCES.md). We reproduce selected figures, not the full paper, ImageNet, SVHN or regression benchmarks. MNIST training is limited to 10 epochs with a 5,000-image validation holdout. We use M≤5 rather than the larger M values in the original plots. Toy x range, full-batch training, learning rates, target normalization and added evaluation protocol are documented. Original and our main toy panels use ±3 SD; separate 95% coverage uses ±1.96 SD on fresh noisy observations. Raw uncertainty scores do not prove universal calibration or reliable OOD detection. Three seeds do not support significance claims.

## Authors, contributions and AI disclosure

| Author | Student ID | Verified contribution |
|---|---|---|
| [Student 1 full name] | [Student 1 ID] | [Complete from actual work/commits] |
| [Student 2 full name] | [Student 2 ID] | [Complete from actual work/commits] |
| [Student 3 full name] | [Student 3 ID] | [Complete from actual work/commits] |

Names, IDs and individual contributions are deliberately not inferred from a computer username or a suggested division of work. ChatGPT and Codex assisted implementation/debugging, test and notebook preparation, analysis, documentation, derivations, report and presentation drafting/validation. Students must independently verify, understand and approve the final work. Numerical findings come from saved experiment outputs, not invented AI values.

## Report, presentation and checks

- [Compiled course-template report](report/report.pdf), [LaTeX source](report/main.tex), [Overleaf ZIP](report/overleaf_submission.zip)
- [Presentation PPTX](presentation/Deep_Ensembles_Reproduction.pptx), [PDF](presentation/Deep_Ensembles_Reproduction.pdf)
- [English script](presentation/PRESENTATION_SCRIPT_EN.md), [Turkish script](presentation/PRESENTATION_SCRIPT_TR.md)
- [Defense Q&A](docs/DEFENSE_QUESTIONS_AND_ANSWERS.md), [study guide](docs/PROJECT_STUDY_GUIDE_TR.md), [equations](docs/KEY_EQUATIONS.md)
- [Audit](docs/REPOSITORY_AUDIT.md), [validation report](docs/FINAL_VALIDATION_REPORT.md), [publication instructions](docs/PUBLISHING.md)

Run `python -B scripts/validate_delivery.py` for saved-result aggregation, numeric provenance, links, notebook structure and source integrity checks. It never trains models or runs checkpoint inference. The report uses the existing public repository URL https://github.com/aligk97/deep-ensembles-reproduction; final delivery content still requires reviewed publication by the students.
