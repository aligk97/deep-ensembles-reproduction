# Repository audit

## Scope and preservation

Read-only audit of 87,720 original files, the supplied 15-page paper, 3-page course announcement and complete original LaTeX template. No training, hyperparameter search, model forward pass or existing experiment rerun occurred. All changes are in the independent submission/build folders and new delivery documents.

## Final sources and evidence

| Experiment | Final source | Local evidence | Real Colab evidence |
|---|---|---|---|
| Toy | `toy_regression/01_...08_*.py`, Colab runtime and fixed 20-row CSV | `outputs/07_seed_experiments/aggregate`, `08_uncertainty_diagnostics` | `toy_regression_colab_run_br453546.zip`, completed T4 manifest, 49 model files, 48 rows / 16 summaries |
| MNIST | data/model/engine/metrics/plots/run_experiment + `configs/local.json` | completed `outputs/local`, 48 checkpoints, 63 rows | `mnist_classification_colab_figure2a_final_local_48.zip`, T4, 48 hashed models, 63 rows / 21 summaries |
| OOD | existing data/metrics/inference/plots/run_experiment + Colab adapters + `configs/final.json` | completed `outputs`, 36 rows / 12 summaries | `mnist_notmnist_ood_colab_figure3a_from_figure2a_final_local_48.zip`, T4, same MNIST checkpoint hashes, 36 rows / 12 summaries |

The three executed notebooks have all code cells executed and zero error outputs. ZIP CRCs, scientific source hashes, 48 MNIST audit anchors, OOD checkpoint identity, OOD artifact hashes and saved mean/sample-SD values passed the machine-readable audit. This audit reads saved evidence and does not rerun model inference.

## Legacy and duplicate material

Original root README and historical preparation READMEs are stale: they predate actual Colab runs. `smoke/`, `notebook_workspace/`, local execution audits, `.venv`, `.deps`, `__pycache__`, raw data and intermediate archives are workspace artifacts, not final public sources. The root `mnist_notmnist_ood.zip` is a redundant older workspace export. All are preserved in place and excluded from the clean repository.

Detected 14 groups of identical files larger than 100 kB outside dependency/audit trees. Large raw archives/checkpoints remain in their original locations. Full duplicate lists and size inventories are in the private build audit.

## Git and publication

Branch: `main`. Existing remote: https://github.com/aligk97/deep-ensembles-reproduction. The GitHub connector confirmed **public** visibility and default branch `main`; no repository creation, push or visibility change was performed. Original working tree has pre-existing tracked modifications and many untracked Colab/OOD files. No original git index or history was modified.

Tracked files: 561. Publication must use the clean allowlist, not an indiscriminate `git add .` in the original workspace. Files exceeding GitHub's ordinary per-file size budget include large archive exports; they are not included in the clean repository.

## Privacy and credentials

Recognized secret-pattern findings: 0. Personal absolute Windows paths occur in 33 source evidence/documents. The public delivery replaces personal paths in metadata copies only and logs every redaction with original/published hashes. Numeric CSV values, figures, scientific Python and checkpoints are not altered. Original evidence remains untouched. Base64 notebook bundles and hashes are not classified as credentials merely because they have high entropy.

## Course requirements extracted from supplied announcement and template

| Requirement | Audit finding / final action |
|---|---|
| Exactly three students; complete Teams sign-up row | Three editable name/ID/contribution placeholders; actual registration/contact email must be confirmed by students. |
| Chosen paper and central-claim reproduction | Deep Ensembles, NeurIPS 2017, listed as paper 14; scope is Figures 1, 2(a), 3(a), not the full paper. |
| PyTorch implementation and mathematical derivations | Sources and derivations are included. Every member must understand all derivations. |
| Several seeds; template specifies at least three | 42, 123, 2026; mean ± sample SD, ddof=1. Ensemble members are not independent experimental repetitions. |
| Own public repository and executable README | Existing public remote verified; final sanitized content still requires user-approved publication. |
| Top-to-bottom notebooks on free GPU platform | Three actual T4 executed notebooks and completed export manifests available. Order: Toy, MNIST, OOD. |
| Free GPU budget; roughly one hour or less per single run; checkpointing | Recorded T4 runs are below one hour; MNIST saves each completed epoch/member and OOD persists group caches. |
| Course LaTeX template (mandatory; otherwise zero) | Original article/11pt/A4/2.5cm preamble and required sections preserved in the separate compiled report. |
| PDF report (mandatory; otherwise zero) | Final 10-page report actually compiled; all pages rendered/reviewed; source ZIP verified. |
| Repository link in report | https://github.com/aligk97/deep-ensembles-reproduction |
| Report date | 13 December 2026, 23:59, per supplied announcement. |
| Presentation dates | 30 November or 7 December 2026; assigned session must be confirmed. |
| Presentation duration and scope | Nine slides; planned 7.5 minutes speaking plus 2.5 minutes questions; focus on own reproduction. |
| No reading slides or notes | Scripts and defense guide explicitly require rehearsal without reading during the talk. |
| Individual questions; everyone understands whole project | Turkish study guide, equations and 39 Q&A provided. |
| Page limit / grading breakdown | Announcement explicitly says these will be announced later. No numerical page cap is supplied; students must check later course updates. |
| Report sections | Abstract, Introduction, Background and Method, Experimental Setup, Results, Discussion, Conclusion, Contributions, Use of AI Tools, Code, References. |
| Hyperparameter provenance and deviations | Use saved final profiles; no new tuning. Earlier tuning/search history is not fully evidenced by the final archives, so do not invent a validation-search narrative. |

## Failure preservation and analysis policy

All three actual Colab environments are selected as the primary reporting environment in advance. All local results remain available as reproducibility comparisons. Left OOD toy coverage is zero for every method/seed in both environments. Local Ensemble M=1, seed=123 AUROC below 0.5 is retained. No rows are removed to improve conclusions.

## Supplied template and reference reading

The entire provided paper, including supplementary pages and bibliography, course announcement, original `main.tex`, and existing `references.bib` were read. The supplied Figure 1 uses ±3 standard deviations; our original primary toy panels also use ±3 SD. Separate 95% coverage metrics use ±1.95996398454 SD on fresh noisy observations; diagnostic clean-curve coverage is a different statistic.
