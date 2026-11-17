# CMP4509 — Deep Ensembles Reproduction

A PyTorch reproduction of **Lakshminarayanan, Pritzel & Blundell (2017)**, [*Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*](https://arxiv.org/abs/1612.01474) (NeurIPS 2017).

> **Status:** Initial repository setup. No experiments or results have been completed yet.

## Scope and status

| Phase | Paper figure | Planned work | Status |
| --- | --- | --- | --- |
| 01 — Toy Regression | Figure 1 | 20-point synthetic cubic regression; MSE ensemble; Gaussian NLL single model; NLL + adversarial training; NLL ensemble | Planned |
| 02 — MNIST Classification | Figure 2(a) | MLP baseline; Deep Ensemble vs MC Dropout; classification error, NLL, Brier score | Planned |
| 03 — OOD | Figure 3(a) | MNIST vs NotMNIST predictive-entropy histograms | Optional |
| 04 — Analysis | Figures 1 and 2(a) | 3 independent repetitions, mean ± sample standard deviation, paper comparison | Planned |
| 05 — Submission | Course requirements | Locally developed PyTorch code that also runs on free Colab/Kaggle, LaTeX report PDF, presentation | Planned |

The objective is to test the paper's **central uncertainty-estimation claims**, not to reproduce all numbers exactly. Deviations from the original paper will be explicitly documented.

## Running the experiments

**Primary development machines:** Ali — NVIDIA RTX 4070 Ti SUPER; Atahan — NVIDIA RTX 30-series (exact model and VRAM to be confirmed). We plan to develop and train locally for convenience, using CUDA-enabled PyTorch. Toy Regression may be just as fast on CPU because the dataset is tiny.

**Important course constraint:** The course announcement specifies work must run on free GPU platforms (Google Colab free tier or Kaggle Notebooks), and deliverable notebooks must execute top-to-bottom on such a platform. **Local GPU training alone does not prove compliance.** We will also verify the complete reproduction on a free platform and clarify with the instructor whether locally executed runs may be the reported final runs.

1. Clone the repository locally and use an isolated Python environment.
2. Install CUDA-capable PyTorch using the [official installation selector](https://pytorch.org/get-started/locally/) for each machine's OS/driver; then install remaining project requirements. Avoid assuming that `pip install torch` necessarily selects the desired CUDA build.
3. Verify `torch.cuda.is_available()` and record PyTorch, CUDA, GPU and driver versions for each run.
4. Run the Python modules and notebooks locally. Save seeds, metrics and figures under `results/`.
5. Before submission, open the same notebook(s) on **free Colab or Kaggle** and verify they run **from top to bottom**, with no absolute local paths or machine-specific dependencies. Save evidence of this check.

Exact per-machine setup and command details should be added after OS/driver checks.

**Current limitation:** Experiment notebooks have not yet been implemented. The repository currently contains the project structure and research protocol only; there are no measured results.

## Reproducibility requirements

- Use **at least 3 independent experiment seeds** for each configuration and report mean ± standard deviation.
- Five independently trained members of one ensemble count as **one ensemble experiment**, not five repetitions.
- Save hyperparameters, seeds, preprocessing, model architecture, code/environment versions, hardware, metrics, and training settings.
- Do not select hyperparameters using test data.
- Train ensemble members independently (random initializations and shuffled training data), using the full designated training split unless a documented deviation is required.
- Compare to the source paper and document all experimental simplifications.

See [experiment protocol](docs/experiment_protocol.md) and [roadmap](docs/roadmap.md).

## Repository layout

| Path | Purpose |
| --- | --- |
| `notebooks/` | Local Jupyter notebooks that also run on Colab/Kaggle (to be added) |
| `src/toy_regression/` | Reusable PyTorch toy-regression modules (to be added) |
| `src/mnist/` | Reusable PyTorch MNIST modules (to be added) |
| `src/evaluation/` | Metrics and plots (to be added) |
| `results/figures/` | Generated figures |
| `results/metrics/` | Reproducible per-run results |
| `docs/` | Roadmap, experiment protocol and contributions |
| `report/` | **Course-provided LaTeX template**, bibliography and instructions |

## Collaboration

The official group has 3 members. Main model and experiment implementation are primarily handled by 2 members; the third member contributes to visualizations, documentation, and presentation work. Actual contributions and pull requests are recorded in [contributions](docs/contributions.md).

**Working branches:** `ali` and `atahan`. Each developer commits code to their own branch; use GitHub pull requests to review and merge completed work into `main`. Keep `main` stable. Before starting new work, update personal branches with changes from `main`. Third member contributions may go through a reviewed pull request or be co-authored with the documentation owner.

## Deliverables and deadlines

- **Public GitHub repository** with code and README, including runnable notebooks on a free GPU platform.
- **Course-template LaTeX report**, submitted as a **PDF** with the GitHub link.
- **Presentation:** 10 minutes including questions; approximately 8–10 slides; each student should understand the whole method and derivations.
- **Dates:** presentations on **30 Nov or 7 Dec 2026**, report due **13 Dec 2026, 23:59**.

## Citation

Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles*. NeurIPS 30. [arXiv:1612.01474](https://arxiv.org/abs/1612.01474).
