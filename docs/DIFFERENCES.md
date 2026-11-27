# Differences from the original paper

| Aspect | Original paper / evidence | This reproduction | Implication |
|---|---|---|---|
| Scope | Main paper plus multiple regression, SVHN and ImageNet studies | Figures 1, 2(a), 3(a) only | No full-paper replication claim |
| Toy data | 20 noisy cubic observations; exact supplied points not available | Fixed own 20-row CSV, seed 42, x uniform on [-4,4] | Exact original curve/numbers not expected |
| Toy architecture | Architecture attributed to prior PBP work | Explicit 1-50-ReLU-1 / 1-50-ReLU-2 | Small ReLU extrapolation may have shared bias |
| Toy optimization | General paper default Adam .1, batch 100 unless specified; exact toy trajectory not fully specified | Adam .01 MSE / .001 Gaussian, 3,000 full-batch epochs, target normalization | Documented stabilization choices; no newly optimized settings in final delivery |
| Main toy band | Figure 1 ±3 predicted SD | Main panels also ±3 SD | Faithful band convention |
| Added toy diagnostics | No corresponding region-wise numeric table in Figure 1 | ±1.96 SD noisy-target coverage; separate clean-curve diagnostics; exact vs moment NLL | Additional analyses, not original benchmark targets |
| MNIST network | Three hidden layers of 200 ReLU + BN; dropout .1 | Same architecture family, 240,610 parameters | Architectural match within selected experiment |
| MNIST budget/split | Standard training set, paper plots larger M; no exact matching final checkpoint supplied | 55k train / 5k validation, split seed 4509, 10 epochs; M=1..5 | Resource-scaled protocol, not exact numerical reproduction |
| Perturbations | FGSM .01 input range; sum clean + perturbed losses | Same rule, train-only per-pixel ranges and [0,1] clipping; random-sign comparator | Explicit practical bounded-image implementation |
| OOD grid | Figure 3(a) displays M=1,5,10 | M=1,3,5 | Smaller ensemble budget |
| OOD measurements | Entropy distributions | Entropy distributions plus entropy gap and AUROC | Extra metrics must not be attributed to the paper |
| Histogram display | Paper figure conventions | 50 common bins, density averaged across three seeds, common within-row scales | Explicit, reproducible normalization |
| Data ordering | OS-specific ordering not part of the paper's scientific claim | Adapter pins completed local order and verifies hashes | Prevents spurious mismatch and changed MC mask/image assignment |
| Randomization | Multiple independent models from initialization/shuffling | Three named run seeds; fixed data and documented member seeds | Sample SD reflects training runs, not dataset variability |
| Numerical environment | Original Torch ecosystem | Recorded PyTorch/Colab/Windows versions | Bitwise training equivalence not promised |
| Hypotheses | Qualitative uncertainty improvement | Shared extrapolation bias and floor collapse discussed as hypotheses | No unperformed causal experiment is implied |

The supplied paper does not provide machine-readable numeric targets for the exact selected figure settings. The report compares claimed trends and settings, never invents original figure numbers by reading a plot as exact data. All final settings were already used in completed runs; no new hyperparameter search was performed. Full earlier tuning provenance is not fully represented by final archives and must not be invented.
