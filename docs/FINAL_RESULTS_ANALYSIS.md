# Final scientific results analysis

## Reporting policy and provenance

All primary report and presentation numbers come from the three completed real Google Colab Tesla T4 runs. This environment policy applies to every method and experiment, before method ranking. Local results are supplementary checks of protocol reproducibility. No environment or seed was selected because it looked better. Every seed result, unfavorable result and large NLL is preserved. The final-delivery analysis reads existing files only; there is no model training, hyperparameter search, checkpoint forward pass or experiment rerun.

The saved-results audit recomputed means and **sample SD (ddof=1)**, checked grids/configs, source hashes, ZIP CRCs, and all 48 checkpoint anchors. The OOD classifier manifest and reference CSV are byte-identical to the real MNIST Colab export. All three executed notebooks have every code cell executed and no recorded error outputs. This verifies the submitted evidence; it is not a new live Colab run.

## Toy Regression: partial reproduction

Twenty fixed points follow y=x³+noise, noise SD=3, with a user-chosen uniform training domain [-4,4]. Models are 1-50-ReLU-1 (MSE) or 1-50-ReLU-2 (Gaussian mean/variance), 3,000 full-batch Adam epochs, lr=.01 for MSE and .001 for Gaussian objectives. Ensembles have five independent members; Gaussian + AT is a single model. The main Deep Ensemble is an ensemble of Gaussian NLL models **without AT**, matching the fourth selected comparison. Gaussian targets are normalized using training mean=12.5197773 and population SD=20.4321270; variances use softplus+1e-6 in normalized units. FGSM epsilon=.01 of observed x range=.06608958. Fixed evaluation: 400 x positions on [-6,6], 20 fresh noisy targets per x, evaluation seed 20261122. Region counts: left 67 x / 1,340 observations; ID 266 x / 5,320 observations; right 67 x / 1,340 observations.

**Band distinction:** the original paper Figure 1 uses ±3 predicted standard deviations, and our original primary four-panel plots also use ±3 SD. The separately added `coverage95`/`width95` use ±1.95996398454 SD against independent **noisy observations**. `clean_curve_coverage_95` from diagnostic step 08 instead asks whether the noiseless x³ curve lies within ±1.96 SD. None of these is a confidence interval over experiment seeds. A Gaussian ±3 SD interval has nominal mass about 99.73%, but the bands are model-based and the ensemble is only moment-matched to a Gaussian.

### Complete regional Colab summaries

NLL below is the moment-matched Gaussian NLL in original y units on noisy observations. MSE uses an evaluation Gaussian proxy based on empirical member variance, with a numerical floor; it was not trained as a probabilistic model. Exact Gaussian-mixture NLL is separate and can be far worse than the moment-matched value. Values are mean ± sample SD across the three fixed-data training runs.

| Method | Region | Clean-curve RMSE | Noisy-target RMSE | Moment NLL | 95% noisy coverage (%) |
|---|---|---:|---:|---:|---:|
| MSE Ensemble | left_ood | 83.133 ± 2.895 | 83.269 ± 2.893 | 355.05 ± 155.85 | 0.00 ± 0.00 |
| MSE Ensemble | in_distribution | 5.421 ± 0.360 | 6.194 ± 0.317 | 7784.98 ± 10950.29 | 13.58 ± 5.52 |
| MSE Ensemble | right_ood | 33.767 ± 1.035 | 33.904 ± 1.032 | 522.02 ± 719.31 | 7.54 ± 9.85 |
| MSE Ensemble | all | 36.994 ± 1.022 | 37.147 ± 1.019 | 5323.92 ± 7420.26 | 10.29 ± 4.78 |
| Single Gaussian NLL | left_ood | 79.187 ± 0.305 | 79.327 ± 0.305 | 7526602.83 ± 70051.76 | 0.00 ± 0.00 |
| Single Gaussian NLL | in_distribution | 5.024 ± 0.073 | 5.849 ± 0.063 | 15966.97 ± 4320.14 | 46.57 ± 0.45 |
| Single Gaussian NLL | right_ood | 39.170 ± 10.714 | 39.296 ± 10.694 | 5.96 ± 2.46 | 69.65 ± 39.55 |
| Single Gaussian NLL | all | 36.525 ± 2.073 | 36.679 ± 2.067 | 1271325.01 ± 14606.03 | 42.64 ± 6.44 |
| Gaussian NLL + AT | left_ood | 80.530 ± 0.472 | 80.668 ± 0.471 | 4030276.63 ± 1599794.23 | 0.00 ± 0.00 |
| Gaussian NLL + AT | in_distribution | 5.019 ± 0.081 | 5.847 ± 0.070 | 171.72 ± 94.17 | 61.60 ± 0.88 |
| Gaussian NLL + AT | right_ood | 35.442 ± 1.039 | 35.574 ± 1.037 | 5.05 ± 0.16 | 67.86 ± 4.42 |
| Gaussian NLL + AT | all | 36.243 ± 0.106 | 36.398 ± 0.105 | 675186.38 ± 268028.11 | 52.33 ± 1.19 |
| Deep Ensemble | left_ood | 79.728 ± 0.454 | 79.867 ± 0.454 | 3961.52 ± 3417.64 | 0.00 ± 0.00 |
| Deep Ensemble | in_distribution | 5.063 ± 0.109 | 5.880 ± 0.095 | 131.06 ± 121.65 | 50.07 ± 2.03 |
| Deep Ensemble | right_ood | 33.814 ± 3.390 | 33.950 ± 3.380 | 4.66 ± 0.31 | 89.33 ± 17.91 |
| Deep Ensemble | all | 35.697 ± 0.695 | 35.854 ± 0.693 | 751.49 ± 653.11 | 48.26 ± 3.58 |

The Deep Ensemble ID RMSE is 5.063 ± 0.109, while the left-OOD RMSE is 79.728 ± 0.454 and its noisy 95% coverage is 0.00 ± 0.00%. On the right, coverage is 89.33 ± 17.91%. This asymmetry directly contradicts a blanket claim of successful, symmetric OOD uncertainty growth. The exact-mixture Deep Ensemble NLL is 5,469.82 ± 5,062.57 in ID and 4,983,343.51 ± 4,307,039.74 in left OOD; moment matching gives 131.06 ± 121.65 and 3,961.52 ± 3,417.64 respectively. These distinct likelihoods must not be interchanged.

### Local versus Colab left OOD

| Method | Local RMSE | Colab RMSE | Local 95% coverage (%) | Colab 95% coverage (%) |
|---|---:|---:|---:|---:|
| MSE Ensemble | 83.156 ± 2.861 | 83.133 ± 2.895 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Single Gaussian NLL | 79.071 ± 0.626 | 79.187 ± 0.305 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Gaussian NLL + AT | 80.578 ± 0.693 | 80.530 ± 0.472 | 0.00 ± 0.00 | 0.00 ± 0.00 |
| Deep Ensemble | 79.717 ± 0.444 | 79.728 ± 0.454 | 0.00 ± 0.00 | 0.00 ± 0.00 |

Every one of the 12 method/seed left-OOD noisy coverage values is zero in **both** environments. Clean-curve diagnostics also retain the failure. In Colab the single Gaussian's left predicted SD is roughly .0205, close to its training-normalized variance floor in original units. Shared ReLU extrapolation bias and variance collapse in unobserved regions are plausible hypotheses, not established causal explanations. No new diagnostic training was done to select a preferred explanation. The main illustrative plot uses the first predeclared seed (42); other seed panels are included without selection.

## MNIST Classification

The network is 784-200-200-200-10, with ReLU and BatchNorm in all three hidden layers (240,610 trainable parameters). MC Dropout adds p=.1 dropout after each nonlinearity. Training uses Adam lr=.1, betas=(.9,.999), eps=1e-8, no weight decay, batch size 100, 10 final epochs. Split seed 4509 gives 55,000 training, 5,000 validation and 10,000 untouched test images. Pixel values are float32/255; no extra normalization or augmentation beyond explicit random/FGSM perturbations. Perturbation is .01 of each training pixel's observed range, clipped to [0,1]. +R and +AT optimize the sum of clean and perturbed cross-entropy. Hyperparameter search was not rerun. The final 10-epoch profile and final epoch evaluation are evidenced; full earlier tuning history is not available in the delivery archives and is not reconstructed by invention.

Forty-eight models = 3 seeds × (5 Ensemble + 5 R + 5 AT + 1 dropout). There are 63 raw rows because each seed has 20 method/M settings plus a redundant single-model M=1 reference, not because 63 models were trained. Below, all M values are included.

| Method | M | Classification error (%) | Accuracy (%) | NLL (nat) | Brier (1/K) |
|---|---:|---:|---:|---:|---:|
| Ensemble | 1 | 2.783 ± 0.051 | 97.217 ± 0.051 | 0.11030 ± 0.00697 | 0.004326 ± 0.000091 |
| Ensemble | 2 | 1.897 ± 0.047 | 98.103 ± 0.047 | 0.07174 ± 0.00516 | 0.003082 ± 0.000073 |
| Ensemble | 3 | 1.687 ± 0.102 | 98.313 ± 0.102 | 0.06134 ± 0.00459 | 0.002744 ± 0.000105 |
| Ensemble | 4 | 1.567 ± 0.038 | 98.433 ± 0.038 | 0.05794 ± 0.00331 | 0.002560 ± 0.000074 |
| Ensemble | 5 | 1.460 ± 0.106 | 98.540 ± 0.106 | 0.05581 ± 0.00223 | 0.002473 ± 0.000060 |
| Ensemble + R | 1 | 2.537 ± 0.421 | 97.463 ± 0.421 | 0.10811 ± 0.01450 | 0.004034 ± 0.000589 |
| Ensemble + R | 2 | 1.923 ± 0.254 | 98.077 ± 0.254 | 0.07095 ± 0.01055 | 0.003082 ± 0.000414 |
| Ensemble + R | 3 | 1.730 ± 0.173 | 98.270 ± 0.173 | 0.06387 ± 0.00526 | 0.002829 ± 0.000271 |
| Ensemble + R | 4 | 1.597 ± 0.166 | 98.403 ± 0.166 | 0.05847 ± 0.00412 | 0.002624 ± 0.000218 |
| Ensemble + R | 5 | 1.533 ± 0.207 | 98.467 ± 0.207 | 0.05665 ± 0.00376 | 0.002505 ± 0.000177 |
| Ensemble + AT | 1 | 2.233 ± 0.343 | 97.767 ± 0.343 | 0.07689 ± 0.01186 | 0.003345 ± 0.000483 |
| Ensemble + AT | 2 | 1.603 ± 0.208 | 98.397 ± 0.208 | 0.05723 ± 0.00700 | 0.002609 ± 0.000268 |
| Ensemble + AT | 3 | 1.507 ± 0.161 | 98.493 ± 0.161 | 0.05258 ± 0.00502 | 0.002406 ± 0.000182 |
| Ensemble + AT | 4 | 1.363 ± 0.095 | 98.637 ± 0.095 | 0.05064 ± 0.00330 | 0.002289 ± 0.000136 |
| Ensemble + AT | 5 | 1.390 ± 0.089 | 98.610 ± 0.089 | 0.04983 ± 0.00393 | 0.002249 ± 0.000147 |
| MC Dropout | 1 | 3.493 ± 0.215 | 96.507 ± 0.215 | 0.13036 ± 0.01515 | 0.005392 ± 0.000480 |
| MC Dropout | 2 | 2.997 ± 0.298 | 97.003 ± 0.298 | 0.11159 ± 0.01219 | 0.004695 ± 0.000458 |
| MC Dropout | 3 | 2.773 ± 0.233 | 97.227 ± 0.233 | 0.10321 ± 0.00898 | 0.004395 ± 0.000408 |
| MC Dropout | 4 | 2.670 ± 0.197 | 97.330 ± 0.197 | 0.10013 ± 0.00760 | 0.004247 ± 0.000379 |
| MC Dropout | 5 | 2.680 ± 0.217 | 97.320 ± 0.217 | 0.09857 ± 0.00695 | 0.004197 ± 0.000370 |

### M=5 local versus Colab

| Method | Local accuracy (%) | Colab accuracy (%) | Local NLL | Colab NLL |
|---|---:|---:|---:|---:|
| Ensemble | 98.57 ± 0.07 | 98.54 ± 0.11 | 0.0544 ± 0.0015 | 0.0558 ± 0.0022 |
| Ensemble + R | 98.51 ± 0.15 | 98.47 ± 0.21 | 0.0558 ± 0.0045 | 0.0567 ± 0.0038 |
| Ensemble + AT | 98.64 ± 0.06 | 98.61 ± 0.09 | 0.0546 ± 0.0076 | 0.0498 ± 0.0039 |
| MC Dropout | 97.26 ± 0.42 | 97.32 ± 0.22 | 0.0995 ± 0.0187 | 0.0986 ± 0.0070 |

M=5 +AT has the lowest mean NLL/Brier in this comparison; ensembles outperform the sampled dropout baseline at the same M. This is descriptive evidence, not a significance test. Mean classification error is not universally monotonic: +AT worsens from 1.363% (M=4) to 1.390% (M=5), and dropout from 2.670% to 2.680%. Five networks within one run are not five seeds. NLL and Brier assess probabilistic quality, but we did not add reliability diagrams/ECE and cannot claim universal calibration from these scores alone.

## MNIST + NotMNIST OOD

All 10,000 MNIST test images and all 18,724 readable NotMNIST small images are included. The archive has 18,726 PNGs; exactly two known files are unreadable and excluded. No OOD training occurs. The 48 checkpoint SHA-256 values and the archived Figure 2(a) manifest/CSV agree with the actual MNIST Colab run. Predictive entropy averages softmax probabilities across M first, then computes -sum p log p using natural logarithms. It is not mean member entropy or entropy of averaged logits. ID=0, OOD=1, and larger entropy is a larger OOD score. Gap = mean OOD entropy - mean ID entropy.

| Method | M | Mean ID entropy (nat) | Mean OOD entropy (nat) | Gap (nat) | AUROC |
|---|---:|---:|---:|---:|---:|
| Ensemble | 1 | 0.0498 ± 0.0098 | 0.1657 ± 0.0439 | 0.1158 ± 0.0342 | 0.6089 ± 0.0224 |
| Ensemble | 3 | 0.0754 ± 0.0047 | 0.8076 ± 0.1073 | 0.7322 ± 0.1048 | 0.9272 ± 0.0284 |
| Ensemble | 5 | 0.0788 ± 0.0030 | 1.0222 ± 0.0320 | 0.9435 ± 0.0309 | 0.9653 ± 0.0038 |
| Ensemble + R | 1 | 0.0421 ± 0.0068 | 0.2024 ± 0.0823 | 0.1603 ± 0.0841 | 0.6856 ± 0.0967 |
| Ensemble + R | 3 | 0.0733 ± 0.0141 | 0.6845 ± 0.0550 | 0.6112 ± 0.0509 | 0.9060 ± 0.0314 |
| Ensemble + R | 5 | 0.0770 ± 0.0097 | 0.8056 ± 0.0404 | 0.7286 ± 0.0363 | 0.9346 ± 0.0118 |
| Ensemble + AT | 1 | 0.0525 ± 0.0066 | 0.5555 ± 0.0949 | 0.5029 ± 0.0911 | 0.8947 ± 0.0382 |
| Ensemble + AT | 3 | 0.0687 ± 0.0069 | 0.9076 ± 0.0153 | 0.8389 ± 0.0084 | 0.9520 ± 0.0064 |
| Ensemble + AT | 5 | 0.0766 ± 0.0073 | 1.0193 ± 0.0209 | 0.9426 ± 0.0153 | 0.9668 ± 0.0023 |
| MC Dropout | 1 | 0.0875 ± 0.0080 | 0.2739 ± 0.0677 | 0.1865 ± 0.0695 | 0.6591 ± 0.0909 |
| MC Dropout | 3 | 0.1092 ± 0.0048 | 0.4802 ± 0.1071 | 0.3710 ± 0.1081 | 0.7491 ± 0.0798 |
| MC Dropout | 5 | 0.1161 ± 0.0053 | 0.5354 ± 0.1175 | 0.4192 ± 0.1189 | 0.7779 ± 0.0736 |

### M=5 local versus Colab

| Method | Local AUROC | Colab AUROC | Local gap (nat) | Colab gap (nat) |
|---|---:|---:|---:|---:|
| Ensemble | 0.9544 ± 0.0106 | 0.9653 ± 0.0038 | 0.8667 ± 0.1023 | 0.9435 ± 0.0309 |
| Ensemble + R | 0.9433 ± 0.0219 | 0.9346 ± 0.0118 | 0.8197 ± 0.0716 | 0.7286 ± 0.0363 |
| Ensemble + AT | 0.9653 ± 0.0060 | 0.9668 ± 0.0023 | 0.9483 ± 0.0050 | 0.9426 ± 0.0153 |
| MC Dropout | 0.7860 ± 0.1115 | 0.7779 ± 0.0736 | 0.4312 ± 0.1530 | 0.4192 ± 0.1189 |

The negative local Ensemble M=1, seed=123 AUROC **0.379028** remains in the raw table. Low entropy on some OOD examples shows residual overconfidence; a high mean AUROC does not guarantee every OOD image receives high uncertainty. At M=5 the Colab +AT/Ensemble AUROC difference is small relative to some seed spreads, so we do not claim significant superiority. The mean entropy increase from M=1 to 5 is larger for each ensemble variant than for dropout in this selected dataset/setting; this supports the paper's qualitative direction without universal claims.

### Actual ordering correction evidence

`dataset_order_alignment.json` in the real completed Colab OOD ZIP records 16,719 reordered positions among 18,724 samples. Linux-native tensor SHA-256 was `a54ca4033cde31b59c3cec969e2646eec28c1bbcb91e7c7259cbd4ffac3b0fe6`; after joint image/label/ID permutation it is `5da5a545dee93c711d8ec8ca5be2398f1114a406690a1efaa74d5d072bf1a46e`, identical to the local reference. Pixel values are unchanged. Both prepare and full-inference loaders use the adapter, and tensor plus sample-order hashes remain mandatory. Four explicit cross-platform regression tests and 274 independent saved-output checks are recorded in the real Colab evidence. This is an ordering correction, not a new preprocessing or data version.

The original Figure 3(a) is a predictive-entropy histogram experiment. AUROC and entropy gap are our additional analyses; the paper does not supply directly comparable numeric AUROC/gap targets. Labels A-J are metadata only and are outside the digit classifier's label space, so digit accuracy on NotMNIST is not reported. Histograms use 50 common bins on [0,ln(10)], per-seed density normalization, then the mean of three densities, with shared row-wise y scales.

## Compute, reproducibility and limits

Recorded T4 workload durations: Toy 333.209 s including all notebook steps; MNIST invocation 1228.190 s with 1188.858 s summed training epochs; OOD groups 15.912 s, smoke 3.133 s, inference invocation 52.424 s. These are different timing scopes; no fair runtime benchmark or speedup is claimed. Local hardware/software: RTX 4070 Ti SUPER, PyTorch 2.11.0+cu128, CUDA 12.8. T4: PyTorch 2.14.1+cu130, CUDA 13.0. Package versions, floating-point kernels and stochastic optimization can affect outcomes; these are plausible explanations, not isolated causal findings.

Scope is limited to three selected figures, M≤5, one OOD dataset, one fixed small toy training set and three training seeds. No significance tests, confidence intervals over seed means, universal calibration claims or full-paper replication are made. Differences include toy data draw/domain and normalization/full-batch optimization, ten MNIST epochs and train-only validation holdout, smaller ensemble grids, extra AUROC/gap and coverage diagnostics, and the explicit ordering adapter.

**Conclusion:** classification and OOD results support the selected qualitative ensemble-versus-dropout trend in this protocol. Toy uncertainty reproduction is partial and fails badly in left extrapolation. All failures, numeric tables and environment differences remain visible.
