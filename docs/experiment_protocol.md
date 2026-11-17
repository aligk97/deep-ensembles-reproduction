# Experiment protocol (draft)

## Reproducibility
1. Run each configuration at least **3 times with independent experiment seeds** (initial suggestion: 42, 123, 2026).
2. Save model configurations, data splits, batch size, optimizer, learning rate, epochs, versions, seed values, hardware and training times.
3. Ensemble members must be independently initialized and trained with randomized data order on the same designated training set, unless an explicit deviation is documented.
4. A five-member ensemble is **one full experimental replicate**; repeat the entire ensemble for each outer seed.
5. Report **mean ± sample standard deviation** across full experimental replicates; preserve raw per-seed metrics.
6. Use validation data for hyperparameter selection. Do not tune on the test set.

## 01 — Toy Regression: paper Figure 1

The paper's Section 3.2 describes **20 training observations** with target `y = x^3 + ε`, where `ε ~ N(0, 3²)`. Its Figure 1 compares:
- Ensemble of five networks trained on MSE, with empirical prediction variance;
- Single Gaussian-output neural network trained on NLL;
- Single Gaussian-output neural network trained on NLL plus adversarial training;
- Ensemble of five Gaussian-output NLL networks.

Note that the paper refers to an earlier work for architecture; any unreported input range, network or training setting we choose must be identified as **our implementation choice**, not attributed to the paper.

## 02 — MNIST: paper Figure 2(a)

Section 3.4 uses an MLP with three hidden layers of 200 units each, ReLU nonlinearities, and batch normalization. MC Dropout applies probability 0.1 after nonlinearities. Compare **classification error, NLL and K-normalized Brier score** for selected ensemble sizes, for example M = 1, 3 and 5. Report model/epoch/computational deviations.

## 03 — OOD (optional): paper Figure 3(a)

Train the classifier on MNIST, evaluate predictive entropy on both MNIST and NotMNIST test images. Do not train the classifier on OOD labels.

## Before reporting

- Notebooks run from top to bottom in a free Colab/Kaggle environment.
- Raw per-run results, actual figures and commands to regenerate them are committed.
- No fabricated numbers or hand-drawn claimed results.
- Any differences from the paper are explicitly listed.
