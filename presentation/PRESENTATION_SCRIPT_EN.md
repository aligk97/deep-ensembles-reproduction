# Presentation rehearsal script (English)

Proposed speaker allocation, not a claim about actual project contributions. Total speaking budget 7:30, plus 2:30 questions. The course forbids reading slides or notes: rehearse, then explain in your own words. Replace names/IDs and practice aloud.

## 1. Deep Ensembles Reproduction

Student 1 — 15 seconds

We reproduce selected experiments from the 2017 Deep Ensembles paper for CMP4509. Our three experiments cover regression, digit classification and unfamiliar letter images. We will distinguish the findings that held from the uncertainty claim that failed.

## 2. Predictive uncertainty

Student 1 — 45 seconds

Our starting problem is that a neural network must produce an answer even when an input is unfamiliar. A high digit-classification accuracy does not tell us whether the probability distribution is trustworthy on a letter. The paper proposes a simple solution: train several independent networks to predict distributions, then combine their probabilities. We focus on three questions. Does learning variance improve the toy regression uncertainty? Do ensembles improve classification scores compared with MC Dropout? And does uncertainty increase appropriately on NotMNIST? We tested those questions under a small free-GPU budget, rather than attempting every experiment in the paper.

## 3. Deep Ensembles method

Student 1 — 90 seconds

Each model predicts a distribution and learns with a proper scoring rule. For log loss, the expected loss is the true entropy plus a nonnegative KL divergence, so the true distribution is optimal in the population. In regression, Gaussian NLL contains a log-variance penalty and a squared residual divided by variance. This balances sharpness against errors. The network uses softplus to keep variance positive. An ensemble is an equally weighted mixture. Its mean is the average member mean. Expanding the second moment gives total variance as average predicted variance plus variance of the member means. These represent observation ambiguity and model disagreement, with important model assumptions. For classification, we average the softmax probabilities. We then calculate predictive entropy, not mean member entropy. Two certain models disagreeing on two classes each have zero entropy, but their mixture has entropy log two. Optional FGSM training adds a small perturbation in the gradient-sign direction and trains on the sum of clean and perturbed losses. Dropout instead samples masks from one model, while BatchNorm stays fixed.

## 4. Reproduction setup

Student 2 — 45 seconds

We kept the final configurations already used in the completed experiments. The three run seeds are forty-two, one hundred twenty-three and twenty twenty-six. Toy regression uses the same twenty observations in every run and three thousand full-batch epochs. MNIST uses a three-hidden-layer MLP with two hundred units per layer, a training-only validation split and ten epochs. The forty-eight models include five members for each ensemble variant and one dropout model for each seed. OOD reuses those exact models. We report all three actual T4 runs as primary results, with the local runs as supplementary comparisons. Standard deviations use the sample definition.

## 5. Toy Regression: partial reproduction

Student 2 — 60 seconds

The toy result is a partial reproduction. This plot shows the Deep Ensemble for the first predeclared seed, rather than the best-looking run. On the left, the predicted mean misses the cubic curve badly, while its uncertainty band stays too narrow. All four methods have zero left-OOD ninety-five-percent noisy-target coverage in every seed. This failure also appears locally. The plotted band is three standard deviations, as in the original figure. The numerical coverage is a separate one-point-nine-six-standard-deviation interval measured against fresh noisy observations. We must not confuse the two. Shared extrapolation bias and variance collapse are plausible explanations, but we did not run a controlled causal ablation. Learning variance and adding an ensemble therefore do not guarantee good uncertainty outside the data.

## 6. MNIST Classification

Student 2 — 45 seconds

For MNIST, the main qualitative trend is more favorable. At five members, Ensemble plus adversarial training reaches 98.61 percent accuracy, compared with 97.32 percent for five dropout passes. Its mean NLL and normalized Brier score are also lower in this protocol. The Brier definition includes the paper's one-over-ten normalization. Each point averages three training runs, and error bars are sample standard deviations. We do not claim that every metric improves at every increase in ensemble size. For example, adversarial classification error is slightly higher at five than at four members. Three seeds also do not establish statistical significance.

## 7. MNIST + NotMNIST predictive entropy

Student 3 — 70 seconds

For OOD evaluation, the top row contains the ten thousand known MNIST digits. The bottom row contains all eighteen thousand seven hundred twenty-four readable NotMNIST letters. The two known corrupted images were excluded. There is no new model training here. We reuse the same forty-eight models from the preceding Colab classification run and verify their hashes. In the known classes, predictive entropy stays low. On unfamiliar letters, the ensemble entropy grows more strongly as we increase the number of members than it does for dropout. Nevertheless, some OOD examples retain low entropy, so this is not a guarantee that every unfamiliar input is detected. Letter labels are outside the digit classifier's label space, which is why we evaluate uncertainty and OOD ranking rather than digit accuracy on these images.

## 8. OOD separation and reproducibility

Student 3 — 50 seconds

Our added AUROC analysis summarizes the entropy ranking. At five members, adversarial ensembles reach about 0.9668, plain ensembles 0.9653, and dropout 0.7779. The table includes every method and its sample spread, together with the entropy gap. AUROC and gap are additions to the paper's histogram experiment. Our provenance checks also found and fixed an operating-system ordering issue: the image bytes were the same, but Linux and Windows sorted filenames differently. Jointly reordering images and labels restored the reference hashes. The limitations remain three seeds, one OOD dataset, a small ensemble grid and the unresolved toy failure.

## 9. Conclusions and questions

Student 3 — 30 seconds

Our conclusion is limited but clear. Ensembles support the selected classification and unknown-class trend under our protocol. Toy regression does not reproduce reliable uncertainty in left extrapolation. We preserve both findings and every seed instead of selecting favorable results. The repository contains the sources, actual execution evidence, derivations and full tables. We are ready to discuss the assumptions, the variance derivation and the limitations.
