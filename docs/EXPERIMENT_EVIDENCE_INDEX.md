# Experiment evidence index

Primary: all three real Colab Tesla T4 exports. Supplementary: all completed local outputs.

Toy final replicated comparison is under results/*/toy_regression/07_seed_experiments/aggregate/metrics/. Root-level toy files are earlier single-seed demonstration panels and training outputs, not independent three-seed summaries. Step 08 is a separately labeled diagnostic of the same saved predictions. No unfavorable row is removed.

MNIST raw_results.csv has 63 rows including the single/Ensemble-M1 duplicate reference; summary.csv has 21 rows. The final profile name local is a profile identifier, not a claim that the Colab run used Windows weights.

OOD raw results have 36 rows and 12 summaries. The actual Colab classifier_reference manifest/CSV and checkpoint audit tie OOD to the exact Figure 2(a) run. Saved per-image probabilities/entropy and model weights remain in the original complete exports outside the clean repository. The notebook generates those caches during a new user-authorized reproduction.

Privacy sanitization changes only personal-path strings in metadata copies; hashes of original and published metadata are logged in PUBLICATION_REDACTIONS.json. Numeric CSVs and figures are copied byte-for-byte and pinned by docs/validation/results_sha256.json. Original archives/experiments are preserved and recorded separately.
