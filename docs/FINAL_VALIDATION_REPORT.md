# Final validation report

## Completed scope

All ten technical delivery phases are complete. Original experiment folders/results are preserved; no model was trained or evaluated again. The existing public remote was verified read-only. Nothing was staged, committed, pushed, created remotely or made public by this preparation.

| Check | Actual result |
|---|---|
| Saved-experiment scientific audit | 478 passed, 0 failed |
| Delivery consistency validator | 979 passed, 0 failed |
| Intentional negative controls | 2/2 correctly rejected, disposable copies only |
| CSV/binary files really opened | 318; PDF parsed, PNG decoded, PPTX/ZIP CRC verified |
| Source/config identity | 42 original source/config hashes pinned |
| Numeric tables/figures identity | 308 original CSV/figure hashes pinned |
| Seed aggregation | All local and Colab groups; sample SD ddof=1 |
| Train/validation/test splits | All three saved arrays exactly equal across local/Colab |
| Report | Actual Tectonic compile; original instructor template; 10 pages |
| Final LaTeX log | No undefined refs/citations or overfull boxes |
| Presentation | Actual native Microsoft PowerPoint PPTX/PDF; 9 editable slides |
| Full visual review | All 10 pages and 9 slide renders inspected; zero slide overflow |
| Rehearsal | 7:30 speaking + 2:30 questions; 2:30 speaking per student; human rehearsal pending |
| Defense | 39 Turkish Q&A, Turkish guide/equations, EN/TR scripts |
| Original input preservation | 87,720 files, 11,235,299,443 bytes; zero changes |
| Privacy and links | No recognized secret pattern/personal Windows path in public tree; relative links resolve |
| Models or experiments rerun | 0 |

The validator checks saved results against the README, 68 provenance-backed LaTeX numeric macros, actual report PDF tokens, native PPTX tables/text and presentation PDF. It also validates complete local/Colab result grids, template/section provenance, six notebook structures and three actual executed notebook records, references, Overleaf ZIP/source identity, review hashes, source/result byte pins and negative findings. False values injected into temporary report/PPTX copies were detected.

Machine-readable receipts are in `docs/validation/` and the outer `06_EXPERIMENT_EVIDENCE/`. The outer package has `MANIFEST_SHA256.json`; `scripts/verify_submission_package.py` verifies every file and final ZIP CRC without executing experiments. The external `FINAL_SUBMISSION.zip.sha256` holds the final archive hash outside the ZIP to avoid self-reference.

## Scientific outcome and limits

All primary findings use actual Colab T4 exports for all methods; local results are supplementary throughout. At M=5, MNIST accuracies are 98.54±0.11%, 98.47±0.21%, 98.61±0.09%, 97.32±0.22%; OOD AUROCs are 0.9653±0.0038, 0.9346±0.0118, 0.9668±0.0023, 0.7779±0.0736 in Ensemble/R/AT/MC Dropout order. SDs are across three seeds.

Toy regression is partial: all methods/seeds in both environments have zero left-OOD noisy 95% coverage. High exact-mixture NLLs and local Ensemble M=1 seed=123 AUROC≈0.3790 are retained. Main toy bands are ±3 SD; noisy coverage uses ±1.96 SD. AUROC/entropy gap are additions to original Figure 3(a). No significance test, universal calibration claim or causal proof of toy failure mechanisms is implied.

## Mandatory human actions still pending

This technically complete package has intentional personal placeholders and is not an official submission.

1. Complete three names, IDs and verified contributions in report/README/PowerPoint. Speaker roles do not establish research contributions.
2. Independently review derivations, findings and ChatGPT/Codex disclosure; rehearse without reading notes/slides.
3. Confirm any later page limit, assigned 30 November/7 December session and Teams group/contact registration. No numerical page cap is in the supplied announcement; the report is 10 pages.
4. Recompile/export personalized documents, recreate the Overleaf ZIP, inspect all pages/slides, update the honest review receipt, synchronize outer copies and rerun validation.
5. Review the allowlisted publication diff and explicitly approve publication. Existing public URL: https://github.com/aligk97/deep-ensembles-reproduction. The final tree has not been pushed.
6. Submit officially by the announced 13 December 2026 23:59 deadline, subject to later course updates.

No unresolved technical build, render, saved-data consistency or package-integrity failure remains. Remaining limitations are documented science and human/course/publication actions above.
