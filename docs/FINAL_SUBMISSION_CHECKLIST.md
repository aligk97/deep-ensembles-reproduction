# Final submission checklist

Technical preparation is complete when the final validator and ZIP checks pass. It is not an authorization to publish or submit, and unknown personal information remains pending.

| Course/project requirement | Status | Evidence / final action |
|---|---|---|
| Own public GitHub repository | Existing public repository verified | https://github.com/aligk97/deep-ensembles-reproduction ; final prepared content must still be reviewed/pushed/merged by the user |
| Reproduction source code | Prepared | Three original source folders; scientific Python/config hashes checked |
| Free GPU platform execution | Actual saved evidence present | Three executed T4 notebooks and completed manifests, not just local compatibility checks |
| Correct execution order | Documented | Toy, MNIST, OOD; OOD requires preceding MNIST Drive RUN_ID or complete ZIP |
| Three seeds | Verified | 42,123,2026 in all primary grids |
| Mean ± SD | Verified | Sample SD ddof=1, recomputed from raw seed tables |
| No cherry-picking | Preserved | Colab primary for all experiments; all local comparisons and unfavorable results retained |
| Toy partial reproduction | Explicit | All methods/seeds have zero left-OOD noisy 95% coverage in both environments |
| Plot bands versus coverage | Explicit | Main ±3 SD; separate ±1.96 SD noisy coverage; clean-curve diagnostics distinct |
| MNIST 48 checkpoints / 63 rows | Verified | Archive SHA anchors and complete grids |
| OOD no new training | Verified | Same classifier manifest/CSV and 48 hashes; new_training=false |
| OOD sample sizes and corrupt files | Verified | 10,000 / 18,724 valid; two excluded PNGs |
| Windows/Linux ordering | Actual correction evidence | 16,719 joint image/label/ID positions; pixels unchanged; both hashes required |
| Original course LaTeX template | Preserved | Preamble/section provenance; guides hidden |
| PDF report | Actually compiled | 10 pages at delivery; final log has no undefined refs/citations/overfull boxes |
| Report contains repository link | Present | Real existing public URL, not a made-up placeholder |
| Mathematical derivations | Present | Proper score/KL, Gaussian NLL, mixture/total variance, FGSM, Brier, entropy and AUROC |
| Reproduction differences | Explicit | docs/DIFFERENCES.md, report setup/discussion |
| Presentation | Created | Nine editable English slides plus PDF, native PowerPoint renders reviewed |
| Presentation time | Planned, rehearsal pending | 7:30 speaking + 2:30 questions, 2:30 per person; measure actual human rehearsal |
| Defense material | Created | 39 Turkish Q&A, complete Turkish guide, equations, EN/TR scripts |
| Numerical consistency | Automated | scripts/validate_delivery.py checks saved CSVs against README, LaTeX macros, PDF and PPTX |
| Raw result/source preservation | Verified by separate full snapshot | Final integrity report in evidence folder; no source experiment is rerun |
| Known secret patterns / private paths | Scanned | No recognized credential patterns; personal paths sanitized in metadata copies only |
| Clean repository / relative links | Automated | No environments, raw image sets, weights or bulk probability caches; links resolve |
| Names and three student IDs | USER PENDING | Fill report author block, README author table and PowerPoint title |
| Actual contributions | USER PENDING | Replace all three contribution placeholders with verified work, not proposed speaker roles |
| AI disclosure | USER REVIEW | ChatGPT/Codex use disclosed; verify and approve independently |
| Later report page cap | USER CONFIRM | Supplied announcement says this will be announced later; no numerical limit supplied |
| Assigned presentation session | USER CONFIRM | 30 November or 7 December 2026; check current course announcements |
| Group registration/contact email | USER CONFIRM | Course Teams sign-up requirement cannot be verified from the project files |
| Read slides/notes during talk | Prohibited | Scripts are rehearsal aids only; every member must understand the whole project |
| Final GitHub publication | USER APPROVAL PENDING | No push, public repo creation or visibility change was performed |
| Final personalized document rebuild | USER PENDING | Recompile/export after author edits and inspect again |
| Formal report submission | USER PENDING | Course deadline: 13 December 2026, 23:59; students submit officially |

## Completing personal fields

Edit the report author/Contributions blocks, the main README Authors table and the PowerPoint title. Student 1/2/3 speaker labels in rehearsal material may remain roles, or be mapped to the actual names. The proposed talk division is not evidence of research contributions.

After any edit, rebuild the report (or Overleaf ZIP) and export the PPTX PDF again. Inspect every page/slide before updating the document-review receipt. The saved original experiments and numeric tables should stay untouched.


## After completing personal fields

Rebuild `report/report.pdf` and `report/overleaf_submission.zip` from edited sources, export the PowerPoint PDF again, and actually inspect every updated page/slide. Synchronize outer delivery copies too. Only after that real review, run:

```bash
python -B scripts/record_document_review.py --confirm-reviewed
python -B scripts/validate_delivery.py --output docs/validation/delivery_validation.json
```

The first command records **your review attestation**, hashes and page counts; it does not perform visual review. Numeric checks still compare documents with unchanged results. Regenerate `publication_files.txt` if files are added/removed. Rebuild the outer manifest/ZIP after final edits rather than reusing outdated hashes.
