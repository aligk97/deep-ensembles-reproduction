# Reviewed publication workflow

Existing repository: https://github.com/aligk97/deep-ensembles-reproduction. Public visibility was confirmed read-only. Nothing has been pushed or staged by delivery preparation. Do not commit the original workspace wholesale.

Use a fresh clone and an ordinary feature branch so existing history remains available. The following commands are **instructions only**, and the push/PR must wait for explicit approval after review. Paths are examples to adjust locally.

```powershell
git clone https://github.com/aligk97/deep-ensembles-reproduction.git deep-ensembles-publish
Set-Location deep-ensembles-publish
git switch -c final-submission
# Copy the CONTENTS of the delivered clean repository into this fresh clone,
# preserving the clone's .git directory. Do not use destructive /MIR copy flags.
git rm -r --cached .
git add --pathspec-from-file=publication_files.txt
git diff --cached --stat
git diff --cached --check
# Inspect all changes and complete names/IDs/contributions before publication.
git commit -m "Prepare final CMP4509 reproduction delivery"
# Only after explicit approval:
git push -u origin final-submission
gh pr create --base main --head final-submission --title "Final CMP4509 reproduction delivery" --body-file PR_DESCRIPTION.md
```

`git rm --cached` changes only this fresh clone's index; it does not delete working files. The explicit allowlist avoids re-adding obsolete large files left in the clone. Review deletions in the staged diff. Never force-push or change repository visibility as part of this preparation. After merge, confirm that the report URL opens publicly, the source/notebooks/report links work and the final documents are present on main.


## After completing personal fields

Rebuild `report/report.pdf` and `report/overleaf_submission.zip` from edited sources, export the PowerPoint PDF again, and actually inspect every updated page/slide. Synchronize outer delivery copies too. Only after that real review, run:

```bash
python -B scripts/record_document_review.py --confirm-reviewed
python -B scripts/validate_delivery.py --output docs/validation/delivery_validation.json
```

The first command records **your review attestation**, hashes and page counts; it does not perform visual review. Numeric checks still compare documents with unchanged results. Regenerate `publication_files.txt` if files are added/removed. Rebuild the outer manifest/ZIP after final edits rather than reusing outdated hashes.
