# Report source and build

The instructor's actual supplied `report/main.tex` is the starting template, not a replacement design. Its article/11pt/A4 format, 2.5 cm margins, package list, guide-box macro and required section order are preserved. Guide display is disabled for final delivery. Student names, IDs and verified contributions remain explicit placeholders.

Build in this directory with `latexmk -pdf main.tex`, or pdfLaTeX, BibTeX, then pdfLaTeX twice. Tectonic also builds this unchanged template and runs BibTeX automatically. The actual local delivery build uses a privately downloaded, SHA-verified official Tectonic 0.17.0; it is not a substitute report template. `results_values.tex` contains numeric macros generated from saved Colab CSVs, with provenance in `../docs/NUMERIC_CLAIMS.json`.

The supplied announcement says report length will be announced later and supplies no page cap. Check later course updates before final submission. This draft contains the requested theory and three central experiments; all complete tables and additional seed figures remain in the repository.

The Overleaf ZIP includes main.tex, references.bib, results_values.tex, figure files and this README. Choose main.tex as the root document. Replace author/ID/contribution placeholders, independently verify the mathematics, rebuild, inspect all pages and submit report.pdf. If final GitHub content is not yet published, complete the reviewed publication step before submission; the existing public URL is already included.
