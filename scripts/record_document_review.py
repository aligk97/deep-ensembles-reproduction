"""Record the user's real visual review after author/ID personalization."""
from pathlib import Path
import argparse
import hashlib
import json
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-reviewed",action="store_true",help="Attest that YOU actually reviewed every updated PDF page and PPTX slide")
    args=parser.parse_args()
    if not args.confirm_reviewed:
        parser.error("First inspect every updated report page and presentation slide, then explicitly confirm. This script does not perform visual review.")
    report=ROOT/"report/report.pdf"
    pptx=ROOT/"presentation/Deep_Ensembles_Reproduction.pptx"
    pdf=ROOT/"presentation/Deep_Ensembles_Reproduction.pdf"
    record={"successful":True,"report_pdf_sha256":digest(report),"report_pages":len(PdfReader(report).pages),
            "presentation_pptx_sha256":digest(pptx),"presentation_pdf_sha256":digest(pdf),
            "presentation_slides":len(PdfReader(pdf).pages),"all_pages_rendered":True,
            "all_rendered_pages_and_slides_visually_reviewed":True,"renders":[],
            "review_mode":"USER ATTESTATION after actual review in PDF/PowerPoint viewers",
            "visual_findings":"User confirmed actual full-page/slide review after personalization; this script only records hashes/page counts.",
            "automated_visual_review":False}
    (ROOT/"docs/validation/visual_review.json").write_text(json.dumps(record,indent=2)+"\n",encoding="utf-8")
    print("User-attested review receipt updated. Now run scripts/validate_delivery.py; no numerical result files were changed.")

if __name__=="__main__":main()
