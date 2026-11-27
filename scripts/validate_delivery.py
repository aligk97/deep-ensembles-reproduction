"""Validate saved delivery artifacts only. NEVER import or execute experiments."""
from pathlib import Path, PurePosixPath
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import posixpath
import re
import statistics
import sys
import urllib.parse
import xml.etree.ElementTree as ET
import zipfile

ROOT = Path(__file__).resolve().parent.parent
CHECKS = []

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024**2), b""):
            h.update(block)
    return h.hexdigest()

def rows(path):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))

def check(name, condition, **details):
    CHECKS.append({"name": name, "passed": bool(condition), **details})
    if not condition:
        raise AssertionError(name)

def close(a,b):
    return math.isclose(float(a),float(b),rel_tol=1e-12,abs_tol=1e-10)

def aggregate(raw_path, summary_path, group_keys, metrics, std_suffix):
    raw, summary = rows(raw_path), rows(summary_path)
    for saved in summary:
        group = [r for r in raw if all(r[k] == saved[k] for k in group_keys)]
        key = "/".join(saved[k] for k in group_keys)
        check("three seeds/"+str(raw_path.relative_to(ROOT))+"/"+key, len(group)==3 and
              {int(r["run_seed"]) for r in group}=={42,123,2026})
        for metric in metrics:
            values=[float(r[metric]) for r in group if r.get(metric) not in ("",None)]
            if not values:
                continue
            check("sample statistics/"+str(raw_path.relative_to(ROOT))+"/"+key+"/"+metric,
                  close(statistics.mean(values),saved[metric+"_mean"]) and
                  close(statistics.stdev(values),saved[metric+"_"+std_suffix]))
    return raw,summary

def link_targets(text):
    pattern=r"!?\[[^\]]*\]\((<[^>]+>|[^)\s]+)(?:\s+[^)]*)?\)"
    for match in re.finditer(pattern,text):
        yield match.group(1).strip("<>")

def main():
    global ROOT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args()
    ROOT=args.root.resolve()
    result={"successful":False,"new_training":False,"model_forwards":0,
            "existing_experiment_reruns":0,"scope":"Saved CSV, source, notebook, PDF, PPTX and link checks only","checks":CHECKS}
    try:
        required=["README.md",".gitignore","requirements.txt","report/main.tex","report/references.bib","report/report.pdf",
                  "report/overleaf_submission.zip","presentation/Deep_Ensembles_Reproduction.pptx",
                  "presentation/Deep_Ensembles_Reproduction.pdf","presentation/PRESENTATION_SCRIPT_EN.md",
                  "presentation/PRESENTATION_SCRIPT_TR.md","presentation/SLIDE_TIMING.md",
                  "docs/DEFENSE_QUESTIONS_AND_ANSWERS.md","docs/PROJECT_STUDY_GUIDE_TR.md","docs/KEY_EQUATIONS.md",
                  "docs/REPOSITORY_AUDIT.md","docs/FINAL_RESULTS_ANALYSIS.md","docs/FINAL_SUBMISSION_CHECKLIST.md",
                  "docs/FINAL_VALIDATION_REPORT.md"]
        check("required deliverables",all((ROOT/p).is_file() for p in required))
        files=[p for p in ROOT.rglob("*") if p.is_file() and ".git" not in p.relative_to(ROOT).parts]
        forbidden=[p.relative_to(ROOT).as_posix() for p in files if
                   any(part in (".venv",".deps","deps","__pycache__") for part in p.relative_to(ROOT).parts)
                   or p.suffix.lower() in (".pt",".pth",".npy",".npz") or p.name.endswith(".tar.gz")]
        check("no dependencies/raw datasets/weights/bulk caches in public tree",not forbidden,files=forbidden)
        check("no oversized public file",all(p.stat().st_size<95*1024**2 for p in files))
        texts={}
        for p in files:
            if p.suffix.lower() in (".md",".txt",".py",".json",".csv",".tex",".bib",".ipynb"):
                texts[p]=p.read_text(encoding="utf-8-sig")
        check("all text artifacts opened as UTF-8",True,files=len(texts))
        secrets=re.compile(r"(?:ghp_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|sk-proj-[A-Za-z0-9_-]{24,}|AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)")
        secret_files=[p.relative_to(ROOT).as_posix() for p,text in texts.items() if secrets.search(text)]
        check("recognized secret patterns absent",not secret_files,files=secret_files)
        private_paths=[p.relative_to(ROOT).as_posix() for p,text in texts.items()
                       if re.search(r"C:(?:\\\\|\\|/)Users(?:\\\\|\\|/)[A-Za-z0-9]",text)]
        check("personal Windows paths absent",not private_paths,files=private_paths)
        missing=[]
        for p,text in texts.items():
            parts=[text] if p.suffix==".md" else []
            if p.suffix==".ipynb":
                notebook=json.loads(text)
                parts=["".join(c["source"]) if isinstance(c["source"],list) else c["source"] for c in notebook["cells"] if c["cell_type"]=="markdown"]
            for part in parts:
                for target in link_targets(part):
                    if target.startswith(("#","http://","https://","mailto:","data:")):
                        continue
                    if "://" in target:
                        continue
                    target=urllib.parse.unquote(target.split("#",1)[0].split("?",1)[0])
                    if target and not (p.parent/target).exists():
                        missing.append({"from":p.relative_to(ROOT).as_posix(),"target":target})
        check("all relative Markdown/notebook links resolve",not missing,missing=missing)
        for p in files:
            if p.suffix==".py":
                compile(p.read_text(encoding="utf-8"),str(p),"exec")
        check("all Python sources compile without execution",True)
        source_manifest=read(ROOT/"docs/validation/source_sha256.json")
        for relative,expected in source_manifest.items():
            check("unchanged scientific/support source/"+relative,sha(ROOT/relative)==expected)
        data_manifest=read(ROOT/"docs/validation/results_sha256.json")
        check("all published raw CSVs/figures byte-preserved",all(sha(ROOT/name)==expected for name,expected in data_manifest.items()),files=len(data_manifest))
        for env in ("local","colab"):
            base=ROOT/"results"/env
            mr,ms=aggregate(base/"mnist_classification/raw_results.csv",base/"mnist_classification/summary.csv",("method","M"),
                            ("classification_error","accuracy","nll","brier"),"std")
            oraw,osum=aggregate(base/"mnist_notmnist_ood/results_per_seed.csv",base/"mnist_notmnist_ood/results_summary.csv",
                                ("method","M"),("mean_id_entropy","mean_ood_entropy","entropy_gap","ood_auroc"),"std")
            tr,ts=aggregate(base/"toy_regression/07_seed_experiments/aggregate/metrics/metrics_by_seed.csv",
                            base/"toy_regression/07_seed_experiments/aggregate/metrics/summary_mean_sd.csv",
                            ("method","region"),("rmse_clean","rmse_noisy","nll_moment","coverage95","width95","avg_aleatoric_std","avg_epistemic_std","nll_exact_mixture"),"sd")
            check(env+"/complete result grids",len(mr)==63 and len(ms)==21 and len(oraw)==36 and len(osum)==12 and len(tr)==48 and len(ts)==16)
            left=[r for r in tr if r["region"]=="left_ood"]
            check(env+"/all negative toy left coverages retained",len(left)==12 and all(float(r["coverage95"])==0 for r in left))
        facts=read(ROOT/"results/numeric_facts.json")
        readme=(ROOT/"README.md").read_text(encoding="utf-8")
        labels={"ensemble":"Ensemble","random":"Ensemble + R","adversarial":"Ensemble + AT","mc_dropout":"MC Dropout"}
        for method,label in labels.items():
            m,o=facts["mnist_m5"][method],facts["ood_m5"][method]
            def form(a,b,digits,scale=1):
                return f"{a*scale:.{digits}f} ± {b*scale:.{digits}f}"
            expected_row="| "+label+" | "+" | ".join([
                form(m["accuracy_mean"],m["accuracy_std"],2,100),
                form(m["nll_mean"],m["nll_std"],4),
                form(m["brier_mean"],m["brier_std"],5),
                form(o["ood_auroc_mean"],o["ood_auroc_std"],4)])+" |"
            check("README numeric row/"+method,expected_row in readme)
        macros=(ROOT/"report/results_values.tex").read_text(encoding="utf-8")
        latex=(ROOT/"report/main.tex").read_text(encoding="utf-8")
        for claim in read(ROOT/"docs/NUMERIC_CLAIMS.json"):
            csv_path,query=claim["evidence"].split(": ",1)
            parts=query.split(",")
            conditions=dict(part.split("=",1) for part in parts[:-1])
            metric=parts[-1]
            row=next(r for r in rows(ROOT/csv_path) if all(r[k]==v for k,v in conditions.items()))
            suffix="sd" if "toy_regression" in csv_path else "std"
            check("claim from actual CSV/"+claim["macro"],close(claim["mean"],row[metric+"_mean"]) and close(claim["sample_sd"],row[metric+"_"+suffix]))
            expected=f"{claim['mean']*claim['scale']:.{claim['digits']}f} \\pm {claim['sample_sd']*claim['scale']:.{claim['digits']}f}"
            check("LaTeX numeric macro/"+claim["macro"],"\\newcommand{\\"+claim["macro"]+"}{"+expected+"}" in macros)
        provenance=read(ROOT/"docs/INSTRUCTOR_TEMPLATE_PROVENANCE.json")
        check("instructor preamble preserved",latex.split("\\input{results_values.tex}",1)[0]==provenance["expected_preamble"])
        check("instructor required section order preserved",re.findall(r"\\section\*?\{([^}]+)\}",latex)==provenance["required_sections"])
        check("guides hidden",r"\showguidesfalse" in latex and r"\showguidestrue" not in latex)
        bib=(ROOT/"report/references.bib").read_text(encoding="utf-8")
        cited=set()
        for group in re.findall(r"\\cite[tp]?\{([^}]+)\}",latex):
            cited.update(group.split(","))
        bib_keys=set(re.findall(r"@\w+\{([^,]+),",bib))
        check("all citations have actual BibTeX entries",cited<=bib_keys)
        check("real public repository URL in report","https://github.com/aligk97/deep-ensembles-reproduction" in latex)
        check("three editable author slots",latex.split(r"\author{",1)[1].split(r"\date{",1)[0].count(r"\and")==2)
        for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",latex):
            check("report figure exists/"+image,(ROOT/"report"/image).is_file())
        import nbformat
        for p in (ROOT/"notebooks").rglob("*.ipynb"):
            nb=nbformat.read(p,as_version=4)
            nbformat.validate(nb)
            code=[c for c in nb.cells if c.cell_type=="code"]
            for n,cell in enumerate(code):
                compile(cell.source,str(p)+":"+str(n),"exec")
            if "executed" in p.parts:
                check("actual executed notebook/"+p.name,all(c.execution_count is not None for c in code) and
                      not any(o.output_type=="error" for c in code for o in c.outputs),code_cells=len(code))
        check("six runnable/evidence notebooks",len(list((ROOT/"notebooks").rglob("*.ipynb")))==6)
        spec=read(ROOT/"presentation/SLIDE_SPEC.json")
        check("9 slides, 7:30 speech, balanced three speakers",len(spec["slides"])==9 and
              sum(s["seconds"] for s in spec["slides"])==450 and all(sum(s["seconds"] for s in spec["slides"] if s["student"]==person)==150 for person in (1,2,3)))
        ns={"a":"http://schemas.openxmlformats.org/drawingml/2006/main","p":"http://schemas.openxmlformats.org/presentationml/2006/main"}
        pptx=ROOT/"presentation/Deep_Ensembles_Reproduction.pptx"
        slide_texts={}
        with zipfile.ZipFile(pptx) as package:
            check("PPTX ZIP CRC",package.testzip() is None)
            slide_names=[n for n in package.namelist() if re.fullmatch(r"ppt/slides/slide\d+\.xml",n)]
            check("PPTX nine actual slide parts",len(slide_names)==9)
            for n in range(1,10):
                tree=ET.fromstring(package.read(f"ppt/slides/slide{n}.xml"))
                slide_texts[n]=" ".join(e.text or "" for e in tree.findall(".//a:t",ns))
                if n in (4,8):
                    check("editable native table/slide"+str(n),tree.find(".//a:tbl",ns) is not None)
                for x in tree.findall(".//a:xfrm",ns):
                    off=x.find("a:off",ns);size=x.find("a:ext",ns)
                    if off is None or size is None: continue
                    check(f"PPTX bounds/{n}/{off.attrib}",int(off.get("x"))>=0 and int(off.get("y"))>=0 and
                          int(off.get("x"))+int(size.get("cx"))<=960*12700+12700 and
                          int(off.get("y"))+int(size.get("cy"))<=540*12700+12700)
            embedded={hashlib.sha256(package.read(name)).hexdigest() for name in package.namelist() if name.startswith("ppt/media/")}
            for s in spec["slides"]:
                if s.get("image"):
                    check("actual unmodified figure embedded/slide"+str(spec["slides"].index(s)+1),sha(ROOT/s["image"]) in embedded)
            # Every internal OOXML relationship resolves within the package.
            package_names=set(package.namelist())
            rel_ns={"r":"http://schemas.openxmlformats.org/package/2006/relationships"}
            missing_rel=[]
            for name in package_names:
                if not name.endswith(".rels"): continue
                directory=posixpath.dirname(posixpath.dirname(name)) if "/_rels/" in name else ""
                for rel in ET.fromstring(package.read(name)):
                    if rel.get("TargetMode")=="External":continue
                    target=posixpath.normpath(posixpath.join(directory,rel.get("Target","")))
                    if target.startswith("/"):target=target[1:]
                    if target not in package_names:missing_rel.append((name,target))
            check("all internal PPTX relationships resolve",not missing_rel,missing=missing_rel)
        for method in ("ensemble","random","adversarial","mc_dropout"):
            for metric in ("ood_auroc","entropy_gap"):
                d=facts["ood_m5"][method]
                token=f"{d[metric+'_mean']:.4f} ± {d[metric+'_std']:.4f}"
                check("PPTX OOD table value/"+method+"/"+metric,token in slide_texts[8])
        for method in ("adversarial","mc_dropout"):
            d=facts["mnist_m5"][method]
            check("PPTX accuracy/"+method,f"{100*d['accuracy_mean']:.2f} ± {100*d['accuracy_std']:.2f}" in slide_texts[6])
        negative=next(r for r in rows(ROOT/"results/local/mnist_notmnist_ood/results_per_seed.csv")
                      if r["run_seed"]=="123" and r["method"]=="ensemble" and r["M"]=="1")
        check("negative local OOD value preserved",close(negative["ood_auroc"],facts["local_negative_auroc"]) and "0.3790" in latex)
        from pypdf import PdfReader
        report_reader=PdfReader(ROOT/"report/report.pdf")
        deck_reader=PdfReader(ROOT/"presentation/Deep_Ensembles_Reproduction.pdf")
        reviewed=read(ROOT/"docs/validation/visual_review.json")
        check("actual readable PDFs match reviewed page count and 9 slides",len(report_reader.pages)==reviewed["report_pages"] and len(deck_reader.pages)==9 and
              all(len(p.extract_text())>100 for p in report_reader.pages) and all(len(p.extract_text())>40 for p in deck_reader.pages))
        report_text=re.sub(r"\s+","", "".join(p.extract_text() for p in report_reader.pages))
        for claim in read(ROOT/"docs/NUMERIC_CLAIMS.json"):
            if re.search(r"\\"+claim["macro"]+r"\b",latex):
                for key in ("mean","sample_sd"):
                    token=f"{claim[key]*claim['scale']:.{claim['digits']}f}"
                    check("rendered report numeric token/"+claim["macro"]+"/"+key,token in report_text)
        deck_text="".join(p.extract_text() for p in deck_reader.pages)
        for key in ("0.9653","0.9346","0.9668","0.7779","98.61","97.32"):
            check("rendered presentation numeric token/"+key,key in deck_text)
        visual=read(ROOT/"docs/validation/visual_review.json")
        check("review receipts match final PDF/PPTX",visual["report_pdf_sha256"]==sha(ROOT/"report/report.pdf") and
              visual["presentation_pptx_sha256"]==sha(pptx) and visual["presentation_pdf_sha256"]==sha(ROOT/"presentation/Deep_Ensembles_Reproduction.pdf")
              and visual["all_pages_rendered"] and visual["all_rendered_pages_and_slides_visually_reviewed"])
        with zipfile.ZipFile(ROOT/"report/overleaf_submission.zip") as package:
            check("Overleaf ZIP CRC",package.testzip() is None)
            for name in ("main.tex","references.bib","results_values.tex","README.md"):
                check("Overleaf source current/"+name,package.read(name)==(ROOT/"report"/name).read_bytes())
            for image in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}",latex):
                check("Overleaf figure current/"+image,package.read(image)==(ROOT/"report"/image).read_bytes())
        questions=(ROOT/"docs/DEFENSE_QUESTIONS_AND_ANSWERS.md").read_text(encoding="utf-8")
        check("39 defense questions",len(re.findall(r"^## \d+\.",questions,re.M))==39)
        equations=(ROOT/"docs/KEY_EQUATIONS.md").read_text(encoding="utf-8")
        check("equation Markdown delimiters/notation",equations.count("$$")%2==0 and r"\mathbb{R}" in equations and "xinmathbb" not in equations)
        result.update(successful=True,report_pages=len(report_reader.pages),presentation_slides=9,
                      manual_actions_pending=["Complete names and student IDs","Complete verified contributions and approve AI disclosure",
                                              "Confirm later course page cap and assigned presentation date","Approve/review final GitHub publication",
                                              "Rebuild/check personalized PDF and rehearse talk","Submit officially"],
                      submission_ready_without_human_review=False)
    except BaseException as exc:
        result["error"]=repr(exc)
        raise
    finally:
        result.update(passed=sum(c["passed"] for c in CHECKS),failed=sum(not c["passed"] for c in CHECKS))
        if args.output:
            args.output.parent.mkdir(parents=True,exist_ok=True)
            args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
        print(f"Delivery validation: successful={result['successful']}; passed={result['passed']}; failed={result['failed']}",flush=True)

if __name__=="__main__":
    main()
