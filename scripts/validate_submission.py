"""Exit nonzero for incomplete evidence; this draft is intentionally not submittable."""
import argparse
import json
import re
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from legal_rag.config import ROOT, EXPECTED_FILES, read_json, write_json
from legal_rag.corpus import digest


def validate(check_hub=False):
    issues = []
    ocr_path = ROOT / "artifacts/reports/ocr.json"
    if not ocr_path.exists() or read_json(ocr_path).get("remaining_sparse"):
        issues.append("OCR coverage incomplete or sparse pages remain")
    if not (ROOT / "artifacts/index/children.faiss").exists():
        issues.append("Missing full-corpus local FAISS index")
    for filename in EXPECTED_FILES:
        if not (ROOT / "data/raw" / filename).is_file():
            issues.append("Missing assigned PDF: " + filename)
    manifest_path = ROOT / "data/processed/manifest.json"
    if manifest_path.exists():
        for source in read_json(manifest_path):
            path = ROOT / "data/raw" / source["filename"]
            if path.exists() and digest(path) != source["sha256"]:
                issues.append("PDF checksum changed: " + source["filename"])
    else:
        issues.append("Missing source PDF manifest")
    required = ["Fine-tuning_submission_PGABL_Kendrick-Filbert.ipynb", "GRPO_submission_PGABL_Kendrick-Filbert.ipynb", "RAG_submission_PGABL_Kendrick-Filbert.ipynb"]
    for filename in required:
        path = ROOT / filename
        if not path.exists():
            issues.append("Missing notebook: " + filename)
            continue
        nb = read_json(path)
        code = [c for c in nb["cells"] if c["cell_type"] == "code" and c["source"]]
        if any(c.get("execution_count") is None for c in code):
            issues.append("Unexecuted cells: " + filename)
        if any(o.get("output_type") == "error" for c in code for o in c.get("outputs", [])):
            issues.append("Notebook errors: " + filename)
    for name in ["sft_a", "sft_b", "grpo"]:
        path = ROOT / "artifacts/reports" / f"{name}.json"
        if not path.exists():
            issues.append("Missing real training evidence: " + name)
        else:
            report = read_json(path)
            if not report.get("complete") or report.get("global_step", 0) < (800 if name.startswith("sft") else 1):
                issues.append("Incomplete training: " + name)
    for filename in ["rag_end_to_end.json", "rag_interface.json"]:
        path = ROOT / "artifacts/reports" / filename
        if not path.exists() or not read_json(path).get("complete"):
            issues.append("Missing actual model/interface evidence: " + filename)
    web_path = ROOT / "artifacts/reports/web_fallback.json"
    if not web_path.exists() or not read_json(web_path).get("fallback_used"):
        issues.append("Missing actual low-score DuckDuckGo fallback evidence")
    links = ROOT / "link_huggingface.txt"
    urls = re.findall(r"https://huggingface.co/([\w.-]+/[\w.-]+)", links.read_text() if links.exists() else "")
    if len(urls) < 2:
        issues.append("Both real public SFT and GRPO model links are required")
    if check_hub:
        from huggingface_hub import HfApi
        for repo in urls:
            try:
                info = HfApi(token=False).model_info(repo)
                if info.private:
                    issues.append("Private model: " + repo)
            except Exception:
                issues.append("Model not anonymously accessible: " + repo)
    report = {"submission_ready": not issues, "issues": issues, "note": "Code implementation and executed CPU checks do not replace completed GPU training and RAG generation evidence."}
    write_json(ROOT / "artifacts/reports/submission_validation.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-hub", action="store_true")
    args = parser.parse_args()
    result = validate(args.check_hub)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["submission_ready"] else 1)
