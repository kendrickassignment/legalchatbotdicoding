import argparse
import sys
import zipfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_submission import ROOT, validate


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--draft", action="store_true", help="Package unfinished engineering work, not a final submission")
    args = p.parse_args()
    report = validate(check_hub=not args.draft)
    if not args.draft and not report["submission_ready"]:
        raise SystemExit("Final ZIP refused: finish the missing evidence in artifacts/reports/submission_validation.json")
    target = ROOT.parent / ("PGABL_Kendrick-Filbert_DRAFT.zip" if args.draft else "PGABL_Kendrick-Filbert.zip")
    excluded = {"__pycache__", ".git", ".pytest_cache", ".ipynb_checkpoints", "cache", "checkpoints", "merged", "adapters"}
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=5) as z:
        for path in ROOT.rglob("*"):
            if path.is_file() and not (excluded & set(path.relative_to(ROOT).parts)) and path.name != ".env" and path.suffix not in {".pyc", ".safetensors", ".bin"}:
                z.write(path, Path(ROOT.name) / path.relative_to(ROOT))
    print(target)
