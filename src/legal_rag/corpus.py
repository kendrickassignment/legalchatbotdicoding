"""Page-aware extraction; original PDFs remain the source of truth."""
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from .config import EXPECTED_FILES, ROOT, settings, write_json


@dataclass
class Chunk:
    id: str
    text: str
    metadata: dict


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text):
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def extract_corpus(raw_dir=None, destination=None):
    from pypdf import PdfReader
    raw_dir = Path(raw_dir or ROOT / "data/raw")
    actual = {p.name for p in raw_dir.glob("*.pdf")}
    if actual != set(EXPECTED_FILES):
        raise ValueError(f"Corpus must contain exactly four assigned PDFs. Missing={set(EXPECTED_FILES)-actual}; extra={actual-set(EXPECTED_FILES)}")
    manifest, pages = [], []
    for name in EXPECTED_FILES:
        path = raw_dir / name
        reader = PdfReader(path)
        kind, number, year = re.match(r"(PP|UU) Nomor (\d+) Tahun (\d+)", name).groups()
        sparse = []
        for i, page in enumerate(reader.pages, 1):
            text = clean_text(page.extract_text() or "")
            if len(text) < 80:
                sparse.append(i)
            pages.append({"text": text, "metadata": {
                "doc_id": f"{kind}_{number}_{year}", "filename": name,
                "regulation_type": kind, "number": int(number), "year": int(year),
                "page": i, "citation": f"{kind} Nomor {number} Tahun {year}, halaman PDF {i}",
                "articles": re.findall(r"\bPasal\s+(\d+[A-Z]?)\b", text),
                "source_type": "assigned_pdf", "legal_status": "historical_corpus_not_current_law_verified",
            }})
        manifest.append({"filename": name, "sha256": digest(path), "pages": len(reader.pages), "sparse_pages": sparse})
        print(f"{name}: {len(reader.pages)} halaman; {len(sparse)} halaman teks minim", flush=True)
    destination = Path(destination or ROOT / "data/processed")
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / "pages.jsonl").open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(__import__("json").dumps(p, ensure_ascii=False) + "\n")
    write_json(destination / "manifest.json", manifest)
    return pages, manifest


def load_pages():
    import json
    path = ROOT / "data/processed/pages.jsonl"
    if not path.exists():
        return extract_corpus()[0]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def make_chunks(pages, cfg=None):
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    cfg = cfg or settings()
    parents, children = [], []
    p_split = RecursiveCharacterTextSplitter(chunk_size=cfg["parent_size"], chunk_overlap=cfg["parent_overlap"])
    c_split = RecursiveCharacterTextSplitter(chunk_size=cfg["child_size"], chunk_overlap=cfg["child_overlap"])
    for p in pages:
        m = p["metadata"]
        for pi, text in enumerate(p_split.split_text(p["text"])):
            pid = f"{m['doc_id']}:p{m['page']}:parent{pi}"
            parents.append(Chunk(pid, text, {**m, "parent_id": pid}))
            for ci, small in enumerate(c_split.split_text(text)):
                children.append(Chunk(f"{pid}:c{ci}", small, {**m, "parent_id": pid}))
    return parents, children


def serialize_chunks(chunks):
    return [asdict(c) for c in chunks]
