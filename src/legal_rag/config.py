import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_FILES = (
    "PP Nomor 5 Tahun 2021.pdf", "PP Nomor 35 Tahun 2021.pdf",
    "PP Nomor 51 Tahun 2023.pdf", "UU Nomor 6 Tahun 2023.pdf",
)


def settings():
    return json.loads((ROOT / "configs/project.json").read_text(encoding="utf-8"))


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
