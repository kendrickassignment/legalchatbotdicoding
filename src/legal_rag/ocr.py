"""Resumable OCR for the scanned landscape tables in the assigned corpus."""
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from .config import ROOT, write_json

_engine = None


def recognize(task):
    import cv2
    import fitz
    import numpy as np
    from rapidocr_onnxruntime import RapidOCR
    global _engine
    if _engine is None:
        _engine = RapidOCR(intra_op_num_threads=2, inter_op_num_threads=1,
                          det_limit_side_len=1280, det_limit_type="max", rec_batch_num=16)
    filename, page_number = task
    with fitz.open(ROOT / "data/raw" / filename) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=140, alpha=False)
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # Infer dominant text direction: some appendix pages are upright, others
        # sideways. Rotating every scanned page corrupts paragraph reading order.
        boxes, _ = _engine.text_det(arr)
        horizontal, vertical = 0., 0.
        if boxes is not None:
            for box in boxes:
                width = np.linalg.norm(box[1] - box[0])
                height = np.linalg.norm(box[3] - box[0])
                if height > 1.5 * width:
                    vertical += width * height
                elif width > 1.5 * height:
                    horizontal += width * height
        rotation = 90 if vertical > horizontal else 0
        if rotation:
            arr = np.rot90(arr, k=-1).copy()
        result, _ = _engine(arr, use_cls=False)
    lines = result or []
    # Preserve OCR reading sequence and boxes. Table associations require visual QA.
    return {"filename": filename, "page": page_number,
        "text": "\n".join(line[1] for line in lines),
        "mean_confidence": sum(float(line[2]) for line in lines) / max(1, len(lines)),
        "line_count": len(lines), "engine": "rapidocr_onnxruntime 1.4.4", "render_dpi": 140, "rotation_degrees": rotation,
        "boxes": [{"box": np.asarray(line[0]).tolist(), "text": line[1], "confidence": float(line[2])} for line in lines]}


def ocr_sparse_pages(workers=2):
    path = ROOT / "data/processed/pages.jsonl"
    pages = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    cache = ROOT / "data/processed/ocr_pages"
    cache.mkdir(parents=True, exist_ok=True)
    targets = [p for p in pages if len(p["text"]) < 80 or p["metadata"].get("extraction_method") == "rapidocr"]
    tasks = [(p["metadata"]["filename"], p["metadata"]["page"]) for p in targets]
    def outfile(task):
        return cache / (Path(task[0]).stem.replace(" ", "_") + f"_p{task[1]}.json")
    def cached(task):
        try:
            obj = json.loads(outfile(task).read_text(encoding="utf-8"))
            return obj["page"] == task[1] and obj["filename"] == task[0]
        except (OSError, ValueError, KeyError):
            return False
    missing = [t for t in tasks if not cached(t)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(recognize, t): t for t in missing}
        for i, future in enumerate(as_completed(futures), 1):
            task = futures[future]
            result = future.result()
            write_json(outfile(task), result)
            if i % 10 == 0 or i == 1:
                print(f"OCR {i}/{len(missing)}: {task[0]}, halaman {task[1]}; confidence {result['mean_confidence']:.3f}", flush=True)
    for p in pages:
        task = (p["metadata"]["filename"], p["metadata"]["page"])
        if outfile(task).exists():
            result = json.loads(outfile(task).read_text(encoding="utf-8"))
            p["text"] = result["text"]
            p["metadata"].update(extraction_method="rapidocr", ocr_mean_confidence=result["mean_confidence"],
                ocr_rotation_degrees=result.get("rotation_degrees", 90),
                table_layout_verified=False, articles=__import__("re").findall(r"\bPasal\s+(\d+[A-Z]?)\b", result["text"]))
        else:
            p["metadata"]["extraction_method"] = "native_pdf"
    with path.open("w", encoding="utf-8") as f:
        for p in pages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    report = {"target_pages": len(targets), "completed": len(tasks), "remaining_sparse": [p["metadata"] for p in pages if len(p["text"]) < 80],
        "low_confidence_pages": [p["metadata"] for p in pages if p["metadata"].get("ocr_mean_confidence", 1.) < .8],
        "limitation": "OCR text and table row/column relationships require manual verification against original page images"}
    write_json(ROOT / "artifacts/reports/ocr.json", report)
    return report
