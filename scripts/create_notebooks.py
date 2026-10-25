"""Generate readable notebooks; execution is performed separately by nbclient/Jupyter."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(s):
    return nbf.v4.new_markdown_cell(s)


def code(s, tags=None):
    return nbf.v4.new_code_cell(s, metadata={"tags": tags or []})


BOOT = '''import os, sys, json, subprocess
from pathlib import Path
ROOT = Path(os.environ.get("LEGAL_RAG_ROOT", Path.cwd())).resolve()
assert (ROOT / "src/legal_rag").is_dir(), "Set LEGAL_RAG_ROOT to the extracted project folder"
sys.path.insert(0, str(ROOT / "src"))
os.environ["PYTHONPATH"] = str(ROOT / "src")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from legal_rag.config import settings, read_json, write_json
cfg = settings()
RUN_TRAINING = os.getenv("LEGAL_RAG_TRAIN") == "1"
if RUN_TRAINING:
    import unsloth  # Must precede transformers / sentence_transformers in this kernel.
print("Nama: Kendrick Filbert")
print("Mode:", "GPU training enabled" if RUN_TRAINING else "CPU verification; GPU training NOT executed")
def cli(*args):
    subprocess.run([sys.executable, "-m", "legal_rag.cli", *args], cwd=ROOT, check=True)
'''

SETUP = '''Jika memakai GPU baru, jalankan `pip install -r requirements-training.txt` dan
`pip install -e . --no-deps`, lalu `pip check` dan restart kernel. Rincian tersedia di
`docs/RUNBOOK_GPU.md`. Ambil token melalui environment/Colab Secrets, jangan tempel di sel.
Untuk mengaktifkan training, set `LEGAL_RAG_TRAIN=1` **sebelum Run All**.
Output CPU yang melewati sel GPU tidak memenuhi persyaratan eksekusi training submission.'''


def save(name, cells):
    nb = nbf.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "authors": [{"name": "Kendrick Filbert"}],
        "project_status": "engineering_draft_until_gpu_training_and_model_inference_complete"})
    nbf.write(nb, ROOT / name)


save("Fine-tuning_submission_PGABL_Kendrick-Filbert.ipynb", [
md("# Fine-tuning Qwen3.8-27B · Kendrick Filbert\n\nTarget Advanced: dua eksperimen QLoRA 800 steps, validation dan publikasi merged_16bit. **Draf CPU: training belum dilakukan.**"),
md(SETUP), code(BOOT),
code('''from legal_rag.training import preflight
info = preflight(require_gpu=False)
print(json.dumps(info, indent=2))
print(json.dumps({k:cfg[k] for k in ["base_model", "base_revision", "dataset", "dataset_revision", "experiments"]}, indent=2))'''),
md("## Dataset resmi, split, mapping native ChatML\nDataset hanya dari tautan tugas. Deduplikasi prompt sebelum pembagian train/validation; seed 3407. Kode mapping berada di `src/legal_rag/data.py`. Cache, bila tersedia, berasal dari eksekusi Dataset.map yang sama."),
code('''from datasets import load_from_disk, DatasetDict
if (ROOT / "data/cache/train").exists():
    ds = DatasetDict({"train": load_from_disk(str(ROOT / "data/cache/train")), "test": load_from_disk(str(ROOT / "data/cache/test"))})
    report = read_json(ROOT / "artifacts/reports/dataset.json")
    print("Loaded verified Dataset.map cache")
else:
    from transformers import PreTrainedTokenizerFast
    from legal_rag.data import prepare_dataset
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg["base_model"], revision=cfg["base_revision"])
    ds, report = prepare_dataset(tokenizer)
print(json.dumps(report, indent=2, ensure_ascii=False))
print(ds["train"][0]["text"])
assert "<|im_start|>user" in ds["train"][0]["text"]
assert report["prompt_leakage"] == 0'''),
md("## NF4 + double quantization + seluruh FFN LoRA\n`load_qlora()` mengaktifkan 4-bit NF4 dan `bnb_4bit_use_double_quant=True`, kemudian memverifikasi konfigurasi hidup. Target FFN: gate_proj, up_proj, down_proj. Model Qwen ini memakai hybrid attention. Dokumentasi Unsloth mencatat risiko kuantisasi keluarga Qwen3.5; uji kompatibilitas diperlukan. Tidak ada penggantian model."),
code('''import inspect
from legal_rag.training import load_qlora
print(inspect.getsource(load_qlora))'''),
md("## Eksperimen A: LR 1e-4, rank 16, alpha 32, 800 optimizer steps\nMulai dari base model resmi. Eval/save setiap 100, logging setiap 10; tidak memakai early stopping sebelum 800 steps."),
code('''if RUN_TRAINING:
    cli("train", "sft_a")
else:
    print("NOT RUN: eksperimen A membutuhkan GPU CUDA. Tidak ada hasil loss yang diklaim.")''', ["gpu-required"]),
md("## Eksperimen B: LR 5e-5, rank 32, alpha 64, 800 optimizer steps\nProses Python baru memuat base revision yang sama, bukan melanjutkan adapter A."),
code('''if RUN_TRAINING:
    cli("train", "sft_b")
else:
    print("NOT RUN: eksperimen B membutuhkan GPU CUDA.")''', ["gpu-required"]),
code('''if all((ROOT / f"artifacts/reports/{x}.json").exists() for x in ["sft_a", "sft_b"]):
    from legal_rag.training import compare_experiments
    from legal_rag.evaluation import plot_training
    from IPython.display import display, Image
    print(compare_experiments())
    display(Image(filename=str(plot_training())))
else:
    print("Analisis kurva ditunda: kedua eksperimen belum selesai. Tidak membuat kurva sintetis.")'''),
md("### Analisis yang diisi dari hasil nyata\nBandingkan train/validation loss, selisih keduanya, checkpoint terbaik, dan indikasi overfitting. Loss lebih rendah pada dataset generik tidak otomatis berarti jawaban legal lebih akurat. Pilihan otomatis memakai validation loss; tuliskan interpretasi berdasarkan kurva sebelum pengumpulan."),
md("## Publikasi hasil SFT\nSet `HF_TOKEN` dan `HF_SFT_REPO`. API yang benar: `model.push_to_hub_merged(..., save_method='merged_16bit')`. Verifikasi anonim dan merged weights dilakukan sebelum URL ditulis."),
code('''if RUN_TRAINING:
    cli("publish", "sft")
else:
    print("NOT PUBLISHED: belum ada model SFT milik mahasiswa yang selesai dilatih.")
print((ROOT / "link_huggingface.txt").read_text(encoding="utf-8"))''', ["gpu-required"]),
])

save("GRPO_submission_PGABL_Kendrick-Filbert.ipynb", [
md("# GRPO Qwen hasil SFT sendiri · Kendrick Filbert\n\nDataset tetap Alpaca Indonesia yang ditentukan. **Draf CPU: tidak ada training GRPO yang diklaim.**"), md(SETUP), code(BOOT),
md("## Reward shaping dan ringkasan alasan\nTag `<think>` berisi ringkasan alasan singkat yang dapat diperiksa, kemudian jawaban akhir. Reward panjang/format tidak menjamin kebenaran. Correctness mengukur ground truth pada jawaban akhir saja; ringkasan sebelum final tidak dapat memperoleh skor correctness."),
code('''import inspect
from legal_rag.rewards import format_reward_func, reasoning_length_reward, correctness_reward, language_reward_func
for fn in [format_reward_func, reasoning_length_reward, correctness_reward, language_reward_func]:
    print(inspect.getsource(fn))'''),
code('''# Input berikut adalah fixture unit test, BUKAN keluaran model.
fixtures = ["<think>x", "<think>"+"a"*50+"</think>Jawab", "<think>"+"a"*200+"</think>Jawab", "<think>x</think><think>y</think>Jawab"]
print("format fixture:", format_reward_func(fixtures))
print("length fixture:", reasoning_length_reward(fixtures))
assert reasoning_length_reward(["<think>"+"a"*200]) == [1.0]
assert format_reward_func(["<think>x</think><think>y</think>Jawab"]) == [0.5]
assert correctness_reward(["<think>hak lembur</think>tidak tahu"], ["hak lembur"]) == [0.0]
print("Reward contract checks passed")'''),
md("## Memuat ulang merged SFT mahasiswa\nGRPO menolak berjalan jika receipt publik SFT sendiri tidak tersedia. Native template yang memprefill `<think>` dihapus suffix-nya saat membuat prompt GRPO agar tag harus dihasilkan model dan dapat dinilai reward."),
code('''from legal_rag.hub import verified_repo
try:
    repo, revision = verified_repo("sft")
    print("Verified student SFT:", repo, revision)
except RuntimeError as exc:
    print(str(exc))
print(json.dumps(cfg["grpo"], indent=2))'''),
md("## GRPOTrainer\nKonfigurasi awal: 200 steps, num_generations=2, completion 384, akumulasi 4, beta 0.04, non-vLLM. TRL 1.15 membatasi prompt melalui filter token karena max_prompt_length sudah dihapus dari GRPOConfig. Efektif batch harus habis dibagi jumlah generasi."),
code('''if RUN_TRAINING:
    cli("train", "grpo")
else:
    print("NOT RUN: tidak ada GPU/runtime dan model SFT publik mahasiswa.")''', ["gpu-required"]),
code('''path = ROOT / "artifacts/reports/grpo.json"
if path.exists():
    report = read_json(path)
    print(json.dumps({k:v for k,v in report.items() if k != "log_history"}, indent=2))
    print("Last real logs:", report["log_history"][-5:])
else:
    print("Belum ada log/reward training GRPO; tidak ada klaim peningkatan kualitas.")'''),
code('''if RUN_TRAINING:
    cli("publish", "grpo")
else:
    print("NOT PUBLISHED: belum ada model GRPO yang telah dilatih.")''', ["gpu-required"]),
md("Lanjutkan notebook RAG untuk menguji model GRPO dengan pertanyaan wajib staf admin lembur 3 jam. Jangan mengganti output model dengan contoh rubrik. Tinjau nilai reward, clipping, keragaman generasi, dan kualitas final; reward tinggi dapat terjadi tanpa peningkatan akurasi."),
])

save("RAG_submission_PGABL_Kendrick-Filbert.ipynb", [
md("# LexID · RAG legal dengan sitasi\n\nKendrick Filbert. Empat PDF tugas, hybrid retrieval, parent-child, HyDE, reranker, DuckDuckGo dan input/Markdown. **Pengujian lokal hanya retrieval; generation menunggu model mahasiswa.**"),
code(BOOT),
md("## 1. Korpus lengkap, metadata dan OCR\nNomor halaman adalah halaman PDF 1-based. Tabel sideways PP5 memerlukan OCR; cache menyimpan confidence dan koordinat. Regulasi tugas adalah korpus historis, bukan verifikasi status hukum terkini."),
code('''from legal_rag.corpus import load_pages, make_chunks
if not (ROOT / "data/processed/pages.jsonl").exists():
    cli("extract")
if not (ROOT / "artifacts/reports/ocr.json").exists():
    cli("ocr")
pages = load_pages()
parents, children = make_chunks(pages)
print(json.dumps(read_json(ROOT / "data/processed/manifest.json"), indent=2))
print("OCR report:", read_json(ROOT / "artifacts/reports/ocr.json"))
print("Pages, parents, children:", len(pages), len(parents), len(children))
assert len({p["metadata"]["doc_id"] for p in pages}) == 4
print("Child sample metadata:", children[0].metadata)
print("Chunk config:", {k:cfg[k] for k in ["child_size", "child_overlap", "parent_size", "parent_overlap"]})'''),
md("## 2. Embedding open-source dan FAISS lokal\nE5 memakai prefix `query:` / `passage:` dan normalized vectors untuk inner product. BM25/RRF berbobot 0.45/0.55; metadata difilter sebelum perankingan. Setidaknya lima parent kandidat diambil sebelum cross-encoder Top-3."),
code('''import torch
torch.set_num_threads(4)
from legal_rag.retrieval import HybridRetriever, Reranker
if (ROOT / "artifacts/index/children.faiss").exists():
    retriever = HybridRetriever.load()
else:
    retriever = HybridRetriever.build(parents, children)
reranker = Reranker()
print("Embedding:", cfg["embedding_model"])
print("Reranker:", cfg["reranker_model"])
print("FAISS vectors:", retriever.index.ntotal)'''),
code('''query = "Apakah pekerja yang lembur harus dibayar upah kerja lembur?"
retrieved = retriever.retrieve(query)
print("Candidates before reranking:", len(retrieved))
assert len(retrieved) >= 5
for i, row in enumerate(retrieved[:5], 1):
    print(f"\\n--- Dokumen {i} ---")
    print(row["chunk"].metadata["citation"])
    print(row["chunk"].text[:1800])
ranked = reranker.rerank(query, retrieved)
for row in ranked:
    print(row["chunk"].metadata["citation"], "raw logit", row["relevance_logit"], "sigmoid score", row["relevance_score"])'''),
code('''filtered = retriever.retrieve("uang kompensasi PKWT", filters={"doc_id": "PP_35_2021"})
assert all(row["chunk"].metadata["doc_id"] == "PP_35_2021" for row in filtered)
print("Filtered candidates:", len(filtered))
print([row["chunk"].metadata["citation"] for row in filtered[:5]])
from legal_rag.evaluation import retrieval_evaluation
evaluation = retrieval_evaluation(retriever, reranker)
print(json.dumps(evaluation, ensure_ascii=False, indent=2))'''),
md("## 3. Model hasil SFT+GRPO mahasiswa dan dua HyDE\nGenerator hanya menerima repo dalam receipt training mahasiswa. Hipotesis HyDE membantu pencarian dan tidak pernah menjadi sumber final. Skor threshold 0.45 provisional; kalibrasi memerlukan relevance labels held-out."),
code('''from legal_rag.pipeline import LegalRAG, OVERTIME_CASE
rag = None
if RUN_TRAINING:
    from legal_rag.generation import StudentGenerator
    rag = LegalRAG(retriever, reranker, StudentGenerator(stage="grpo"))
    result = rag.ask(OVERTIME_CASE)
    print("HyDE outputs:", result.get("hypotheses"))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    assert len(result["hypotheses"]) >= 2
    assert result["status"] == "answered", "Inspect sources/threshold before asserting success"
    assert "<think>" in result["answer"] and "</think>" in result["answer"]
    assert result["citation_ids_valid"]
    result["complete"] = True
    write_json(ROOT / "artifacts/reports/rag_end_to_end.json", result)
else:
    print("NOT RUN: HyDE, model reasoning summary, dan final answer belum tersedia karena training GPU belum dilakukan.")''', ["gpu-required"]),
md("## 4. Fallback DuckDuckGo ketika Top-1 rendah\nAkses web memerlukan `LEGAL_RAG_WEB=1` dan kueri publik eksplisit. Jika threshold dilampaui, sumber lokal tetap dipakai. Jika skor rendah, sumber lokal dibuang dan hanya hasil resmi .go.id menjadi konteks. Snippet web tetap memerlukan pembacaan sumber lengkap. Jalur fixture diuji terpisah; fixture bukan inferensi LLM."),
code('''from legal_rag.pipeline import should_fallback
print("Threshold:", cfg["rerank_threshold"], cfg["threshold_status"])
print("Actual top1 retrieval requires fallback:", should_fallback(ranked, cfg["rerank_threshold"]))
if rag is not None and os.getenv("LEGAL_RAG_WEB") == "1":
    public_query = "peraturan perundang-undangan perlindungan data pribadi Indonesia"
    web_demo = rag.ask(public_query, filters={"doc_id": "PP_35_2021"}, public_web_query=public_query)
    print(json.dumps(web_demo, ensure_ascii=False, indent=2))
    write_json(ROOT / "artifacts/reports/web_fallback.json", web_demo)
else:
    print("Live web fallback not executed: enable web and complete student model first.")'''),
md("## 5. Antarmuka wajib: input() + IPython.display.Markdown\nPada eksekusi otomatis, pertanyaan contoh dan `keluar` diinjeksikan ke input() agar notebook selesai tanpa menunggu terminal. Jawaban tetap dihasilkan model nyata. Jalankan sel berikutnya secara manual untuk bertanya bebas."),
code('''from legal_rag.interfaces import interactive_loop
if rag is not None:
    from unittest.mock import patch
    # Only input is scripted; model inference is real.
    with patch("builtins.input", side_effect=[OVERTIME_CASE, "keluar"]):
        interactive_loop(rag)
    write_json(ROOT / "artifacts/reports/rag_interface.json", {"complete": True, "interface": "input()+IPython.display.Markdown", "question": OVERTIME_CASE})
else:
    print("Interface inference NOT RUN: no trained student model. Function implementation:")
    import inspect
    print(inspect.getsource(interactive_loop))''', ["gpu-required"]),
code('''if rag is not None and os.getenv("LEGAL_RAG_INTERACTIVE") == "1":
    interactive_loop(rag)
else:
    print("Set LEGAL_RAG_INTERACTIVE=1 for an unscripted interactive session.")'''),
md("Opsional portfolio: `python -m legal_rag.cli ui` membuka Gradio pada localhost. Ini melengkapi loop wajib, tidak menggantikannya."),
code('''from subprocess import run
check = run([sys.executable, "scripts/validate_submission.py"], cwd=ROOT, text=True, capture_output=True)
print(check.stdout)
print("Validator exit code:", check.returncode)
print("Draf belum layak dikumpulkan" if check.returncode else "Bukti struktural lengkap; tinjau kualitas sebelum dikumpulkan")'''),
])

print("Created 3 notebooks")
