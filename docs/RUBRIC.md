# Matriks rubrik dan bukti

Target implementasi adalah Advanced pada kedua kriteria. Status akhir lihat `artifacts/reports/submission_validation.json`. Implementasi kode tidak sama dengan bukti eksekusi.

| Syarat | Lokasi implementasi | Bukti yang harus ada |
|---|---|---|
| Dataset wajib, Dataset.map, native chat template tercetak | `data.py`, notebook Fine-tuning | `dataset.json`, contoh native, output sel |
| NF4 + double quantization + FFN LoRA | `training.load_qlora` | konfigurasi model hidup, jumlah parameter trainable |
| SFTTrainer ≥800 langkah | `training.run_sft` | `sft_a.json`, `sft_b.json`, output notebook |
| Train/validation, eval per steps, logging | `data.py`, SFTConfig | split tanpa prompt leakage, kurva loss nyata |
| ≥2 eksperimen hyperparameter | `configs/project.json` | masing-masing run selesai, analisis kurva dan pilihan |
| Public merged_16bit | `hub.publish` | repo publik, revision dan receipts, link_huggingface.txt |
| Reload SFT sendiri, GRPOTrainer | `training.run_grpo` | source repo/revision, grpo.json |
| 4 reward sesuai tahapan | `rewards.py` | unit tests + kurva reward GRPO nyata |
| Test case lembur dengan rationale dan final | `pipeline.OVERTIME_CASE`, notebook RAG | rag_end_to_end.json, output model asli |
| Seluruh 4 PDF | `corpus.py`, `ocr.py` | manifest hash, 1.949 halaman, OCR yang dibutuhkan |
| Ukuran chunk dan overlap eksplisit | `configs/project.json` | children/parents, provenance halaman |
| Embedding open-source, local FAISS | `retrieval.py` | children.faiss dan konfigurasi |
| Retrieval query relevan | `evaluation.py`, notebook RAG | retrieval_eval.json dan output potongan sumber |
| Student model, context/question, generation | `generation.py`, `pipeline.py` | repo training diverifikasi + output inference |
| input() dan IPython Markdown loop | `interfaces.interactive_loop` | output interaksi model asli tersimpan |
| Metadata enrichment/filter/citation | `corpus.py`, `retrieval.py`, `pipeline.py` | filter demo, nomor halaman dan ID sumber |
| Ensemble BM25 + semantic, bobot, ≥5 kandidat | `HybridRetriever.retrieve` | candidate_count ≥5 sebelum rerank |
| Parent-child retrieval | `corpus.make_chunks`, `retrieval.py` | child→parent, parent dideduplikasi |
| HyDE ≥2 keluaran model | `LegalRAG.hyde` | hypotheses dari model mahasiswa, bukan string mock |
| Cross-encoder Top-K | `Reranker.rerank` | logit/score nyata dan Top-3 |
| Top-1 threshold → DuckDuckGo | `should_fallback`, `LegalRAG.ask` | jalur skor rendah, kueri publik, hasil web nyata |

Bonus portfolio: source modular, CLI, konfigurasi revision, CI, contract tests, OCR resume, provenance, evaluasi retrieval, validasi sitasi, private-by-default web egress, runbook, validasi submission dan ZIP. Ini bukan sertifikasi production; multi-user authentication, deployment, monitoring, legal approval, dan kalibrasi belum tersedia.

Fungsi language reward menggunakan klasifikasi bahasa sebagai proxy; tidak dapat membuktikan seluruh tata bahasa Indonesia sempurna. Correctness overlap dapat reward-hack dan bukan evaluator hukum. Hindari menuliskan klaim bahwa GRPO pasti meningkatkan akurasi tanpa pengukuran sebelum/sesudah pada data held-out.
