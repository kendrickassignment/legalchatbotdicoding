# Menyelesaikan training dan submission

## Kapasitas dan biaya

Model yang diminta berukuran 27B. Dokumentasi umum Unsloth menyebut minimum sekitar 22 GB VRAM untuk QLoRA 27B; kebutuhan nyata bertambah menurut arsitektur, sequence length, batch, dan GRPO. T4 free-tier 16 GB berada di bawah angka umum tersebut. Jangan menganggap Colab gratis akan menyediakan GPU yang cukup.

A100 80 GB atau kelas setara adalah sasaran konservatif untuk percobaan ini, bukan hasil benchmark proyek. Proyek tidak menyewa resource, membeli kredit, atau menjalankan job berbayar. Jika belum memiliki GPU, simpan draf ini dan sediakan runtime sebelum mengaktifkan training. Tidak ada estimasi waktu/biaya yang dapat dipercaya sebelum benchmark beberapa langkah pada hardware sebenarnya.

Qwen3.8 memakai arsitektur Qwen3.5. Dokumentasi Unsloth memperingatkan perbedaan kuantisasi 4-bit pada keluarga ini; rubrik justru mewajibkan QLoRA. Konfigurasi memenuhi tuntutan rubrik, tetapi kualitas dan kompatibilitas GPU wajib diuji. Jangan mengganti ke BF16 LoRA lalu menyebutnya QLoRA. Jangan mengganti model tanpa persetujuan pemilik proyek.

## Colab / server

1. Ekstrak ZIP ke penyimpanan yang cukup. Untuk server, salin folder proyek. Untuk Colab, unggah ZIP ke Drive, mount Drive dari notebook kosong, dan ekstrak ke `/content/PGABL_Kendrick-Filbert`.
2. Buka notebook Fine-tuning; atur direktori kerja ke folder proyek. Semua notebook mencari folder melalui `LEGAL_RAG_ROOT` atau current directory. Saat file dibuka dari Drive, pastikan source `.py` ada di folder yang sama.
3. Pilih runtime GPU yang cukup. Periksa `nvidia-smi`. Sediakan ruang untuk base weights, dua adapter/checkpoint, cache, dan merge 16-bit. Weight 27B dalam 16-bit sendiri sekitar 54 GB; disk dan RAM CPU saat merge juga perlu ruang tambahan. Cadangkan setidaknya sekitar 150 GB disk sebagai titik awal, lalu sesuaikan hasil preflight nyata.
4. Pasang `requirements-training.txt`, lalu `pip install -e . --no-deps`; jalankan `pip check`. Restart kernel. Stack kandidat dikunci ke Unsloth 2026.10.8, Transformers 5.17.0, TRL 1.15.0 sesuai batas dependency paket. Versi ini belum diuji training GPU di komputer pengembangan.
5. Tambahkan Colab Secrets `HF_TOKEN`, `HF_SFT_REPO`, `HF_GRPO_REPO`. Token write diperlukan. Repo harus di namespace Anda/organisasi Anda dan akan dibuat publik karena submission mensyaratkannya.
6. Set `LEGAL_RAG_TRAIN=1`, lalu Run All notebook Fine-tuning. Setiap eksperimen terpisah memuat base revision yang sama. Kedua run wajib selesai 800 langkah. Log setiap 10, eval/save setiap 100. Jangan berhenti dini untuk mengejar bentuk kurva.
7. Simpan notebook dengan output; periksa kurva train/validation dan generalization gap. Model terbaik berasal dari checkpoint validation terbaik, bukan snapshot sebelum training.
8. Publikasikan SFT merged_16bit melalui sel notebook. Verifikasi repo publik dan tokenizer. Lanjutkan notebook GRPO yang memuat ulang repo SFT tersebut, menjalankan GRPO, lalu mempublikasikan GRPO merged_16bit.
9. Jalankan notebook RAG: semua PDF, OCR lengkap, FAISS lokal, retrieval minimal lima kandidat, metadata filter, dua HyDE, cross-encoder Top-3, dan test case lembur wajib. Buka juga loop input/Markdown yang diwajibkan rubrik.
10. Simpan seluruh output notebook ke folder proyek yang akan dikumpulkan. Jalankan validator lalu packager final.

## Environment tanpa token di output

Dalam Colab, baca Secrets sebelum menjalankan sel GPU:

```python
import os
from google.colab import userdata
for name in ("HF_TOKEN", "HF_SFT_REPO", "HF_GRPO_REPO"):
    os.environ[name] = userdata.get(name)
os.environ["LEGAL_RAG_TRAIN"] = "1"
```

Di server, ekspor variabel melalui secret manager atau environment shell. `.env.example` hanya contoh nama. CLI tidak otomatis membaca `.env`; bila memilih file `.env`, muat dengan `dotenv.load_dotenv()` pada launcher Anda. Jangan menuliskan token dalam perintah yang disimpan di notebook.

## Menghadapi OOM / runtime putus

Checkpoint dapat dilanjutkan dengan menjalankan perintah eksperimen yang sama. Untuk OOM, periksa apakah proses lain memakai GPU, lalu turunkan sequence length atau panjang completion; jangan menurunkan SFT di bawah 800 langkah. Jika konfigurasi diubah, gunakan nama eksperimen/folder baru agar checkpoint dengan hyperparameter berbeda tidak tercampur. Untuk evaluasi, sesuaikan panjang prompt GRPO tanpa mencampurkan validation ke train. `num_generations=2` dan `gradient_accumulation_steps=4` memiliki effective batch yang habis dibagi dua.

Lakukan smoke test kompatibilitas pada proses terpisah sebelum dua run penuh. Smoke test dengan langkah singkat bukan run submission dan tidak boleh menghasilkan laporan `complete=true`. Catat perubahan konfigurasi dan simpan output `pip freeze` dari runtime yang akhirnya berhasil.

## Kriteria hasil yang perlu ditinjau manusia

Ringkasan alasan harus singkat dan merujuk bukti. Test case admin lembur memerlukan analisis syarat, perintah/persetujuan lembur, pengecualian jabatan tertentu, jam kerja, serta sumber yang relevan; jangan menyimpulkan hak seseorang hanya dari jabatan informal. Dokumen historis tugas tidak membuktikan regulasi masih berlaku pada 2026. Review sitasi halaman dan baca sumber asli, terutama tabel OCR, sebelum menandai jawaban akurat.
