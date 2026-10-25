# Isi model card setelah training nyata

Gunakan metadata `base_model: Qwen/Qwen3.8-27B`, `datasets: [Ichsan2895/alpaca-gpt4-indonesian]`, `language: [id]`, dan pipeline sesuai artefak export yang benar-benar dihasilkan. Jangan menyebut model text-only jika export masih menyertakan konfigurasi vision.

Tuliskan nama Kendrick Filbert, upstream model revision, dataset revision dan atribusi CC BY-SA 4.0, tanggal/hardware/stack aktual, cara preprocessing, deduplikasi/split, dua eksperimen lengkap, alasan pilihan checkpoint, QLoRA NF4 double quantization, target FFN dan LoRA rank, hyperparameter GRPO, reward serta keterbatasannya. Verifikasi kewajiban lisensi dataset dan upstream sebelum distribusi yang lebih luas.

Cantumkan loss dan evaluasi yang benar-benar terukur beserta ukuran dan batas dataset uji. Sediakan contoh inference dengan chat template yang sama, contoh sitasi, serta refusal ketika bukti kurang. Jelaskan bahwa model tidak dilatih pada dokumen hukum; pengetahuan dokumen diberikan melalui RAG saat inference. Jangan menyertakan hasil uji CPU sebagai hasil training, dan jangan mengklaim nasihat hukum atau status regulasi terkini.
