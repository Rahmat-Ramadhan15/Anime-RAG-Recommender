"""
Minggu 3-4: RAG Pipeline (LangChain)
Status: BELUM DIIMPLEMENTASI

Mengikuti pola yang sama seperti src/build_index.py:
- Logika inti (load index, retrieval top-k, prompting, generation) ditulis di sini
  sebagai fungsi yang bisa diimpor.
- Notebook Kaggle GPU (notebooks/03_rag_pipeline_kaggle.ipynb) hanya meng-import
  dan menjalankan fungsi-fungsi ini -- dibuat setelah implementasi ini selesai.
- Menerima argumen `device` (cpu/cuda) agar portable antara laptop dan Kaggle.

Rencana:
1. Load FAISS index dari src/build_index.py (data/index/anime.index)
2. Retrieval top-k (nilai k diuji: 3, 5, 10 -- lihat configs/config.yaml)
3. Susun prompt dengan konteks hasil retrieval
4. Generate jawaban dengan model SLM (Llama-3.2-3B-Instruct), load via `device`
   yang sama dengan build_index.detect_device()
"""

raise NotImplementedError("Akan diimplementasikan pada Minggu 3-4")
