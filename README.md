# Chatbot Rekomendasi Anime Berbasis RAG + SLM

Skripsi: **Rancang Bangun Chatbot Rekomendasi Anime Menggunakan Retrieval-Augmented Generation (RAG) dan Transformer**

Sistem rekomendasi anime yang menggabungkan retrieval semantik (FAISS) atas dataset MyAnimeList dengan enrichment data real-time (Jikan API) dan generasi jawaban menggunakan Small Language Model (SLM ±3B parameter).

## Stack

| Komponen | Teknologi |
|---|---|
| Dataset utama | [Anime & Manga Analytics Dataset (2026)](https://www.kaggle.com/datasets/patelris/anime-and-manga-dataset-2026) — Kaggle |
| Data pelengkap | [Jikan API](https://jikan.moe/) (unofficial MyAnimeList REST API) |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB | FAISS |
| Framework RAG | LangChain |
| Model bahasa | SLM ±3B (kandidat utama: Llama-3.2-3B-Instruct), kuantisasi GGUF 4-bit |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

## Struktur Repo

```
skripsi-anime-rag/
├── README.md
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml            # seluruh parameter project terpusat di sini
├── data/
│   ├── raw/                    # taruh anime_dataset.csv di sini (tidak di-commit, lihat .gitignore)
│   ├── processed/              # hasil preprocessing (tidak di-commit)
│   └── api_cache/              # snapshot respons Jikan API untuk reproducibility
├── src/
│   ├── preprocess.py           # Minggu 1 -- DONE (script, CPU, tidak butuh GPU)
│   ├── build_index.py          # Minggu 2 -- DONE (modul portable CPU/GPU)
│   ├── rag_pipeline.py         # Minggu 3-4 -- DONE (modul, mendukung Kondisi A/B/C)
│   ├── enrichment.py           # Minggu 5 -- TODO
│   └── guardrails.py           # Minggu 5 -- TODO
├── notebooks/
│   ├── 02_build_index_kaggle.ipynb   # runner Kaggle GPU untuk build_index.py
│   └── 03_rag_pipeline_kaggle.ipynb  # runner Kaggle GPU untuk rag_pipeline.py
├── tests/                      # 100 query test set + ground truth (Minggu 4)
└── docs/
    ├── Dokumen_Rincian_Project_Skripsi.docx
    └── TROUBLESHOOTING_KAGGLE.md   # gated model, git clone, GPU quota, dsb.
```

## Alur Kerja: Script vs Notebook

Prinsip yang dipakai di repo ini: **logika inti selalu ditulis sebagai modul `.py` di `src/`**, bukan langsung di notebook. Notebook hanya dipakai sebagai *runner* tipis untuk memanfaatkan GPU gratis di Kaggle, dengan meng-*import* modul yang sama.

| Tahap | Butuh GPU? | Dijalankan sebagai |
|---|---|---|
| Minggu 1 — Preprocessing (`preprocess.py`) | Tidak | Script biasa di laptop |
| Minggu 2 — Embedding + FAISS (`build_index.py`) | Opsional (lebih cepat dengan GPU) | Modul, dipanggil dari `notebooks/02_build_index_kaggle.ipynb` di Kaggle |
| Minggu 3-4 — RAG pipeline + SLM (`rag_pipeline.py`) | Ya (inferensi LLM) | Modul, akan dipanggil dari notebook Kaggle serupa |
| Minggu 5 — Enrichment API (`enrichment.py`) | Tidak | Script biasa (panggilan HTTP) |

Keuntungan pola ini: kode tetap bisa di-*review* dan di-*diff* di Git (bukan tersebar di sel-sel notebook), sekaligus tetap bisa memanfaatkan GPU gratis Kaggle tanpa menulis ulang logika.

**Cara pakai `notebooks/02_build_index_kaggle.ipynb` di Kaggle:**
1. Upload notebook ini ke Kaggle, aktifkan GPU (Settings → Accelerator) dan Internet (Settings → Internet)
2. Upload `anime_documents.jsonl` (hasil Minggu 1) sebagai Kaggle Dataset input, atau sesuaikan path di sel notebook
3. Jalankan seluruh sel — notebook akan `git clone` repo ini, lalu memanggil `src/build_index.py` langsung
4. Unduh `data/index/anime.index` dan `data/index/id_mapping.pkl` dari panel Output, taruh di repo lokal Anda (folder ini sudah di-gitignore)

## Setup

```bash
git clone <url-repo-anda>
cd skripsi-anime-rag
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Unduh `anime_dataset.csv` dari Kaggle (link di atas) dan letakkan di `data/raw/anime_dataset.csv`.

## Progres per Minggu

- [x] **Minggu 1** — Preprocessing data: verifikasi `mal_id`, filtering rating Rx/Hentai, data hygiene (duplikat, status belum tayang + info minim), desain template chunking
- [x] **Minggu 2** — Modul embedding + FAISS indexing (`build_index.py`) siap, portable CPU/GPU; notebook runner Kaggle tersedia
- [x] **Minggu 3** — Modul `rag_pipeline.py` siap dan terverifikasi di Kaggle GPU: retrieval top-k, chat template resmi Llama-3.x (bukan prompt mentah), guardrail dasar di system prompt, mendukung Kondisi A/B/C langsung lewat satu fungsi `generate()`. Kondisi A terbukti berhalusinasi (contoh: kesalahan genre & karakter karangan) -- bukti kualitatif awal untuk argumen kebutuhan RAG.
- [x] **Minggu 4** — Generator 100 query test set + ground truth (`tests/build_test_set.py`, tidak butuh GPU); script evaluasi Recall@k/MRR (`src/evaluate_retrieval.py`, tidak butuh GPU) siap dijalankan
- [ ] **Minggu 5** — Enrichment via Jikan API (Kondisi C), guardrail konten, system prompt out-of-scope handling
- [ ] **Minggu 6** — Implementasi Kondisi A (baseline SLM murni)
- [ ] **Minggu 7** — Evaluasi otomatis (Recall@k/MRR) + LLM-as-a-Judge
- [ ] **Minggu 8** — Snapshot Jikan API, analisis statistik (Wilcoxon signed-rank)
- [ ] **Minggu 9** — Human evaluation, UI Gradio, deployment HF Spaces
- [ ] **Minggu 10** — Buffer: perbaikan bug, revisi
- [ ] **Minggu 11-12** — Penulisan bab hasil & pembahasan

## Menjalankan Minggu 1 (Preprocessing)

```bash
python3 src/preprocess.py
```

Output:
- `data/processed/anime_filtered.csv` — dataset setelah filtering
- `data/processed/anime_documents.jsonl` — dokumen siap-embedding (satu baris JSON per anime)
- `data/processed/filtering_report.json` — laporan jumlah entri yang difilter (untuk dilampirkan di bab metodologi)

### Kebijakan Filtering (ringkas)

| Kategori | Kriteria | Alasan |
|---|---|---|
| Konten | `rating` memuat "Rx", atau `genres`/`themes` memuat "Hentai" | Batasan konten wajib (Bagian 3.1) |
| Data hygiene | Duplikat `mal_id`, status "Not yet aired" + sinopsis kosong, metadata inti kosong semua | Kebersihan data, bukan kurasi kualitas |
| **Tidak difilter** | Skor rendah, popularitas rendah, tag "Ecchi" | Menjaga cakupan retrieval & menghindari bias komunitas MAL |

> **Update kebijakan (ditemukan saat eksplorasi data):** genre "Erotica" (berbeda dari "Ecchi") menandakan
> konten seksual eksplisit, sehingga ditambahkan ke daftar filter konten sejajar dengan "Hentai".
> Ini mengubah `anime_documents.jsonl` -- **index FAISS perlu dibangun ulang** kalau Anda sudah
> pernah menjalankan Minggu 2 sebelum update ini.

## Menjalankan Minggu 4 (Test Set + Evaluasi Retrieval)

Tidak butuh GPU/Kaggle -- jalankan langsung di laptop:

```bash
python3 tests/build_test_set.py        # generate 100 query + ground truth -> tests/test_set.jsonl
python3 src/evaluate_retrieval.py      # hitung Recall@k & MRR untuk k=3/5/10 -> tests/topk_evaluation_report.json
```

Setelah melihat hasilnya, isi `retrieval.top_k_final` di `configs/config.yaml` dengan nilai k terpilih.

**Catatan validasi:** 35 query kategori `similaritas_rekomendasi` punya ground truth *semi-otomatis*
(overlap genre + demografi) dan wajib divalidasi manual pada subset (`needs_manual_validation: true`
di `tests/test_set.jsonl`) sebelum dilaporkan sebagai metrik final di skripsi -- lihat Bagian 6.3
dokumen rincian project.

Lihat `docs/Dokumen_Rincian_Project_Skripsi.docx` untuk pembahasan lengkap alasan akademik di balik setiap keputusan.

## Lisensi

Untuk keperluan akademik (skripsi S1). Dataset dan API pihak ketiga mengikuti lisensi/ketentuan masing-masing (Kaggle dataset license, Jikan API — unofficial, tanpa SLA).
