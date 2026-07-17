# Chatbot Rekomendasi Anime Berbasis RAG + SLM

Skripsi: **Rancang Bangun Chatbot Rekomendasi Anime Menggunakan Retrieval-Augmented Generation (RAG) dan Transformer**

Sistem rekomendasi anime yang menggabungkan retrieval semantik (FAISS) atas dataset MyAnimeList dengan enrichment data real-time (Jikan API) dan generasi jawaban menggunakan Small Language Model (SLM ±3B parameter).

## Stack

| Komponen       | Teknologi                                                                                                                |
| -------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Dataset utama  | [Anime & Manga Analytics Dataset (2026)](https://www.kaggle.com/datasets/patelris/anime-and-manga-dataset-2026) — Kaggle |
| Data pelengkap | [Jikan API](https://jikan.moe/) (unofficial MyAnimeList REST API)                                                        |
| Embedding      | `sentence-transformers/all-MiniLM-L6-v2`                                                                                 |
| Vector DB      | FAISS                                                                                                                    |
| Framework RAG  | LangChain                                                                                                                |
| Model bahasa   | SLM ±3B (kandidat utama: Llama-3.2-3B-Instruct), kuantisasi GGUF 4-bit                                                   |
| UI             | Gradio                                                                                                                   |
| Deployment     | Hugging Face Spaces                                                                                                      |

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
│   ├── enrichment.py           # Minggu 5 -- DONE (Jikan API client, rate limit + cache)
│   ├── guardrails.py           # Minggu 5 -- DONE (deteksi & tolak konten eksplisit)
│   ├── run_experiment.py       # Minggu 7 -- DONE (jalankan 99 query x 3 kondisi, butuh GPU/Kaggle)
│   ├── llm_judge.py            # Minggu 7 -- DONE (LLM-as-a-Judge via Gemini, tidak butuh GPU)
│   └── analyze_results.py      # Minggu 7 -- DONE (Wilcoxon signed-rank, refusal rate, tidak butuh GPU)
├── notebooks/
│   ├── 02_build_index_kaggle.ipynb   # runner Kaggle GPU untuk build_index.py
│   ├── 03_rag_pipeline_kaggle.ipynb  # runner Kaggle GPU untuk rag_pipeline.py
│   └── 04_run_experiment_kaggle.ipynb # runner Kaggle GPU untuk run_experiment.py
├── tests/                      # 100 query test set + ground truth (Minggu 4)
└── docs/
    ├── Dokumen_Rincian_Project_Skripsi.docx
    └── TROUBLESHOOTING_KAGGLE.md   # gated model, git clone, GPU quota, dsb.
```

## Alur Kerja: Script vs Notebook

Prinsip yang dipakai di repo ini: **logika inti selalu ditulis sebagai modul `.py` di `src/`**, bukan langsung di notebook. Notebook hanya dipakai sebagai _runner_ tipis untuk memanfaatkan GPU gratis di Kaggle, dengan meng-_import_ modul yang sama.

| Tahap                                               | Butuh GPU?                        | Dijalankan sebagai                                                      |
| --------------------------------------------------- | --------------------------------- | ----------------------------------------------------------------------- |
| Minggu 1 — Preprocessing (`preprocess.py`)          | Tidak                             | Script biasa di laptop                                                  |
| Minggu 2 — Embedding + FAISS (`build_index.py`)     | Opsional (lebih cepat dengan GPU) | Modul, dipanggil dari `notebooks/02_build_index_kaggle.ipynb` di Kaggle |
| Minggu 3-4 — RAG pipeline + SLM (`rag_pipeline.py`) | Ya (inferensi LLM)                | Modul, akan dipanggil dari notebook Kaggle serupa                       |
| Minggu 5 — Enrichment API (`enrichment.py`)         | Tidak                             | Script biasa (panggilan HTTP)                                           |

Keuntungan pola ini: kode tetap bisa di-_review_ dan di-_diff_ di Git (bukan tersebar di sel-sel notebook), sekaligus tetap bisa memanfaatkan GPU gratis Kaggle tanpa menulis ulang logika.

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
- [x] **Minggu 4** — Test set 99 query + ground truth lengkap, evaluasi Recall@k/Precision@k/MRR selesai. **k final = 5** (lihat `configs/config.yaml`). Temuan penting: retrieval semantik murni lemah pada constraint numerik (episode/rating) di query multi-turn -- dicatat sebagai keterbatasan arsitektural untuk bab pembahasan.
- [x] **Minggu 5** — `enrichment.py`: Jikan API client dengan rate limit (~1 req/detik) dan cache snapshot JSON di `data/api_cache/` untuk reproducibility. `guardrails.py`: deteksi & tolak query eksplisit LANGSUNG sebelum panggil LLM (Lapisan 3), terintegrasi ke `rag_pipeline.py`. Poster tetap dari `image_url` dataset (bukan Jikan) sesuai keputusan sebelumnya.
- [x] **Minggu 6** — Kondisi A (baseline SLM murni) **sudah otomatis tercakup** sejak Minggu 3, karena `generate()` dirancang satu fungsi untuk ketiga kondisi lewat flag `use_retrieval`/`use_enrichment` (bukan tiga implementasi terpisah). Tidak ada pekerjaan tambahan di sini -- lebih cepat dari jadwal.
- [x] **Minggu 7** — Eksperimen penuh 99 query x 3 kondisi (`run_experiment.py`, GPU/Kaggle), LLM-as-a-Judge via Gemini (`llm_judge.py`, tidak butuh GPU), analisis Wilcoxon signed-rank + refusal rate (`analyze_results.py`, tidak butuh GPU)
- [ ] **Minggu 8** — Snapshot Jikan API final, tinjau ulang `tests/final_evaluation_report.json` untuk bab hasil
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

| Kategori           | Kriteria                                                                                | Alasan                                                     |
| ------------------ | --------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| Konten             | `rating` memuat "Rx", atau `genres`/`themes` memuat "Hentai"                            | Batasan konten wajib (Bagian 3.1)                          |
| Data hygiene       | Duplikat `mal_id`, status "Not yet aired" + sinopsis kosong, metadata inti kosong semua | Kebersihan data, bukan kurasi kualitas                     |
| **Tidak difilter** | Skor rendah, popularitas rendah, tag "Ecchi"                                            | Menjaga cakupan retrieval & menghindari bias komunitas MAL |

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

**Cara membaca hasil evaluasi:**

- Kategori **faktual** (1 jawaban benar per query) → baca **Recall@k dan MRR** sebagai metrik utama.
- Kategori **filter_atribut**, **similaritas_rekomendasi**, **multi_turn_refinement** (bisa ratusan
  hingga ribuan jawaban valid per query, karena kriterianya longgar) → baca **Precision@k** sebagai
  metrik utama. Recall@k di kategori ini akan **selalu kecil/mendekati 0 secara struktural**
  (k=10 dibagi ratusan/ribuan kandidat valid), bukan berarti retrieval gagal -- ini keterbatasan
  definisi ground truth, bukan kualitas sistem. Precision@k menjawab pertanyaan yang relevan bagi
  user: dari k rekomendasi yang ditampilkan, berapa yang benar-benar valid?

**Catatan validasi:** 35 query kategori `similaritas_rekomendasi` punya ground truth _semi-otomatis_
(overlap genre + demografi) dan wajib divalidasi manual pada subset (`needs_manual_validation: true`
di `tests/test_set.jsonl`) sebelum dilaporkan sebagai metrik final di skripsi -- lihat Bagian 6.3
dokumen rincian project.

## Menjalankan Minggu 5 (Enrichment + Guardrail)

Enrichment butuh koneksi internet (panggilan ke Jikan API), tidak butuh GPU:

```bash
python3 src/enrichment.py    # uji cepat: ambil data 3 anime contoh, cek data/api_cache/
python3 src/guardrails.py    # uji cepat: cek deteksi query eksplisit vs query normal
```

`rag_pipeline.py` otomatis memakai keduanya:

- Setiap `generate()` dipanggil, query dicek dulu lewat `guard_query()` -- kalau terdeteksi
  eksplisit, langsung ditolak tanpa memanggil retrieval maupun LLM sama sekali (Lapisan 3).
- Kondisi C sekarang memanggil `pipe.enrich(mal_ids)` untuk data Jikan API sungguhan
  (status tayang, episode terkini, trailer), bukan lagi placeholder kosong.

Cache respons API tersimpan di `data/api_cache/<mal_id>.json` (gitignored) -- ini snapshot
untuk reproducibility (Bagian 6.6), simpan folder ini secara terpisah (mis. zip manual) kalau
mau melampirkan bukti data mentah di lampiran skripsi.

**Catatan keandalan Jikan API:** error `504 Gateway Timeout` sesekali itu wajar -- Jikan API
gratis tanpa SLA (sudah dicatat sebagai keterbatasan sejak awal). `enrichment.py` sudah
memakai endpoint dasar (bukan `/full`, yang membawa data relasi/streaming yang tidak kita
butuhkan dan lebih berat) plus retry otomatis (3x percobaan, backoff 2/4/6 detik) untuk
error 502/503/504. Kalau tetap gagal setelah retry, query itu dilewati (dicatat sebagai
`{}` -- enrichment kosong, bukan error yang menghentikan seluruh proses).

**Catatan keterbatasan (bukan bug):** sebagian trailer YouTube bisa muncul "Video tidak
tersedia di negara Anda" -- ini pembatasan wilayah oleh pengunggah/pemegang hak di YouTube,
di luar kendali sistem kita. URL yang dihasilkan `enrichment.py` sudah benar (bisa dicek
lewat `youtube_id`-nya), pembatasannya murni dari sisi platform YouTube berdasarkan lokasi
pengakses. Sebutkan ini sebagai keterbatasan sumber data eksternal di bab keterbatasan skripsi.

## Menjalankan Minggu 7 (Eksperimen Penuh + LLM-as-a-Judge)

Tiga tahap, dua di antaranya TIDAK butuh GPU:

**Tahap 1 (butuh GPU/Kaggle)** -- jalankan `notebooks/04_run_experiment_kaggle.ipynb`:
menjalankan 99 query x 3 kondisi (297 pemanggilan LLM), hasil `tests/experiment_results.jsonl`.
Unduh file ini dari panel Output Kaggle, taruh di `tests/` repo lokal Anda.

**Tahap 2 (tidak butuh GPU, hanya internet)** -- di laptop:

```bash
# Set API key dulu (dapatkan gratis di aistudio.google.com/apikey)
$env:GEMINI_API_KEY="xxx"          # PowerShell
# atau: export GEMINI_API_KEY=xxx  # Linux/Mac

pip install google-genai   # paket google-generativeai sudah deprecated per pertengahan 2026
python src/llm_judge.py
```

Menilai setiap jawaban dengan model BEDA dari generator (Gemini, bukan Llama-3.2-3B) supaya
tidak self-preference bias -- lihat Bagian 6.2 dokumen rincian project. Hasil:
`tests/judge_scores.jsonl`.

**Tahap 3 (tidak butuh GPU/internet, murni statistik)** -- di laptop:

```bash
python src/analyze_results.py
```

Menghasilkan `tests/final_evaluation_report.json`: skor kualitas per kondisi/kategori,
uji Wilcoxon signed-rank berpasangan (A vs B, B vs C), dan refusal rate untuk kategori
out-of-scope/adversarial. Ini yang dilampirkan ke bab hasil skripsi.

Lihat `docs/Dokumen_Rincian_Project_Skripsi.docx` untuk pembahasan lengkap alasan akademik di balik setiap keputusan.

## Lisensi

Untuk keperluan akademik (skripsi S1). Dataset dan API pihak ketiga mengikuti lisensi/ketentuan masing-masing (Kaggle dataset license, Jikan API — unofficial, tanpa SLA).
