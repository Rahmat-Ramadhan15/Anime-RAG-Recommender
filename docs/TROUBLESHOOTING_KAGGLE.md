# Troubleshooting Kaggle Notebook (GPU)

## 1. Repo GitHub privat -> `git clone` gagal
Kalau repo Anda privat, `git clone https://github.com/...` akan minta autentikasi dan gagal di Kaggle.
Solusi tercepat: buat repo **public** (aman untuk skripsi, tidak ada data sensitif/API key di dalamnya
karena sudah di-gitignore), atau gunakan Personal Access Token:
```
!git clone https://<token>@github.com/<username>/skripsi-anime-rag.git
```

## 2. Llama-3.2-3B-Instruct adalah GATED MODEL
Ini yang PALING SERING bikin stuck: model Llama-3.2 di Hugging Face perlu Anda:
1. Login ke huggingface.co, buka halaman model, klik "Agree and access repository"
2. Buat Access Token di huggingface.co/settings/tokens
3. Di Kaggle: Add-ons -> Secrets -> tambahkan `HF_TOKEN`
4. Di notebook:
```python
from huggingface_hub import login
from kaggle_secrets import UserSecretsClient
login(UserSecretsClient().get_secret("HF_TOKEN"))
```
Tanpa ini, `from_pretrained(...)` akan gagal dengan error 401/403 walau nama modelnya benar.

## 3. GPU quota habis
Akun Kaggle gratis punya kuota ~30 jam GPU/minggu. Cek sisa kuota di halaman notebook (kanan atas).
Kalau habis, kembali ke CPU sementara (device otomatis fallback ke "cpu" di kode kita) atau tunggu reset mingguan.

## 4. Session restart = install ulang semuanya
Kaggle session bersifat sementara -- setiap kali notebook di-restart, `pip install` dan `git clone`
harus dijalankan ulang dari awal. Ini normal, bukan bug.

## 5. Path dataset tidak ketemu
Kalau upload `anime_documents.jsonl` sebagai Kaggle Dataset, path-nya selalu
`/kaggle/input/<nama-dataset-anda>/anime_documents.jsonl` -- sesuaikan `SRC` di notebook
dengan nama dataset Anda sendiri (bisa dilihat di panel "Input" sebelah kanan notebook).

## 6. Out of memory saat load model 3B
Kalau muncul CUDA OOM meski sudah pakai GPU:
- Tambahkan `torch_dtype=torch.float16` atau `load_in_4bit=True` (butuh `bitsandbytes`) saat load model
- Turunkan `batch_size` pada tahap embedding (Minggu 2) kalau OOM terjadi di situ, bukan saat LLM
