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

## 7. Kondisi A (atau kondisi lain) menghasilkan jawaban yang "melanjutkan sendiri" percakapan palsu
Contoh gejala: model menjawab pertanyaan Anda, lalu mengarang pertanyaan susulan dan
menjawabnya sendiri, diakhiri teks aneh seperti "Konteks selesai."

Penyebab: model instruction-tuned (Llama-3.x) punya format chat khusus dengan token
`<|start_header_id|>`, `<|eot_id|>`, dsb. Kalau prompt dikirim sebagai teks mentah
(tanpa `tokenizer.apply_chat_template()`) dan tanpa terminator token yang benar,
model tidak tahu kapan harus berhenti -- ia melanjutkan generate hingga `max_new_tokens`
habis, termasuk mengarang giliran percakapan baru.

Ini sudah diperbaiki di `src/rag_pipeline.py`: `load_llm()` sekarang menyimpan
`self.tokenizer` dan `self.terminators` (termasuk `<|eot_id|>`), dan `generate()`
memakai `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` alih-alih
menyusun prompt sebagai string biasa. Kalau Anda menulis pemanggilan LLM sendiri di
tempat lain (mis. notebook eksperimen), pastikan selalu memakai pola yang sama:
```python
messages = [{"role": "system", "content": ...}, {"role": "user", "content": ...}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
output = pipe(prompt, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id)
```

## 8. `TypeError: ...__init__() got an unexpected keyword argument 'load_in_4bit'`
Versi `transformers` yang lebih baru (di Kaggle biasanya auto-update ke versi terbaru)
sudah tidak menerima `load_in_4bit=True` langsung di `from_pretrained()`. Ini sudah
diperbaiki di `src/rag_pipeline.py` dengan `BitsAndBytesConfig(load_in_4bit=True, ...)`.
Kalau Anda menulis kode load model sendiri di tempat lain, pola yang benar:
```python
from transformers import BitsAndBytesConfig
quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map="auto")
```
Juga: `torch_dtype=` sudah deprecated, ganti jadi `dtype=` (masih berfungsi tapi memunculkan warning).
Kalau error serupa muncul lagi untuk parameter lain, cek changelog `transformers` versi yang terpasang
(`pip show transformers`) -- API loading model di library ini cukup sering berubah antar rilis.
