"""
Minggu 3-4: RAG Pipeline (LangChain)

Modul ini portable CPU/GPU (sama seperti build_index.py) dan mendukung
LANGSUNG tiga Kondisi ablation study yang sudah disepakati (Bagian 5
dokumen rincian project):

    Kondisi A -- SLM saja            : use_retrieval=False, use_enrichment=False
    Kondisi B -- SLM + RAG           : use_retrieval=True,  use_enrichment=False
    Kondisi C -- SLM + RAG + Enrich  : use_retrieval=True,  use_enrichment=True

Enrichment (Jikan API) akan disambungkan setelah src/enrichment.py selesai
(Minggu 5) -- untuk sekarang parameter use_enrichment disiapkan sebagai
placeholder di signature fungsi supaya struktur ablation-nya sudah utuh.

Dijalankan lewat:
  - Laptop (CPU)          : python3 src/rag_pipeline.py   (lambat untuk generasi LLM)
  - Kaggle Notebook (GPU) : notebooks/03_rag_pipeline_kaggle.ipynb
"""

import pickle
from pathlib import Path

import faiss
import yaml
import re
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

from build_index import load_config, detect_device, load_documents
from guardrails import guard_query


SYSTEM_PROMPT = """Anda adalah asisten rekomendasi anime berbahasa Indonesia.
Tugas Anda: merekomendasikan anime dan menjawab pertanyaan seputar anime
HANYA berdasarkan konteks yang diberikan di bawah.

Batasan wajib:
- JANGAN merekomendasikan atau membahas anime dengan konten dewasa/eksplisit,
  meskipun diminta pengguna. Tolak dengan sopan jika diminta.
- JANGAN menjawab pertanyaan di luar topik anime (coding, matematika, curhat, dsb).
  Arahkan pengguna kembali ke topik rekomendasi anime.
- JANGAN membahas detail plot/ending secara mendalam (hindari spoiler).
- Jika informasi tidak ada dalam konteks, katakan tidak tahu -- jangan mengarang.
- SANGAT PENTING: HANYA rekomendasikan anime yang JUDULNYA ADA di dalam konteks
  yang diberikan. JANGAN PERNAH menyebut atau merekomendasikan anime lain di luar
  konteks, sekalipun anime itu terkenal atau menurut Anda relevan (mis. Naruto,
  One Piece, Bleach, dsb, kecuali memang tercantum di konteks). Kalau tidak ada
  satupun anime di konteks yang benar-benar cocok, katakan dengan jujur bahwa
  Anda tidak menemukan kecocokan yang baik dari data yang tersedia -- JANGAN
  mengarang alternatif dari luar konteks.

Format jawaban WAJIB seperti ini untuk SETIAP anime yang Anda rekomendasikan
(salin judul PERSIS seperti tertulis di baris "Judul:" pada konteks):

### [Judul Persis Sesuai Konteks]
(1-2 kalimat alasan kenapa direkomendasikan, dikaitkan dengan permintaan pengguna)

Ulangi blok ini untuk setiap anime yang direkomendasikan. Jangan pakai format lain
(jangan pakai penomoran 1/2/3, jangan pakai bold **judul**) -- WAJIB pakai "### "
di awal baris judul persis seperti contoh di atas.
"""

PROMPT_TEMPLATE = PromptTemplate.from_template(
    """Konteks anime yang relevan:
{context}

Pertanyaan pengguna: {query}"""
)


class RagPipeline:
    """
    Satu kelas untuk ketiga kondisi (A/B/C) -- kondisi ditentukan oleh
    flag use_retrieval dan use_enrichment saat generate() dipanggil,
    BUKAN oleh kelas terpisah. Ini menjaga system prompt tetap identik
    di ketiga kondisi (poin penting untuk perbandingan apple-to-apple,
    lihat Bagian 5 dokumen rincian project).
    """

    def __init__(self, config_path: str = "configs/config.yaml", device: str | None = None):
        self.config_path = config_path
        self.cfg = load_config(config_path)
        self.device = device or detect_device()
        print(f"[INFO] Device: {self.device}")

        self.embed_model = SentenceTransformer(
            self.cfg["embedding"]["model_name"], device=self.device
        )

        self.index = None
        self.mal_ids = []
        self.doc_lookup = {}
        self.metadata = {}  # {mal_id: {title, title_english, image_url, score}} -- untuk UI (poster)
        self.llm = None  # diisi oleh load_llm() atau load_llm_gguf()
        self.llm_backend = None  # "transformers" (GPU/Kaggle) atau "gguf" (CPU/deployment)
        self.tokenizer = None
        self.terminators = None
        self._jikan_client = None  # diisi lazy oleh enrich()

    def load_index(self, index_dir: str = "data/index"):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "anime.index"))
        with open(index_dir / "id_mapping.pkl", "rb") as f:
            self.mal_ids = pickle.load(f)

        doc_mal_ids, doc_texts = load_documents(self.cfg["data"]["documents_path"])
        self.doc_lookup = dict(zip(doc_mal_ids, doc_texts))

        # Metadata untuk UI (poster + info dasar) -- image_url, status, episodes dari
        # dataset Kaggle (instan, tanpa panggilan API), BUKAN dari Jikan
        import pandas as pd
        df_meta = pd.read_csv(self.cfg["data"]["filtered_path"])
        meta_cols = [c for c in ["title", "title_english", "image_url", "score", "status", "episodes", "genres", "themes"] if c in df_meta.columns]
        self.metadata = df_meta.set_index("mal_id")[meta_cols].to_dict("index")

        print(f"[OK] Index dimuat: {self.index.ntotal} vektor, metadata: {len(self.metadata)} entri")

    def load_llm(self, model_name: str | None = None, quantize: bool = False):
        """
        Load model generator. `quantize=True` -> load 4-bit via BitsAndBytesConfig
        (butuh package bitsandbytes), berguna untuk Kaggle GPU dengan VRAM terbatas.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
        import torch

        model_name = model_name or self.cfg["llm"]["model_name"]
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model_kwargs = {"dtype": torch.float16 if self.device == "cuda" else torch.float32}
        if quantize:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs.pop("dtype", None)

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto" if self.device == "cuda" else None,
            **model_kwargs,
        )
        self.tokenizer = tokenizer
        self.llm = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=400,
            return_full_text=False,  # hanya kembalikan teks hasil generate, bukan prompt+jawaban
        )

        # Token terminator eksplisit -- WAJIB untuk model instruction-tuned seperti Llama-3.x,
        # kalau tidak diset, model bisa melanjutkan generate dengan giliran percakapan palsu
        # (mengarang pertanyaan+jawaban lanjutan sendiri).
        terminators = [tokenizer.eos_token_id]
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            terminators.append(eot_id)
        self.terminators = terminators
        self.llm_backend = "transformers"

        print(f"[OK] LLM dimuat: {model_name} (quantize={quantize})")

    def load_llm_gguf(
        self,
        repo_id: str = "bartowski/Llama-3.2-3B-Instruct-GGUF",
        filename: str = "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
        n_ctx: int = 4096,
    ):
        """
        Jalur load model KHUSUS untuk deployment CPU (HF Spaces gratis).

        `load_llm()` (bitsandbytes 4-bit) HANYA bisa jalan di GPU -- tidak
        kompatibel dengan HF Spaces gratis yang defaultnya CPU-only. Method
        ini memakai GGUF (llama-cpp-python), format kuantisasi yang memang
        didesain untuk CPU. Model diunduh sekali dari Hugging Face Hub,
        di-cache otomatis oleh huggingface_hub untuk run berikutnya.

        llama.cpp menangani chat template secara otomatis lewat metadata
        GGUF (tidak perlu tokenizer.apply_chat_template manual seperti di
        load_llm()).
        """
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama
        import os as _os

        model_path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=_os.cpu_count(),
            verbose=False,
        )
        self.llm_backend = "gguf"
        print(f"[OK] LLM (GGUF, CPU) dimuat: {repo_id}/{filename}")

    def enrich(self, mal_ids: list[int], use_cache: bool = True, max_retries: int = 3) -> dict:
        """
        Ambil data enrichment (Jikan API) untuk daftar mal_id hasil retrieval.
        Dipakai Kondisi C -- lihat src/enrichment.py untuk rate limiting & cache.
        max_retries lebih rendah (mis. 1) berguna untuk app.py (live chat) supaya
        latensi tidak membengkak kalau Jikan sedang bermasalah.
        Lazy import supaya `requests` tidak wajib ter-install kalau hanya mau
        pakai Kondisi A/B.
        """
        if self._jikan_client is None:
            from enrichment import JikanClient
            self._jikan_client = JikanClient(self.config_path)
        return self._jikan_client.enrich_batch(mal_ids, use_cache=use_cache, max_retries=max_retries)

    # Genre resmi di dataset -- dipakai untuk pre-filter eksplisit saat query
    # menyebut genre secara langsung (mis. "anime action", "anime comedy").
    GENRE_KEYWORDS = [
        "action", "adventure", "avant garde", "award winning", "boys love", "comedy",
        "drama", "ecchi", "fantasy", "girls love", "gourmet", "horror", "mystery",
        "romance", "sci-fi", "slice of life", "sports", "supernatural", "suspense",
    ]
    TITLE_PATTERN = re.compile(
        r"(?:suka|mirip dengan|mirip seperti|seperti|serupa dengan)\s+([^,\.\?!]+)",
        re.IGNORECASE,
    )

    def _detect_genre_filter(self, query: str) -> list[str]:
        q = query.lower()
        return [g for g in self.GENRE_KEYWORDS if g in q]

    def _find_anchor_mal_id(self, query: str):
        """
        Deteksi pola "suka X" / "mirip dengan X" lalu cari X di metadata
        (exact match dulu, abaikan titik/kapitalisasi; fallback substring).
        Mengembalikan mal_id anchor kalau ketemu, None kalau tidak.
        """
        m = self.TITLE_PATTERN.search(query)
        if not m:
            return None
        candidate = m.group(1).strip().rstrip(".").lower()
        if len(candidate) < 3:
            return None

        substring_match = None
        for mal_id, meta in self.metadata.items():
            title = str(meta.get("title") or "").rstrip(".").lower()
            title_en = str(meta.get("title_english") or "").rstrip(".").lower()
            if candidate == title or (title_en and candidate == title_en):
                return mal_id
            if substring_match is None and (candidate in title or (title_en and candidate in title_en)):
                substring_match = mal_id
        return substring_match

    def retrieve(self, query: str, k: int = 5, use_rerank: bool = False):
        """
        Retrieval top-k dengan dua peningkatan (hybrid retrieval):
          1. Anchor anime: kalau query menyebut "suka X"/"mirip dengan X" dan X
             ditemukan di dataset, similarity dihitung dari TEKS ANIME ITU SENDIRI
             (bukan kalimat percakapan user) -- lebih akurat untuk query similaritas.
          2. Filter genre eksplisit: kalau query menyebut genre langsung, hasil
             disaring berdasarkan genre asli di metadata sebelum diranking --
             mengatasi kelemahan semantic search murni untuk query kategori luas
             (lihat temuan Minggu 4).

        `use_rerank=True` (dipakai app.py, TIDAK dipakai skrip evaluasi Minggu 4/7
        supaya hasil lama tetap reproducible): ambil kandidat lebih banyak, lalu
        urutkan ulang berdasarkan kombinasi skor similarity + skor MAL, supaya
        anime usang/nyaris tak berperingkat tidak mendominasi rekomendasi.

        Dipakai oleh Kondisi B & C, dilewati oleh Kondisi A.
        """
        anchor_mal_id = self._find_anchor_mal_id(query)
        detected_genres = self._detect_genre_filter(query)

        if anchor_mal_id is not None and anchor_mal_id in self.doc_lookup:
            embed_input = self.doc_lookup[anchor_mal_id]
        else:
            embed_input = query

        search_k = k + (1 if anchor_mal_id is not None else 0)
        if detected_genres:
            search_k = max(search_k, 50)  # ambil kandidat lebih banyak dulu, baru difilter genre
        if use_rerank:
            search_k = max(search_k, 40)  # kandidat lebih banyak supaya re-ranking punya pilihan

        q_emb = self.embed_model.encode([embed_input], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, search_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            mal_id = self.mal_ids[idx]
            if mal_id == anchor_mal_id:
                continue  # jangan rekomendasikan anime yang sama dengan yang disebut user

            meta = self.metadata.get(mal_id, {})
            if detected_genres:
                doc_genres = str(meta.get("genres", "")).lower()
                if not any(g in doc_genres for g in detected_genres):
                    continue

            results.append({
                "mal_id": mal_id,
                "score": float(score),
                "text": self.doc_lookup.get(mal_id, ""),
                "title": meta.get("title"),
                "title_english": meta.get("title_english"),
                "image_url": meta.get("image_url"),
                "genres": meta.get("genres"),
                "mal_score": meta.get("score"),
            })
            if not use_rerank and len(results) >= k:
                break

        if use_rerank:
            results = self._rerank(results)

        return results[:k]

    def _rerank(self, candidates: list) -> list:
        """
        Urutkan ulang kandidat berdasarkan kombinasi:
          - kemiripan semantik (skor FAISS, sudah 0-1 karena embedding dinormalisasi)
          - skor MAL dinormalisasi (score/10), default netral 0.3 kalau skor kosong/NaN
        Bobot bisa diatur di configs/config.yaml (retrieval.rerank).
        Data mentah TIDAK diubah -- ini murni pengurutan ulang hasil retrieval yang
        sudah ada, bukan filtering dataset (lihat diskusi strategi dataset sebelumnya).
        """
        rr_cfg = self.cfg.get("retrieval", {}).get("rerank", {})
        w_semantic = rr_cfg.get("weight_semantic", 0.6)
        w_score = rr_cfg.get("weight_score", 0.4)
        default_norm = rr_cfg.get("missing_score_default", 0.3)

        def normalize_score(mal_score):
            try:
                if mal_score is None:
                    return default_norm
                val = float(mal_score)
                if val != val:  # NaN check tanpa perlu import math
                    return default_norm
                return val / 10
            except (TypeError, ValueError):
                return default_norm

        def combined(c):
            return w_semantic * c["score"] + w_score * normalize_score(c.get("mal_score"))

        return sorted(candidates, key=combined, reverse=True)

    def build_context(self, retrieved_docs, enrichment_data: dict | None = None) -> str:
        """
        enrichment_data: {mal_id: {"trailer": ..., "status_terkini": ...}, ...}
        Dipakai Kondisi C. None/kosong berarti Kondisi B (RAG tanpa enrichment).
        """
        if not retrieved_docs:
            return "(tidak ada konteks -- lihat Kondisi A)"

        blocks = []
        for doc in retrieved_docs:
            block = doc["text"]
            if enrichment_data and doc["mal_id"] in enrichment_data:
                extra = enrichment_data[doc["mal_id"]]
                block += f"\n[Info terkini] {extra}"
            blocks.append(block)
        return "\n\n---\n\n".join(blocks)

    def generate(
        self,
        query: str,
        use_retrieval: bool = True,
        use_enrichment: bool = False,
        k: int | None = None,
        enrichment_data: dict | None = None,
        pre_retrieved: list | None = None,
    ) -> dict:
        """
        Satu fungsi untuk ketiga kondisi ablation:
          Kondisi A: use_retrieval=False
          Kondisi B: use_retrieval=True,  use_enrichment=False
          Kondisi C: use_retrieval=True,  use_enrichment=True (enrichment_data wajib diisi)

        `pre_retrieved`: opsional -- kalau diisi, PAKAI daftar ini alih-alih memanggil
        self.retrieve() lagi. Dipakai app.py (deployment) untuk menyuntikkan hasil
        candidate selection sendiri (exclude anchor + re-ranking skor MAL) tanpa
        mengubah perilaku retrieve() inti yang dipakai run_experiment.py/evaluate_retrieval.py
        (supaya hasil evaluasi Minggu 4/7 tetap reproducible apa adanya).
        """
        k = k or self.cfg["retrieval"]["top_k_final"] or 5

        # Lapisan 3: tolak LANGSUNG kalau query eksplisit -- tidak perlu panggil
        # retrieval maupun LLM sama sekali (lebih cepat, jaring pengaman kalau
        # system prompt/Lapisan 2 gagal menolak dengan benar).
        refusal = guard_query(query)
        if refusal is not None:
            return {
                "query": query,
                "condition": "BLOCKED",
                "retrieved_mal_ids": [],
                "answer": refusal,
            }

        if not use_retrieval:
            retrieved = []
        elif pre_retrieved is not None:
            retrieved = pre_retrieved
        else:
            retrieved = self.retrieve(query, k=k)
        context = self.build_context(retrieved, enrichment_data if use_enrichment else None)

        user_content = PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if self.llm is None:
            raise RuntimeError("Panggil load_llm() atau load_llm_gguf() dulu sebelum generate().")

        if self.llm_backend == "gguf":
            # llama-cpp-python menangani chat template otomatis lewat metadata GGUF
            response = self.llm.create_chat_completion(messages=messages, max_tokens=400)
            answer = response["choices"][0]["message"]["content"].strip()
        else:
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            output = self.llm(
                prompt,
                eos_token_id=self.terminators,
                pad_token_id=self.tokenizer.eos_token_id,
            )[0]["generated_text"]
            answer = output.strip()  # return_full_text=False -> ini murni jawaban

        return {
            "query": query,
            "condition": "A" if not use_retrieval else ("C" if use_enrichment else "B"),
            "retrieved_mal_ids": [d["mal_id"] for d in retrieved],
            "answer": answer,
        }


if __name__ == "__main__":
    pipe = RagPipeline()
    pipe.load_index()
    pipe.load_llm()

    # Contoh menjalankan ketiga kondisi untuk satu query yang sama
    query = "Rekomendasikan anime action dengan tema luar angkasa"

    # Kondisi C butuh enrichment_data sungguhan: retrieval dulu untuk dapat mal_id,
    # baru panggil Jikan API (rate limit ~1 req/detik, jadi ini bagian paling lambat)
    retrieved_for_enrichment = pipe.retrieve(query, k=pipe.cfg["retrieval"]["top_k_final"])
    enrichment_data = pipe.enrich([d["mal_id"] for d in retrieved_for_enrichment])

    for cond_name, kwargs in [
        ("A", {"use_retrieval": False}),
        ("B", {"use_retrieval": True, "use_enrichment": False}),
        ("C", {"use_retrieval": True, "use_enrichment": True, "enrichment_data": enrichment_data}),
    ]:
        result = pipe.generate(query, **kwargs)
        print(f"\n=== Kondisi {cond_name} ===")
        print(result["answer"])
