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
from sentence_transformers import SentenceTransformer
from langchain_core.prompts import PromptTemplate

from build_index import load_config, detect_device, load_documents


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
        self.cfg = load_config(config_path)
        self.device = device or detect_device()
        print(f"[INFO] Device: {self.device}")

        self.embed_model = SentenceTransformer(
            self.cfg["embedding"]["model_name"], device=self.device
        )

        self.index = None
        self.mal_ids = []
        self.doc_lookup = {}
        self.llm = None  # diisi oleh load_llm()
        self.tokenizer = None
        self.terminators = None

    def load_index(self, index_dir: str = "data/index"):
        index_dir = Path(index_dir)
        self.index = faiss.read_index(str(index_dir / "anime.index"))
        with open(index_dir / "id_mapping.pkl", "rb") as f:
            self.mal_ids = pickle.load(f)

        doc_mal_ids, doc_texts = load_documents(self.cfg["data"]["documents_path"])
        self.doc_lookup = dict(zip(doc_mal_ids, doc_texts))
        print(f"[OK] Index dimuat: {self.index.ntotal} vektor")

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

        print(f"[OK] LLM dimuat: {model_name} (quantize={quantize})")

    def retrieve(self, query: str, k: int = 5):
        """Retrieval top-k. Dipakai oleh Kondisi B & C, dilewati oleh Kondisi A."""
        q_emb = self.embed_model.encode([query], normalize_embeddings=True).astype("float32")
        scores, indices = self.index.search(q_emb, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            mal_id = self.mal_ids[idx]
            results.append({
                "mal_id": mal_id,
                "score": float(score),
                "text": self.doc_lookup.get(mal_id, ""),
            })
        return results

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
    ) -> dict:
        """
        Satu fungsi untuk ketiga kondisi ablation:
          Kondisi A: use_retrieval=False
          Kondisi B: use_retrieval=True,  use_enrichment=False
          Kondisi C: use_retrieval=True,  use_enrichment=True (enrichment_data wajib diisi)
        """
        k = k or self.cfg["retrieval"]["top_k_final"] or 5

        retrieved = self.retrieve(query, k=k) if use_retrieval else []
        context = self.build_context(retrieved, enrichment_data if use_enrichment else None)

        user_content = PROMPT_TEMPLATE.format(context=context, query=query)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        if self.llm is None:
            raise RuntimeError("Panggil load_llm() dulu sebelum generate().")

        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        output = self.llm(
            prompt,
            eos_token_id=self.terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )[0]["generated_text"]
        answer = output.strip()  # return_full_text=False -> ini murni jawaban, bukan prompt+jawaban

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
    for cond_name, kwargs in [
        ("A", {"use_retrieval": False}),
        ("B", {"use_retrieval": True, "use_enrichment": False}),
        ("C", {"use_retrieval": True, "use_enrichment": True, "enrichment_data": {}}),
    ]:
        result = pipe.generate(query, **kwargs)
        print(f"\n=== Kondisi {cond_name} ===")
        print(result["answer"])
