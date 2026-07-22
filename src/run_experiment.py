"""
Minggu 7 (Tahap 1, versi 2 -- setelah penyempurnaan sistem): Menjalankan 99
query test set pada Kondisi A/B/C

Butuh GPU (Kaggle) karena memanggil LLM 99 x 3 = 297 kali. Kategori
out_of_scope/adversarial_eksplisit TETAP dijalankan (bukan dilewati) --
justru itu yang diukur: apakah sistem menolak dengan benar (refusal rate).

CATATAN PERUBAHAN vs run pertama: sejak evaluasi Minggu 7 awal, sistem sudah
disempurnakan (system prompt anti-halusinasi + format terstruktur, re-ranking
skor MAL, exclusion anchor anime, konteks percakapan). Run ini memakai
use_rerank=True dan pre_retrieved supaya evaluasi benar-benar mencerminkan
perilaku app.py yang di-deploy, bukan versi lama.

Kondisi C tetap disertakan (bukan dihapus) untuk mempertahankan struktur
ablation study yang sama dengan run pertama -- lihat keputusan menghilangkan
enrichment dari DEPLOYMENT (bukan dari metodologi evaluasi/riset) di README.

Output: tests/experiment_results.jsonl (satu baris per query x kondisi)
-- TIMPA hasil run pertama. Simpan salinan lama dulu kalau ingin membandingkan
(mis. rename ke experiment_results_v1.jsonl sebelum menjalankan ini).
"""

import json
from pathlib import Path

from rag_pipeline import RagPipeline

TEST_SET_PATH = Path("tests/test_set.jsonl")
OUT_PATH = Path("tests/experiment_results.jsonl")

CONDITIONS = [
    ("A", {"use_retrieval": False, "use_enrichment": False}),
    ("B", {"use_retrieval": True, "use_enrichment": False}),
    ("C", {"use_retrieval": True, "use_enrichment": True}),
]


def load_test_set():
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def query_text(q) -> str:
    return " ".join(q["query"]) if isinstance(q["query"], list) else q["query"]


def main():
    pipe = RagPipeline()
    pipe.load_index()
    pipe.load_llm(quantize=True)

    queries = load_test_set()
    k = pipe.cfg["retrieval"]["top_k_final"]
    total_calls = len(queries) * len(CONDITIONS)
    print(f"[INFO] {len(queries)} query x {len(CONDITIONS)} kondisi = {total_calls} pemanggilan generate()")

    n_written = 0
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for i, q in enumerate(queries, start=1):
            text = query_text(q)
            print(f"[{i}/{len(queries)}] {q['id']} ({q['category']})")

            # Retrieval sekali per query, SAMA PERSIS dengan yang dipakai app.py
            # (use_rerank=True) -- dipakai bareng Kondisi B & C via pre_retrieved.
            retrieved = pipe.retrieve(text, k=k, use_rerank=True)
            mal_ids = [d["mal_id"] for d in retrieved]
            enrichment_data = None

            for cond_name, kwargs in CONDITIONS:
                call_kwargs = dict(kwargs)
                if cond_name == "C" and mal_ids:
                    if enrichment_data is None:
                        enrichment_data = pipe.enrich(mal_ids)
                    call_kwargs["enrichment_data"] = enrichment_data

                result = pipe.generate(text, k=k, pre_retrieved=retrieved, **call_kwargs)
                record = {
                    "query_id": q["id"],
                    "category": q["category"],
                    "condition": cond_name,
                    "query": text,
                    "answer": result["answer"],
                    "retrieved_mal_ids": result["retrieved_mal_ids"],
                    "blocked": result["condition"] == "BLOCKED",
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                n_written += 1

    print(f"\n[OK] {n_written} hasil tersimpan: {OUT_PATH}")


if __name__ == "__main__":
    main()
