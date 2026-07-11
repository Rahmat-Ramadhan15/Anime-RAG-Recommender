"""
Minggu 4: Evaluasi Retrieval -- Recall@k & MRR

Tidak butuh GPU/Kaggle -- hanya perlu model embedding (ringan) + index FAISS
yang sudah dibangun di Minggu 2 (unduh dari Kaggle, taruh di data/index/).

Menguji k = 3, 5, 10 (configs/config.yaml -> retrieval.top_k_candidates)
pada query yang punya ground truth otomatis (filter_atribut, faktual,
multi_turn_refinement) dan semi-otomatis (similaritas_rekomendasi).
Kategori out_of_scope/adversarial dilewati di sini -- itu diukur lewat
refusal rate saat pipeline LLM sudah lengkap (Minggu 7), bukan Recall@k.

Output: tests/topk_evaluation_report.json
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import faiss
import yaml
from sentence_transformers import SentenceTransformer

TEST_SET_PATH = Path("tests/test_set.jsonl")
INDEX_DIR = Path("data/index")
REPORT_PATH = Path("tests/topk_evaluation_report.json")

SKIP_CATEGORIES = {"out_of_scope", "adversarial_eksplisit"}


def load_config(path="configs/config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_test_set():
    queries = []
    with open(TEST_SET_PATH, encoding="utf-8") as f:
        for line in f:
            q = json.loads(line)
            if q["category"] not in SKIP_CATEGORIES and q["ground_truth_mal_ids"]:
                queries.append(q)
    return queries


def recall_at_k(retrieved_ids, ground_truth_ids):
    """Recall@k = proporsi ground truth yang berhasil ditemukan dalam top-k. TIDAK PERNAH turun saat k naik."""
    if not ground_truth_ids:
        return None
    hit = len(set(retrieved_ids) & set(ground_truth_ids))
    return hit / len(ground_truth_ids)


def precision_at_k(retrieved_ids, ground_truth_ids):
    """Precision@k = proporsi hasil top-k yang valid/relevan. Metrik utama untuk kategori
    dengan ground truth besar (filter_atribut, similaritas, multi_turn) -- lihat catatan di README."""
    if not ground_truth_ids or not retrieved_ids:
        return None
    hit = len(set(retrieved_ids) & set(ground_truth_ids))
    return hit / len(retrieved_ids)


def reciprocal_rank(retrieved_ids, ground_truth_ids):
    gt_set = set(ground_truth_ids)
    for rank, mal_id in enumerate(retrieved_ids, start=1):
        if mal_id in gt_set:
            return 1.0 / rank
    return 0.0


def query_text(q):
    # multi_turn_refinement -> query berupa list giliran; gabungkan jadi satu string untuk embedding
    if isinstance(q["query"], list):
        return " ".join(q["query"])
    return q["query"]


def main():
    cfg = load_config()
    top_k_candidates = cfg["retrieval"]["top_k_candidates"]
    max_k = max(top_k_candidates)

    print("[INFO] Memuat model embedding & index...")
    model = SentenceTransformer(cfg["embedding"]["model_name"], device="cpu")
    index = faiss.read_index(str(INDEX_DIR / "anime.index"))
    with open(INDEX_DIR / "id_mapping.pkl", "rb") as f:
        mal_ids = pickle.load(f)

    queries = load_test_set()
    print(f"[INFO] {len(queries)} query akan dievaluasi (dari total test set, dikurangi out-of-scope)")

    # results[k][category] = list of (recall, rr)
    results = {k: defaultdict(list) for k in top_k_candidates}

    for q in queries:
        text = query_text(q)
        q_emb = model.encode([text], normalize_embeddings=True).astype("float32")
        _, indices = index.search(q_emb, max_k)
        retrieved_all = [mal_ids[i] for i in indices[0]]

        for k in top_k_candidates:
            retrieved_k = retrieved_all[:k]
            r = recall_at_k(retrieved_k, q["ground_truth_mal_ids"])
            p = precision_at_k(retrieved_k, q["ground_truth_mal_ids"])
            rr = reciprocal_rank(retrieved_k, q["ground_truth_mal_ids"])
            results[k][q["category"]].append((r, p, rr))
            results[k]["__overall__"].append((r, p, rr))

    # Ringkas jadi rata-rata per kategori per k
    summary = {}
    for k in top_k_candidates:
        summary[f"k={k}"] = {}
        for cat, triples in results[k].items():
            recalls = [r for r, _, _ in triples if r is not None]
            precisions = [p for _, p, _ in triples if p is not None]
            rrs = [rr for _, _, rr in triples]
            summary[f"k={k}"][cat] = {
                "n_query": len(triples),
                "mean_recall": round(sum(recalls) / len(recalls), 4) if recalls else None,
                "mean_precision": round(sum(precisions) / len(precisions), 4) if precisions else None,
                "mrr": round(sum(rrs) / len(rrs), 4) if rrs else None,
            }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Ringkasan Recall@k, Precision@k, & MRR ===")
    for k_label, cats in summary.items():
        overall = cats["__overall__"]
        print(f"{k_label}: Recall={overall['mean_recall']}, Precision={overall['mean_precision']}, MRR={overall['mrr']} (n={overall['n_query']})")
        for cat, stats in cats.items():
            if cat == "__overall__":
                continue
            print(f"    {cat:28s} Recall={stats['mean_recall']}, Precision={stats['mean_precision']}, MRR={stats['mrr']} (n={stats['n_query']})")

    print(f"\nLaporan lengkap: {REPORT_PATH}")
    print(
        "\n[CATATAN INTERPRETASI]\n"
        "- Kategori 'faktual' (1 jawaban benar): baca Recall@k & MRR sebagai metrik utama.\n"
        "- Kategori 'filter_atribut', 'similaritas_rekomendasi', 'multi_turn_refinement' (ratusan\n"
        "  jawaban valid per query): baca Precision@k sebagai metrik utama -- Recall@k akan SELALU\n"
        "  kecil secara struktural di kategori ini (k kecil dibagi ratusan kandidat valid), bukan\n"
        "  berarti sistem buruk. Precision@k menjawab pertanyaan yang relevan bagi user: 'dari k\n"
        "  rekomendasi yang ditampilkan, berapa yang benar-benar valid?'\n"
        "- Pilih k final dengan trade-off: Precision lebih tinggi vs konteks lebih panjang yang\n"
        "  harus diproses model 3B (memengaruhi latensi & fokus jawaban)."
    )


if __name__ == "__main__":
    main()
