"""
Minggu 7 (Tahap 3): Analisis Hasil Evaluasi

Menggabungkan tests/judge_scores.jsonl (LLM-as-a-Judge) menjadi:
  - Rata-rata skor kualitas jawaban per kondisi (A/B/C) per kategori
  - Uji Wilcoxon signed-rank berpasangan per query_id: A vs B, B vs C
    (Bagian 6.4 dokumen rincian project)
  - Refusal rate & ketepatan penolakan untuk kategori out_of_scope/adversarial

Tidak butuh GPU/internet -- murni analisis statistik lokal (pip install scipy).

Output: tests/final_evaluation_report.json
"""

import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import wilcoxon

JUDGE_SCORES_PATH = Path("tests/judge_scores.jsonl")
OUT_PATH = Path("tests/final_evaluation_report.json")
REFUSAL_CATEGORIES = {"out_of_scope", "adversarial_eksplisit"}


def load_judge_scores():
    with open(JUDGE_SCORES_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def combined_score(judge: dict):
    """Skor gabungan (rata-rata 3 kriteria) untuk kategori normal."""
    keys = ["relevansi", "akurasi_faktual", "koherensi"]
    values = [judge.get(k) for k in keys if isinstance(judge.get(k), (int, float))]
    return sum(values) / len(values) if values else None


def analyze_refusal(records):
    by_condition = defaultdict(list)
    for r in records:
        if r["category"] not in REFUSAL_CATEGORIES:
            continue
        judge = r.get("judge", {})
        menolak = judge.get("menolak")
        ketepatan = judge.get("ketepatan_penolakan")
        if r.get("blocked"):
            # Lapisan 3 (guardrails.py) langsung blokir sebelum LLM dipanggil -- pasti benar menolak
            menolak, ketepatan = True, 5
        by_condition[r["condition"]].append((menolak, ketepatan))

    summary = {}
    for cond, pairs in by_condition.items():
        n = len(pairs)
        n_refused = sum(1 for m, _ in pairs if m is True)
        valid_ketepatan = [k for _, k in pairs if isinstance(k, (int, float))]
        summary[cond] = {
            "n_query": n,
            "refusal_rate": round(n_refused / n, 4) if n else None,
            "avg_ketepatan_penolakan": round(sum(valid_ketepatan) / len(valid_ketepatan), 4) if valid_ketepatan else None,
        }
    return summary


def analyze_quality(records):
    by_cond_cat = defaultdict(list)
    scores_by_query = defaultdict(dict)  # {query_id: {condition: score}}

    for r in records:
        if r["category"] in REFUSAL_CATEGORIES:
            continue
        score = combined_score(r.get("judge", {}))
        if score is None:
            continue
        by_cond_cat[(r["condition"], r["category"])].append(score)
        scores_by_query[r["query_id"]][r["condition"]] = score

    summary = {}
    for (cond, cat), scores in by_cond_cat.items():
        summary.setdefault(cond, {})[cat] = {
            "n_query": len(scores),
            "mean_score": round(sum(scores) / len(scores), 4),
        }
    return summary, scores_by_query


def run_wilcoxon(scores_by_query):
    def paired(cond1, cond2):
        return [
            (v[cond1], v[cond2])
            for v in scores_by_query.values()
            if cond1 in v and cond2 in v
        ]

    results = {}
    for cond1, cond2 in [("A", "B"), ("B", "C")]:
        pairs = paired(cond1, cond2)
        label = f"{cond1}_vs_{cond2}"
        if len(pairs) < 2:
            results[label] = {"error": "data tidak cukup untuk uji statistik"}
            continue
        x = [p[0] for p in pairs]
        y = [p[1] for p in pairs]
        try:
            stat, p_value = wilcoxon(x, y)
            results[label] = {
                "n_pairs": len(pairs),
                f"mean_{cond1}": round(sum(x) / len(x), 4),
                f"mean_{cond2}": round(sum(y) / len(y), 4),
                "wilcoxon_statistic": round(float(stat), 4),
                "p_value": round(float(p_value), 6),
                "signifikan_p<0.05": bool(p_value < 0.05),
            }
        except ValueError as e:
            results[label] = {"error": str(e)}
    return results


def main():
    records = load_judge_scores()

    quality_summary, scores_by_query = analyze_quality(records)
    wilcoxon_results = run_wilcoxon(scores_by_query)
    refusal_summary = analyze_refusal(records)

    report = {
        "kualitas_jawaban_per_kondisi_kategori": quality_summary,
        "uji_wilcoxon_signed_rank": wilcoxon_results,
        "refusal_rate_per_kondisi": refusal_summary,
    }

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[OK] Laporan lengkap: {OUT_PATH}")


if __name__ == "__main__":
    main()
