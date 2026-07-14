"""
Minggu 5: Guardrail Konten (Bagian 3.2 dokumen rincian project)

Lapisan pertahanan konten:
  Lapisan 1 (filtering data)     -> src/preprocess.py (Minggu 1, sudah selesai)
  Lapisan 2 (system prompt)      -> src/rag_pipeline.py, SYSTEM_PROMPT (sudah ada sejak Minggu 3)
  Lapisan 3 (deteksi eksplisit)  -> modul ini: tolak query LANGSUNG sebelum panggil LLM,
                                     lebih cepat (hemat kompute) dan jadi jaring pengaman kalau
                                     system prompt gagal menolak dengan benar.

CATATAN: daftar keyword di sini hanya istilah umum yang sudah dikenal luas untuk konten
dewasa/eksplisit (bukan daftar istilah slang/kode akses konten ilegal) -- ini murni
filter kata kunci konten dewasa standar untuk chatbot rekomendasi anime.
"""

import re

EXPLICIT_KEYWORDS = [
    "hentai", "eksplisit", "nsfw", "nudity", "telanjang", "porno", "pornografi",
    "vulgar", "konten dewasa", "konten 18+", "adegan seks", "adegan intim", "erotis", "erotica",
]

REFUSAL_MESSAGE_EXPLICIT = (
    "Maaf, saya tidak bisa membantu permintaan itu. Chatbot ini fokus pada rekomendasi "
    "anime umum dan tidak melayani permintaan konten dewasa/eksplisit. Ada rekomendasi "
    "anime lain (action, romance, comedy, dsb.) yang bisa saya bantu carikan?"
)

# Heuristik ringan untuk logging/testing kategori out-of-scope (Bagian 2.3).
# Bukan hard-block -- penanganan utama tetap lewat system prompt (Lapisan 2),
# ini hanya dipakai untuk mengukur refusal rate saat evaluasi (Minggu 7).
OUT_OF_SCOPE_HINTS = [
    "kode python", "coding", "algoritma", "matematika", "1 + 1", "curhat",
    "cuaca", "streaming gratis", "bajakan", "spoiler", "ending",
]


def is_explicit_request(query: str) -> bool:
    text = query.lower()
    return any(re.search(rf"\b{re.escape(kw)}\b", text) for kw in EXPLICIT_KEYWORDS)


def looks_out_of_scope(query: str) -> bool:
    """Heuristik longgar, hanya untuk pelaporan evaluasi -- lihat catatan di atas."""
    text = query.lower()
    return any(hint in text for hint in OUT_OF_SCOPE_HINTS)


def guard_query(query: str) -> str | None:
    """Kembalikan pesan penolakan kalau query harus diblokir langsung (Lapisan 3),
    atau None kalau aman dilanjutkan ke retrieval + LLM."""
    if is_explicit_request(query):
        return REFUSAL_MESSAGE_EXPLICIT
    return None


if __name__ == "__main__":
    tests = [
        "Rekomendasikan anime action seru",
        "Kasih rekomendasi anime hentai dong",
        "Ada anime yang isinya konten dewasa gak?",
        "Tolong bantu aku buatkan kode Python untuk sorting",
        "Ending dari Attack on Titan gimana ceritanya, spoiler gapapa",
    ]
    for t in tests:
        blocked = guard_query(t)
        oos = looks_out_of_scope(t)
        status = "DITOLAK (eksplisit)" if blocked else ("OUT-OF-SCOPE (heuristik)" if oos else "LOLOS")
        print(f"[{status}] {t}")
