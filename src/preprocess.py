"""
Preprocessing dataset anime sesuai Dokumen Rincian Project (Bagian 3 & 4.2).

Langkah:
1. Load anime_dataset.csv
2. Verifikasi kolom wajib (terutama mal_id)
3. Filter konten: buang rating == 'Rx' dan genre/themes yang memuat 'Hentai'
   (Ecchi TIDAK difilter -- lihat Bagian 3.1 dokumen rincian project)
4. Bangun dokumen terstruktur per anime (template chunking, Bagian 4.2)
5. Simpan hasil ke data/processed/
"""

import pandas as pd
import json
import re
from pathlib import Path

RAW_PATH = Path("data/raw/anime_dataset.csv")
OUT_CSV = Path("data/processed/anime_filtered.csv")
OUT_DOCS = Path("data/processed/anime_documents.jsonl")
REPORT_PATH = Path("data/processed/filtering_report.json")

REQUIRED_COLUMNS = [
    "mal_id", "title", "title_english", "title_japanese", "type", "source",
    "episodes", "status", "airing", "aired_from", "aired_to", "duration",
    "rating", "score", "scored_by", "rank", "popularity", "members",
    "favorites", "season", "year", "studios", "producers", "licensors",
    "genres", "themes", "demographics", "synopsis", "image_url",
]


def load_and_verify(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan di {path}. "
            f"Unduh anime_dataset.csv dari Kaggle dan letakkan di data/raw/."
        )
    df = pd.read_csv(path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"[PERINGATAN] Kolom berikut tidak ditemukan di dataset: {missing}")
        print("Cek kembali skema dataset -- beberapa langkah di bawah mungkin perlu disesuaikan.")

    if "mal_id" in df.columns:
        n_missing_id = df["mal_id"].isna().sum()
        n_dup_id = df["mal_id"].duplicated().sum()
        print(f"[INFO] mal_id kosong: {n_missing_id} baris | mal_id duplikat: {n_dup_id} baris")
    else:
        raise ValueError(
            "Kolom mal_id tidak ada. Arsitektur enrichment (Bagian 4.1) bergantung "
            "pada mal_id untuk mencocokkan hasil retrieval dengan Jikan API -- "
            "wajib diselesaikan sebelum lanjut ke tahap berikutnya."
        )

    return df


def contains_tag(value: str, tag: str) -> bool:
    if pd.isna(value):
        return False
    parts = [p.strip().lower() for p in re.split(r"\|", str(value))]
    return tag.lower() in parts


def filter_content(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Filtering dua kategori (sesuai kesepakatan konsultasi):

    A. Batasan konten (Bagian 3.1, diperluas):
       - rating memuat 'Rx' (format aktual di data: 'Rx - Hentai') -> DROP (hard filter)
       - genres/themes memuat 'Hentai'                             -> DROP (pengecekan redundan)
       - genres/themes memuat 'Erotica'                            -> DROP (konten seksual eksplisit,
         berbeda dari Ecchi yang sekadar fanservice ringan -- kebijakan konservatif)
       - genres/themes memuat 'Ecchi'                              -> TIDAK di-drop

    B. Data hygiene (bukan filter kualitas/skor -- lihat pembahasan strategi dataset):
       - duplikat mal_id (baris identik)                           -> DROP, simpan kemunculan pertama
       - status 'Not yet aired' DENGAN synopsis kosong              -> DROP
         (bukan seluruh anime 'Not yet aired' -- hanya yang informasinya benar-benar minim)
       - title, synopsis, DAN genres kosong bersamaan               -> DROP (dokumen tidak bisa di-embed bermakna)

    Skor (score) dan popularitas (members/scored_by) TIDAK dipakai sebagai filter penghapusan.
    """
    n_total = len(df)

    # --- A. Batasan konten ---
    mask_rx = df["rating"].astype(str).str.lower().str.contains("rx", na=False)
    mask_hentai_genre = df["genres"].apply(lambda v: contains_tag(v, "hentai")) if "genres" in df.columns else pd.Series(False, index=df.index)
    mask_hentai_theme = df["themes"].apply(lambda v: contains_tag(v, "hentai")) if "themes" in df.columns else pd.Series(False, index=df.index)
    mask_erotica_genre = df["genres"].apply(lambda v: contains_tag(v, "erotica")) if "genres" in df.columns else pd.Series(False, index=df.index)
    mask_erotica_theme = df["themes"].apply(lambda v: contains_tag(v, "erotica")) if "themes" in df.columns else pd.Series(False, index=df.index)
    mask_content_block = mask_rx | mask_hentai_genre | mask_hentai_theme | mask_erotica_genre | mask_erotica_theme

    # --- B. Data hygiene ---
    mask_dup = df["mal_id"].duplicated(keep="first")
    mask_unaired_empty = (df["status"] == "Not yet aired") & (df["synopsis"].isna())
    mask_all_empty = df["title"].isna() & df["synopsis"].isna() & df["genres"].isna()

    mask_drop = mask_content_block | mask_dup | mask_unaired_empty | mask_all_empty
    df_filtered = df[~mask_drop].copy()

    report = {
        "total_sebelum_filter": int(n_total),
        "dihapus_konten__rating_rx_hentai_atau_erotica": int(mask_content_block.sum()),
        "dihapus_hygiene__duplikat_mal_id": int(mask_dup.sum()),
        "dihapus_hygiene__belum_tayang_info_minim": int(mask_unaired_empty.sum()),
        "dihapus_hygiene__semua_metadata_inti_kosong": int(mask_all_empty.sum()),
        "dihapus_total": int(mask_drop.sum()),
        "total_setelah_filter": int(len(df_filtered)),
        "catatan": (
            "Ecchi tidak difilter. Skor (score) dan popularitas (members/scored_by) "
            "TIDAK dipakai sebagai kriteria penghapusan -- lihat analisis strategi dataset."
        ),
    }
    return df_filtered, report


def truncate_synopsis(text, max_chars=1000):
    if pd.isna(text):
        return "Tidak tersedia"
    text = str(text)
    return text[:max_chars]


def clean_value(value, default="Tidak diketahui") -> str:
    """Ubah NaN/None jadi placeholder yang jelas, dan rapikan angka float (26.0 -> 26)."""
    if pd.isna(value):
        return default
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_document(row: pd.Series) -> str:
    """Template chunking terstruktur -- Bagian 4.2 dokumen rincian project."""
    year = clean_value(row.get("year"), default="")
    season = clean_value(row.get("season"), default="")
    season_year = f"{season} {year}".strip() or "Tidak diketahui"

    title = clean_value(row.get("title"))
    title_en = clean_value(row.get("title_english"), default="")
    if title_en and title_en.strip().lower() != title.strip().lower():
        title_line = f"Judul: {title} ({title_en})"
    else:
        title_line = f"Judul: {title}"

    return (
        f"{title_line}\n"
        f"Tipe: {clean_value(row.get('type'))} | Status: {clean_value(row.get('status'))} | "
        f"Episode: {clean_value(row.get('episodes'))} | Durasi: {clean_value(row.get('duration'))}\n"
        f"Genre: {clean_value(row.get('genres'))} | Tema: {clean_value(row.get('themes'))} | "
        f"Demografi: {clean_value(row.get('demographics'))}\n"
        f"Studio: {clean_value(row.get('studios'))} | Season: {season_year}\n"
        f"Rating: {clean_value(row.get('score'), default='Belum ada skor')}/10\n"
        f"Sinopsis: {truncate_synopsis(row.get('synopsis'))}"
    )


def main():
    df = load_and_verify(RAW_PATH)
    df_filtered, report = filter_content(df)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df_filtered.to_csv(OUT_CSV, index=False)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    with open(OUT_DOCS, "w", encoding="utf-8") as f:
        for _, row in df_filtered.iterrows():
            doc = {
                "mal_id": row["mal_id"],
                "text": build_document(row),
            }
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print("\n=== Laporan Filtering (Bagian 3.1) ===")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nDataset terfilter disimpan di: {OUT_CSV}")
    print(f"Dokumen siap-embedding disimpan di: {OUT_DOCS}")


if __name__ == "__main__":
    main()
