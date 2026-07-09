"""
Minggu 2: Embedding + FAISS Indexing

Modul ini portable (CPU maupun GPU) -- device dideteksi otomatis atau
dipaksa lewat argumen. Dijalankan lewat:
  - Laptop (CPU)              : python3 src/build_index.py
  - Kaggle Notebook (GPU)     : lihat notebooks/02_build_index_kaggle.ipynb

Input : data/processed/anime_documents.jsonl (hasil src/preprocess.py)
Output: data/index/anime.index      (FAISS index)
        data/index/id_mapping.pkl   (urutan index -> mal_id, untuk enrichment)
"""

import json
import pickle
from pathlib import Path

import faiss
import yaml
from sentence_transformers import SentenceTransformer


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_documents(path: str):
    mal_ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            mal_ids.append(d["mal_id"])
            texts.append(d["text"])
    return mal_ids, texts


def build_index(
    documents_path: str,
    model_name: str,
    out_dir: str,
    device: str = "cpu",
    batch_size: int = 64,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mal_ids, texts = load_documents(documents_path)
    print(f"[INFO] Memuat {len(texts)} dokumen untuk di-embed (device={device})")

    model = SentenceTransformer(model_name, device=device)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # supaya inner product == cosine similarity
    ).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    faiss.write_index(index, str(out_dir / "anime.index"))
    with open(out_dir / "id_mapping.pkl", "wb") as f:
        pickle.dump(mal_ids, f)

    print(f"[OK] Index: {out_dir/'anime.index'} ({index.ntotal} vektor, dim={dim})")
    print(f"[OK] Mapping index->mal_id: {out_dir/'id_mapping.pkl'}")
    return index, mal_ids


def detect_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


if __name__ == "__main__":
    cfg = load_config()
    device = detect_device()
    print(f"[INFO] Device terdeteksi: {device}")
    build_index(
        documents_path=cfg["data"]["documents_path"],
        model_name=cfg["embedding"]["model_name"],
        out_dir="data/index",
        device=device,
    )
