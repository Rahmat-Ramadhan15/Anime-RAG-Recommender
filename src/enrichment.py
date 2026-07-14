"""
Minggu 5: Enrichment via Jikan API

Rate limit Jikan: 3 request/detik DAN 60 request/menit. Batasan yang lebih
ketat adalah 60/menit (~1 request/detik rata-rata), jadi limiter di sini
memakai interval minimum berdasarkan batas yang lebih ketat tsb.

Reproducibility (Bagian 6.6 dokumen rincian project): setiap respons API
disimpan sebagai snapshot JSON + timestamp di data/api_cache/, terpisah dari
sistem live, supaya hasil yang dilaporkan di skripsi tidak berubah meski
data Jikan berubah di kemudian hari.

Catatan arsitektur (lihat diskusi strategi dataset): poster TIDAK diambil
dari sini -- dataset Kaggle sudah punya image_url. Jikan API di sini hanya
untuk data yang benar-benar butuh live update: status tayang, episode
terkini, dan trailer.
"""

import json
import re
import time
from pathlib import Path
from datetime import datetime, timezone

import requests
import yaml


def load_config(path: str = "configs/config.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


class JikanClient:
    def __init__(self, config_path: str = "configs/config.yaml"):
        cfg = load_config(config_path)
        self.base_url = cfg["enrichment"]["jikan_base_url"]
        self.cache_dir = Path(cfg["enrichment"]["cache_dir"])
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Batasan lebih ketat menentukan interval: 60/menit == 1 request/detik,
        # lebih ketat daripada batas 3/detik -- jadi ini yang dipakai.
        self._min_interval = 60.0 / cfg["enrichment"]["rate_limit_per_minute"]
        self._last_request_time = 0.0

    def _wait_for_rate_limit(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)

    def _cache_path(self, mal_id: int) -> Path:
        return self.cache_dir / f"{mal_id}.json"

    def get_anime(self, mal_id: int, use_cache: bool = True, max_retries: int = 3) -> dict | None:
        """
        Ambil detail anime dari cache lokal kalau ada, kalau tidak panggil Jikan API.

        Memakai endpoint dasar (/anime/{id}), BUKAN /anime/{id}/full -- endpoint
        /full membawa data relasi/theme song/streaming links yang tidak kita
        butuhkan, membuat respons lebih berat dan lebih rentan timeout di API
        gratis tanpa SLA seperti Jikan. Field yang kita perlukan (status,
        episodes, trailer, score) sudah tersedia di endpoint dasar.

        Retry dengan backoff untuk error 5xx sementara (502/503/504) -- umum
        terjadi di Jikan karena beban server publik, bukan indikasi bug.
        """
        cache_path = self._cache_path(mal_id)
        if use_cache and cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                return json.load(f)["data"]

        url = f"{self.base_url}/anime/{mal_id}"
        for attempt in range(1, max_retries + 1):
            self._wait_for_rate_limit()
            try:
                resp = requests.get(url, timeout=10)
                self._last_request_time = time.time()
                if resp.status_code == 404:
                    return None
                if resp.status_code in (502, 503, 504):
                    print(f"[WARN] Jikan API {resp.status_code} untuk mal_id={mal_id} "
                          f"(percobaan {attempt}/{max_retries}), coba lagi...")
                    time.sleep(2 * attempt)  # backoff: 2s, 4s, 6s
                    continue
                resp.raise_for_status()
            except requests.RequestException as e:
                print(f"[WARN] Jikan API gagal untuk mal_id={mal_id} (percobaan {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    return None
                time.sleep(2 * attempt)
                continue

            payload = resp.json()
            data = payload.get("data", {})
            snapshot = {
                "mal_id": mal_id,
                "fetched_at": datetime.now(timezone.utc).isoformat(),
                "data": data,
            }
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(snapshot, f, ensure_ascii=False, indent=2)
            return data

        print(f"[WARN] Jikan API tetap gagal untuk mal_id={mal_id} setelah {max_retries}x percobaan -- dilewati.")
        return None

    def extract_enrichment_fields(self, anime_data: dict | None) -> dict:
        """Ambil hanya field yang relevan untuk enrichment (Bagian 4.1) --
        BUKAN poster (sudah tersedia dari dataset lewat image_url)."""
        if not anime_data:
            return {}
        trailer = anime_data.get("trailer") or {}
        trailer_url = self._resolve_trailer_url(trailer)
        return {
            "status_terkini": anime_data.get("status"),
            "episode_terkini": anime_data.get("episodes"),
            "trailer_url": trailer_url,
            "skor_terkini": anime_data.get("score"),
        }

    @staticmethod
    def _resolve_trailer_url(trailer: dict) -> str | None:
        """
        Jikan API kadang mengisi 'url' & 'youtube_id' sebagai null tapi tetap
        menyediakan 'embed_url' yang valid. Masalahnya, embed_url didesain
        untuk dipasang di dalam <iframe> (bukan dibuka langsung sebagai link)
        -- kalau dibuka langsung di browser, YouTube menolak dengan
        "Error 153" karena parameter autoplay/enablejsapi butuh konteks iframe.

        Solusi: kalau 'url' tidak ada, ekstrak video ID dari embed_url lalu
        bentuk ulang jadi URL watch standar yang aman dibuka langsung.
        """
        if trailer.get("url"):
            return trailer["url"]

        youtube_id = trailer.get("youtube_id")
        embed_url = trailer.get("embed_url")
        if not youtube_id and embed_url:
            match = re.search(r"embed/([a-zA-Z0-9_-]{11})", embed_url)
            if match:
                youtube_id = match.group(1)

        if youtube_id:
            return f"https://www.youtube.com/watch?v={youtube_id}"
        return None

    def enrich_batch(self, mal_ids: list[int], use_cache: bool = True) -> dict:
        """mal_ids -> {mal_id: {field enrichment}}"""
        result = {}
        for mal_id in mal_ids:
            data = self.get_anime(mal_id, use_cache=use_cache)
            result[mal_id] = self.extract_enrichment_fields(data)
        return result


if __name__ == "__main__":
    client = JikanClient()
    test_ids = [1, 5, 6]  # Cowboy Bebop, Cowboy Bebop: Movie, Trigun
    print(f"[INFO] Menguji enrichment untuk mal_id: {test_ids} (rate limit ~1 req/detik)...")
    enrichment = client.enrich_batch(test_ids)
    print(json.dumps(enrichment, ensure_ascii=False, indent=2))
