"""
Minggu 5: Enrichment via Jikan API
Status: BELUM DIIMPLEMENTASI

Rencana:
1. Terima daftar mal_id hasil retrieval
2. Panggil Jikan API (https://api.jikan.moe/v4/anime/{mal_id})
   -- hormati rate limit: 3 req/detik, 60 req/menit (configs/config.yaml)
3. Ambil poster, trailer, status tayang, info terkini
4. Simpan cache respons (JSON + timestamp) ke data/api_cache/ untuk reproducibility
"""

raise NotImplementedError("Akan diimplementasikan pada Minggu 5")
