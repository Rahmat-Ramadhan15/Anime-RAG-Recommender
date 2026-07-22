"""
Minggu 9: Aplikasi Gradio untuk Deployment (Hugging Face Spaces)

Entry point untuk HF Spaces -- file ini HARUS bernama app.py di root repo
(konvensi HF Spaces).

Desain latensi: teks jawaban dihasilkan gaya Kondisi B (retrieval saja, tanpa
menunggu enrichment) karena B vs C tidak berbeda signifikan secara kualitas
(Minggu 7). Poster instan dari dataset, trailer menyusul lewat respons
progresif (generator, yield 2x).

Poster HANYA ditampilkan untuk anime yang benar-benar disebut di teks jawaban
(dicocokkan via judul), bukan seluruh top-k hasil retrieval mentah -- supaya
galeri tidak menampilkan anime yang tidak relevan dengan apa yang sebenarnya
direkomendasikan LLM.

Model dimuat lewat load_llm_gguf() (CPU-compatible), BUKAN load_llm()
(bitsandbytes, GPU-only) yang dipakai untuk eksperimen Kaggle.
"""

import sys
import re
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import gradio as gr
from rag_pipeline import RagPipeline

print("[INFO] Memuat pipeline (ini hanya terjadi sekali saat Space start)...")
pipe = RagPipeline()
pipe.load_index()
pipe.load_llm_gguf()
print("[INFO] Pipeline siap.")

EXAMPLE_PROMPTS = [
    "Rekomendasikan anime action dengan tema luar angkasa",
    "Aku suka Kimi no Na wa, ada yang mirip?",
    "Anime comedy yang sudah tamat, episodenya di bawah 13",
]


def parse_structured_answer(answer_text: str) -> list:
    """
    Parse jawaban LLM yang mengikuti format wajib "### Judul\n(alasan)".
    Kembalikan list of {"title": str, "reasoning": str}. Kalau format tidak
    terdeteksi sama sekali (LLM tidak mematuhi instruksi), kembalikan list
    kosong -- caller harus fallback ke tampilan teks polos tanpa kartu.
    """
    blocks = re.split(r"^###\s+", answer_text, flags=re.MULTILINE)
    parsed = []
    for block in blocks[1:]:  # blocks[0] adalah teks sebelum heading pertama (biasanya kosong/intro)
        lines = block.strip().split("\n", 1)
        title = lines[0].strip().strip("[]")
        reasoning = lines[1].strip() if len(lines) > 1 else ""
        if title:
            parsed.append({"title": title, "reasoning": reasoning})
    return parsed


def match_to_retrieved(parsed_title: str, retrieved: list):
    """Cocokkan judul hasil parsing ke salah satu kandidat retrieval (exact/substring, case-insensitive)."""
    title_lower = parsed_title.strip().lower()
    for d in retrieved:
        candidates = [t for t in [d.get("title"), d.get("title_english")] if t]
        for c in candidates:
            c_lower = str(c).strip().lower()
            if c_lower == title_lower or c_lower in title_lower or title_lower in c_lower:
                return d
    return None


def build_interleaved_message(answer_text: str, retrieved: list) -> tuple:
    """
    Susun pesan akhir: tiap anime yang direkomendasikan (format ### Judul) diikuti
    LANGSUNG oleh poster + info singkatnya, jadi teks dan gambar berada di tempat
    yang sama (sesuai permintaan). Kembalikan (pesan_markdown, list_anime_yang_cocok).
    """
    parsed = parse_structured_answer(answer_text)
    if not parsed:
        # LLM tidak mengikuti format -- tampilkan apa adanya tanpa poster
        return answer_text, []

    parts = []
    matched_docs = []
    for item in parsed:
        doc = match_to_retrieved(item["title"], retrieved)
        parts.append(f"**{item['title']}**")
        if item["reasoning"]:
            parts.append(item["reasoning"])
        if doc and doc.get("image_url"):
            meta_bits = []
            valid_score = _valid_score(doc.get("mal_score"))
            if valid_score is not None:
                meta_bits.append(f"★ {valid_score}")
            if doc.get("genres"):
                meta_bits.append("|".join(str(doc["genres"]).split("|")[:3]))
            caption = " — ".join(meta_bits)
            parts.append(f"![{item['title']}]({doc['image_url']})")
            if caption:
                parts.append(f"_{caption}_")
            matched_docs.append(doc)
        parts.append("")  # spasi antar blok

    return "\n\n".join(parts).strip(), matched_docs


def _valid_score(value) -> float | None:
    """Kembalikan float valid, atau None kalau kosong/NaN (NaN itu 'truthy' di Python,
    jadi pengecekan `if value:` biasa tidak cukup -- ini sumber bug '★ nan' sebelumnya)."""
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def respond(message: str, history):
    if not message.strip():
        yield history
        return

    k = pipe.cfg["retrieval"]["top_k_final"]
    retrieved = pipe.retrieve(message, k=k, use_rerank=True)

    # Tahap 1: generate jawaban TANPA menunggu Jikan (gaya Kondisi B).
    result = pipe.generate(message, k=k, use_retrieval=True, use_enrichment=False, pre_retrieved=retrieved)

    if result["condition"] == "BLOCKED":
        new_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": result["answer"]},
        ]
        yield new_history
        return

    # Susun pesan: tiap anime yang direkomendasikan (### Judul) langsung diikuti posternya
    final_message, matched_docs = build_interleaved_message(result["answer"], retrieved)

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_message},
    ]
    yield new_history

    matched_mal_ids = [d["mal_id"] for d in matched_docs]
    if not matched_mal_ids:
        return

    try:
        enrichment_data = pipe.enrich(matched_mal_ids, max_retries=1)
    except Exception as e:
        print(f"[WARN] Enrichment gagal, tampilkan tanpa trailer: {e}")
        enrichment_data = {}

    trailer_lines = []
    for d in matched_docs:
        enrich = enrichment_data.get(d["mal_id"], {})
        if enrich.get("trailer_url"):
            trailer_lines.append(f"- **{d.get('title') or d['mal_id']}**: [{enrich['trailer_url']}]({enrich['trailer_url']})")

    if trailer_lines:
        final_message_with_trailer = final_message + "\n\n---\n**Trailer:**\n" + "\n".join(trailer_lines)
        new_history[-1] = {"role": "assistant", "content": final_message_with_trailer}
        yield new_history


CUSTOM_CSS = """
:root {
    --paper: #F5F6F8;
    --ink: #1E2233;
    --ink-soft: #4B5066;
    --accent: #3454D1;
    --accent-soft: #E8ECFB;
    --mark: #FFB100;
}
.gradio-container {
    background: var(--paper) !important;
    font-family: -apple-system, "Segoe UI", Roboto, sans-serif !important;
}
#app-header {
    background: var(--ink);
    color: white;
    padding: 22px 26px;
    border-radius: 14px;
    margin-bottom: 18px;
}
#app-header h1 {
    font-family: Georgia, serif;
    margin: 0 0 6px 0;
    font-size: 26px;
}
#app-header p {
    margin: 0;
    color: #C7CCE8;
    font-size: 14px;
}
.example-btn {
    border: 1.5px solid var(--accent) !important;
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    border-radius: 999px !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
.example-btn:hover {
    background: var(--accent) !important;
    color: white !important;
}
#poster-gallery {
    border-radius: 12px !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title="Chatbot Rekomendasi Anime") as demo:
    gr.HTML(
        '<div id="app-header">'
        '<h1>🎌 Chatbot Rekomendasi Anime</h1>'
        '<p>Rekomendasi berbasis RAG + SLM &middot; skripsi &middot; tidak melayani permintaan konten dewasa/eksplisit</p>'
        '</div>'
    )

    chatbot = gr.Chatbot(height=520, label="Percakapan", avatar_images=(None, "🎬"))

    gr.Markdown("**Coba salah satu contoh ini:**")
    with gr.Row():
        example_buttons = [gr.Button(p, elem_classes="example-btn", size="sm") for p in EXAMPLE_PROMPTS]

    msg = gr.Textbox(
        label="Pesan Anda",
        placeholder="mis. rekomendasikan anime action dengan tema luar angkasa",
    )

    # Setiap tombol contoh mengisi textbox dengan teks promptnya sendiri.
    # Pakai default-argument (p=p) supaya tidak kena late-binding closure bug di loop.
    for btn, prompt_text in zip(example_buttons, EXAMPLE_PROMPTS):
        btn.click(fn=lambda p=prompt_text: p, inputs=None, outputs=msg)

    # Poster & trailer sekarang menyatu di dalam pesan chatbot itu sendiri
    # (lihat build_interleaved_message) -- tidak perlu komponen Gallery/Markdown terpisah lagi.
    msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="indigo"))
