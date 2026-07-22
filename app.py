"""
Minggu 9: Aplikasi Gradio untuk Deployment (Hugging Face Spaces)

Entry point untuk HF Spaces -- file ini HARUS bernama app.py di root repo
(konvensi HF Spaces).

KEPUTUSAN ARSITEKTUR (setelah evaluasi Minggu 7): aplikasi deployment TIDAK
memakai enrichment (Jikan API/trailer) sama sekali -- hanya Kondisi B (RAG)
+ poster dari dataset lokal. Alasan:
  1. Uji Wilcoxon Minggu 7 menunjukkan B vs C TIDAK berbeda signifikan
     (p=0.53) secara kualitas jawaban -- enrichment tidak terbukti menambah
     kualitas secara terukur.
  2. Jikan API (unofficial, tanpa SLA) sering timeout/gagal, menambah
     ketidakandalan tanpa manfaat kualitas yang terbukti.
  3. Poster dari image_url dataset (zero risiko API eksternal) sudah cukup
     memenuhi revisi penguji ("inovasi selain teks").
Kondisi C, enrichment.py, dan hasil evaluasinya TETAP dipertahankan di repo
sebagai bagian sah dari ablation study (Minggu 7) -- keputusan ini berbasis
bukti dari situ, bukan sekadar penyederhanaan tanpa alasan.

Poster HANYA ditampilkan untuk anime yang benar-benar disebut di teks jawaban
(format terstruktur "### Judul", dicocokkan ke hasil retrieval) -- bukan
seluruh top-k mentah, supaya tidak ada mismatch poster vs teks.

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

FOUND_HEADER = "✅ **Rekomendasi ditemukan!**\n\n"
NOT_FOUND_HEADER = "🔍 "  # dipakai kalau LLM tidak menghasilkan format terstruktur (fallback teks polos)


def _valid_score(value) -> float | None:
    """Kembalikan float valid, atau None kalau kosong/NaN (NaN itu 'truthy' di Python,
    jadi pengecekan `if value:` biasa tidak cukup)."""
    try:
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def parse_structured_answer(answer_text: str) -> list:
    """
    Parse jawaban LLM yang mengikuti format wajib:
        ### Judul
        Plot: ...
        Alasan: ...
    Kembalikan list of {"title": str, "body": str}. Kalau format tidak
    terdeteksi sama sekali, kembalikan list kosong -- caller fallback ke
    tampilan teks polos tanpa kartu/poster.
    """
    blocks = re.split(r"^###\s+", answer_text, flags=re.MULTILINE)
    parsed = []
    for block in blocks[1:]:  # blocks[0] = teks sebelum heading pertama (biasanya kosong)
        lines = block.strip().split("\n", 1)
        title = lines[0].strip().strip("[]")
        body = lines[1].strip() if len(lines) > 1 else ""
        if title:
            parsed.append({"title": title, "body": body})
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


def build_interleaved_message(answer_text: str, retrieved: list) -> str:
    """
    Susun pesan akhir: tiap anime yang direkomendasikan (### Judul) diikuti
    LANGSUNG oleh poster + info singkatnya (skor, genre) -- teks dan gambar
    di tempat yang sama. Kalau LLM tidak mengikuti format, kembalikan teks
    apa adanya tanpa poster (lebih aman daripada poster yang salah).
    """
    parsed = parse_structured_answer(answer_text)
    if not parsed:
        return NOT_FOUND_HEADER + answer_text

    parts = [FOUND_HEADER.strip()]
    for item in parsed:
        doc = match_to_retrieved(item["title"], retrieved)
        parts.append(f"### {item['title']}")
        if item["body"]:
            parts.append(item["body"])
        if doc and doc.get("image_url"):
            meta_bits = []
            valid_score = _valid_score(doc.get("mal_score"))
            if valid_score is not None:
                meta_bits.append(f"★ {valid_score}")
            if doc.get("genres"):
                meta_bits.append("|".join(str(doc["genres"]).split("|")[:3]))
            if doc.get("themes"):
                meta_bits.append("|".join(str(doc["themes"]).split("|")[:2]))
            parts.append(f"![{item['title']}]({doc['image_url']})")
            if meta_bits:
                parts.append(f"_{' — '.join(meta_bits)}_")
        parts.append("---")

    if parts and parts[-1] == "---":
        parts.pop()

    return "\n\n".join(parts).strip()


FOLLOWUP_HINTS = [
    "sebelumnya", "tadi", "barusan", "yang lain", "lainnya", "itu tadi",
    "seperti itu", "yang serupa", "mirip itu",
]


def is_followup(message: str) -> bool:
    """Heuristik ringan: pesan pendek yang menyebut kata rujukan ('sebelumnya',
    'tadi', dsb) dianggap melanjutkan konteks giliran sebelumnya, bukan query baru."""
    text = message.lower()
    return len(text.split()) <= 12 and any(hint in text for hint in FOLLOWUP_HINTS)


def build_effective_query(message: str, history: list) -> str:
    """
    Kalau pesan terdeteksi sebagai follow-up, gabungkan dengan pesan user
    SEBELUMNYA dari history supaya retrieval punya konteks yang cukup.
    """
    if not is_followup(message) or not history:
        return message

    last_user_msg = None
    for turn in reversed(history):
        if turn.get("role") == "user":
            last_user_msg = turn.get("content")
            break

    return f"{last_user_msg} {message}" if last_user_msg else message


def respond(message: str, history):
    if not message.strip():
        yield history
        return

    effective_query = build_effective_query(message, history)

    k = pipe.cfg["retrieval"]["top_k_final"]
    retrieved = pipe.retrieve(effective_query, k=k, use_rerank=True)

    # Kondisi B saja (RAG, tanpa enrichment) -- lihat alasan di docstring atas.
    result = pipe.generate(
        effective_query, k=k, use_retrieval=True, use_enrichment=False, pre_retrieved=retrieved
    )

    if result["condition"] == "BLOCKED":
        final_message = result["answer"]
    else:
        final_message = build_interleaved_message(result["answer"], retrieved)

    new_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": final_message},
    ]
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

    for btn, prompt_text in zip(example_buttons, EXAMPLE_PROMPTS):
        btn.click(fn=lambda p=prompt_text: p, inputs=None, outputs=msg)

    msg.submit(respond, [msg, chatbot], [chatbot]).then(lambda: "", None, msg)

if __name__ == "__main__":
    demo.launch(css=CUSTOM_CSS, theme=gr.themes.Soft(primary_hue="indigo"))
