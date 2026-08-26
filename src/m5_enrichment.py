from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY

_API_UNAVAILABLE = False


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


def _complete(system: str, user: str, max_tokens: int) -> str:
    """Return an LLM completion or an empty string; callers provide local fallbacks."""
    global _API_UNAVAILABLE
    if os.getenv("RAG_DISABLE_LLM") == "1" or not OPENAI_API_KEY or _API_UNAVAILABLE:
        return ""
    try:
        from openai import OpenAI
        response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as exc:
        # Quota/network errors apply to all following chunks in this run. Do not
        # retry 100+ times and make an offline fallback unexpectedly slow.
        _API_UNAVAILABLE = True
        print(f"[warn] Enrichment API failed: {exc}")
        return ""


def _sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+|\n+", text) if sentence.strip()]


def _parse_json(text: str) -> dict:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if not text.strip():
        return ""
    result = _complete("Tóm tắt đoạn văn trong tối đa 2 câu tiếng Việt, không thêm thông tin ngoài đoạn văn.", text, 150)
    if result:
        return result
    return " ".join(_sentences(text)[:2])


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if not text.strip() or n_questions <= 0:
        return []
    result = _complete(
        f"Tạo đúng {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. Mỗi câu một dòng.", text, 200)
    if result:
        questions = [line.strip().lstrip("0123456789.-) ") for line in result.splitlines()]
        return [question if question.endswith("?") else f"{question}?" for question in questions if question][:n_questions]
    return [f"Thông tin nào được nêu về: {sentence.rstrip('.')}?" for sentence in _sentences(text)[:n_questions]]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    if not text.strip():
        return ""
    context = _complete(
        "Viết đúng một câu ngắn mô tả chủ đề và vị trí của đoạn trích trong tài liệu. Không suy đoán.",
        f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}", 80)
    if not context:
        context = f"Trích từ tài liệu {document_title or 'nguồn nội bộ'} về: {summarize_chunk(text)}"
    return f"{context}\n\n{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    result = _parse_json(_complete(
        'Trả về JSON duy nhất theo schema {"topic":"...","entities":["..."],"category":"policy|hr|it|finance","language":"vi|en"}.', text, 150))
    if result:
        return result
    lowered = text.lower()
    category = "it" if any(word in lowered for word in ("mật khẩu", "vpn", "bảo mật")) else "hr"
    return {"topic": _sentences(text)[0][:80] if _sentences(text) else "general",
            "entities": [], "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    result = _parse_json(_complete(
        """Phân tích đoạn văn và trả về JSON duy nhất:
{"summary":"2 câu", "questions":["3 câu hỏi"], "context":"1 câu mô tả vị trí/chủ đề",
 "metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}}""",
        f"Tài liệu: {source}\n\nĐoạn văn:\n{text}", 400))
    if result:
        return result
    summary = " ".join(_sentences(text)[:2])
    return {"summary": summary, "questions": generate_hypothesis_questions(text),
            "context": f"Trích từ tài liệu {source or 'nguồn nội bộ'}.",
            "metadata": extract_metadata(text)}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
