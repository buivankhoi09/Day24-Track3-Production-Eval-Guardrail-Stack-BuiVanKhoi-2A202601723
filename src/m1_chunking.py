from __future__ import annotations

"""
Module 1: Advanced Chunking Strategies
=======================================
Implement semantic, hierarchical, và structure-aware chunking.
So sánh với basic chunking (baseline) để thấy improvement.

Test: pytest tests/test_m1.py
"""

import glob
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR,
    HIERARCHICAL_CHILD_SIZE,
    HIERARCHICAL_PARENT_SIZE,
    SEMANTIC_THRESHOLD,
)


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)
    parent_id: str | None = None


def _extract_pdf_text(path: str) -> str:
    """Extract text layer từ PDF. Trả về "" nếu PDF là scan ảnh (không có text)."""
    try:
        from pypdf import PdfReader
        reader = PdfReader(path)
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages).strip()
    except Exception as exc:
        # A broken or image-only PDF must not abort indexing the remaining corpus.
        print(f"[warn] Could not extract PDF {os.path.basename(path)}: {exc}")
        return ""


def load_documents(data_dir: str = DATA_DIR) -> list[dict]:
    """Load tất cả markdown và PDF (có text layer) từ data/. (Đã implement sẵn)

    - .md: đọc trực tiếp.
    - .pdf: trích text layer bằng pypdf. PDF scan ảnh (không có text) bị bỏ qua
      kèm cảnh báo — RAG text-based không xử lý được scan nếu chưa OCR.
    """
    docs = []
    for fp in sorted(glob.glob(os.path.join(data_dir, "*.md"))):
        with open(fp, encoding="utf-8") as f:
            docs.append({"text": f.read(), "metadata": {"source": os.path.basename(fp)}})

    for fp in sorted(glob.glob(os.path.join(data_dir, "*.pdf"))):
        text = _extract_pdf_text(fp)
        if text:
            docs.append({"text": text, "metadata": {"source": os.path.basename(fp)}})
        else:
            # Keep console output ASCII-safe for Windows terminals using legacy encodings.
            print(f"[warn] Skipping {os.path.basename(fp)}: no PDF text layer (OCR required).")

    return docs


# ─── Baseline: Basic Chunking (để so sánh) ──────────────


def chunk_basic(text: str, chunk_size: int = 500, metadata: dict | None = None) -> list[Chunk]:
    """
    Basic chunking: split theo paragraph (\\n\\n).
    Đây là baseline — KHÔNG phải mục tiêu của module này.
    (Đã implement sẵn)
    """
    metadata = metadata or {}
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    for i, para in enumerate(paragraphs):
        if len(current) + len(para) > chunk_size and current:
            chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(Chunk(text=current.strip(), metadata={**metadata, "chunk_index": len(chunks)}))
    return chunks


# ─── Strategy 1: Semantic Chunking ───────────────────────


def chunk_semantic(text: str, threshold: float = SEMANTIC_THRESHOLD,
                   metadata: dict | None = None) -> list[Chunk]:
    """
    Split text by sentence similarity — nhóm câu cùng chủ đề.
    Tốt hơn basic vì không cắt giữa ý.
    """
    metadata = metadata or {}
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if s.strip()]
    if not sentences:
        return []
    if len(sentences) == 1:
        return [Chunk(sentences[0], {**metadata, "strategy": "semantic", "chunk_index": 0})]

    try:
        if os.getenv("RAG_SKIP_MODEL_LOAD") == "1":
            raise RuntimeError("model loading disabled by RAG_SKIP_MODEL_LOAD")
        from numpy import dot
        from numpy.linalg import norm
        from sentence_transformers import SentenceTransformer

        # Cache the model on the function so repeated documents do not reload it.
        model = getattr(chunk_semantic, "_model", None)
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
            chunk_semantic._model = model
        embeddings = model.encode(sentences, show_progress_bar=False)

        groups: list[list[str]] = [[sentences[0]]]
        for index in range(1, len(sentences)):
            similarity = float(dot(embeddings[index - 1], embeddings[index]) /
                               (norm(embeddings[index - 1]) * norm(embeddings[index]) + 1e-9))
            if similarity < threshold:
                groups.append([sentences[index]])
            else:
                groups[-1].append(sentences[index])
    except Exception as exc:
        # Deterministic fallback keeps the pipeline usable before a model is downloaded.
        print(f"[warn] Semantic model unavailable; using sentence groups: {exc}")
        groups = [sentences]

    return [
        Chunk(" ".join(group), {**metadata, "strategy": "semantic", "chunk_index": index})
        for index, group in enumerate(groups)
    ]


# ─── Strategy 2: Hierarchical Chunking ──────────────────


def chunk_hierarchical(text: str, parent_size: int = HIERARCHICAL_PARENT_SIZE,
                       child_size: int = HIERARCHICAL_CHILD_SIZE,
                       metadata: dict | None = None) -> tuple[list[Chunk], list[Chunk]]:
    """
    Parent-child hierarchy: retrieve child (precision) → return parent (context).
    Đây là default recommendation cho production RAG.

    Returns:
        (parents, children) — mỗi child có parent_id link đến parent.
    """
    metadata = metadata or {}
    if not text.strip():
        return [], []

    def split_units(value: str, limit: int) -> list[str]:
        """Prefer paragraph boundaries, then safely split an oversized paragraph."""
        paragraphs = [p.strip() for p in value.split("\n\n") if p.strip()]
        units: list[str] = []
        current = ""
        for paragraph in paragraphs:
            pieces = [paragraph[i:i + limit] for i in range(0, len(paragraph), limit)] or [paragraph]
            for piece in pieces:
                candidate = f"{current}\n\n{piece}".strip() if current else piece
                if current and len(candidate) > limit:
                    units.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            units.append(current)
        return units

    source = str(metadata.get("source", "document"))
    source_key = re.sub(r"\W+", "_", source).strip("_") or "document"
    parents: list[Chunk] = []
    children: list[Chunk] = []
    for parent_index, parent_text in enumerate(split_units(text, parent_size)):
        parent_id = f"{source_key}_parent_{parent_index}"
        parent_metadata = {**metadata, "chunk_type": "parent", "parent_id": parent_id,
                           "parent_index": parent_index}
        parents.append(Chunk(parent_text, parent_metadata, parent_id=parent_id))
        for child_index, child_text in enumerate(split_units(parent_text, child_size)):
            children.append(Chunk(
                child_text,
                {**metadata, "chunk_type": "child", "child_index": child_index,
                 "parent_text": parent_text},
                parent_id=parent_id,
            ))
    return parents, children


# ─── Strategy 3: Structure-Aware Chunking ────────────────


def chunk_structure_aware(text: str, metadata: dict | None = None) -> list[Chunk]:
    """
    Parse markdown headers → chunk theo logical structure.
    Giữ nguyên tables, code blocks, lists — không cắt giữa chừng.
    """
    metadata = metadata or {}
    if not text.strip():
        return []

    parts = re.split(r"(^#{1,3}\s+.+$)", text, flags=re.MULTILINE)
    current_header = ""
    chunks: list[Chunk] = []
    preamble = parts[0].strip()
    if preamble:
        chunks.append(Chunk(preamble, {**metadata, "section": "", "strategy": "structure", "chunk_index": 0}))

    for index in range(1, len(parts), 2):
        header = parts[index].strip()
        content = parts[index + 1].strip() if index + 1 < len(parts) else ""
        current_header = header
        section_text = f"{current_header}\n\n{content}".strip()
        if section_text:
            chunks.append(Chunk(section_text, {
                **metadata,
                "section": current_header.lstrip("#").strip(),
                "strategy": "structure",
                "chunk_index": len(chunks),
            }))

    return chunks


# ─── A/B Test: Compare All Strategies ────────────────────


def compare_strategies(documents: list[dict]) -> dict:
    """
    Run all strategies on documents and compare.
    (Đã implement sẵn — sẽ hoạt động khi bạn implement 3 strategies ở trên)
    """
    def _stats(chunk_list):
        lengths = [len(c.text) for c in chunk_list]
        if not lengths:
            return {"count": 0, "avg_len": 0, "min_len": 0, "max_len": 0}
        return {
            "count": len(lengths),
            "avg_len": round(sum(lengths) / len(lengths)),
            "min_len": min(lengths),
            "max_len": max(lengths),
        }

    all_text = "\n\n".join(d["text"] for d in documents)
    meta = {"source": "all"}

    basic = chunk_basic(all_text, metadata=meta)
    semantic = chunk_semantic(all_text, metadata=meta)
    parents, children = chunk_hierarchical(all_text, metadata=meta)
    structure = chunk_structure_aware(all_text, metadata=meta)

    results = {
        "basic": _stats(basic),
        "semantic": _stats(semantic),
        "hierarchical": {**_stats(children), "parents": len(parents)},
        "structure": _stats(structure),
    }

    print(f"{'Strategy':<15} {'Chunks':>7} {'Avg':>5} {'Min':>5} {'Max':>5}")
    for name, s in results.items():
        print(f"{name:<15} {s['count']:>7} {s['avg_len']:>5} {s['min_len']:>5} {s['max_len']:>5}")

    return results


if __name__ == "__main__":
    docs = load_documents()
    print(f"Loaded {len(docs)} documents")
    results = compare_strategies(docs)
    for name, stats in results.items():
        print(f"  {name}: {stats}")
