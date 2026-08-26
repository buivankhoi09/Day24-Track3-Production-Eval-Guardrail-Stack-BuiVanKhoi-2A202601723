from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if os.getenv("RAG_SKIP_MODEL_LOAD") == "1":
                raise RuntimeError("model loading disabled by RAG_SKIP_MODEL_LOAD")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []
        try:
            scores = self._load_model().predict([(query, doc.get("text", "")) for doc in documents])
            if isinstance(scores, (int, float)):
                scores = [scores]
        except Exception as exc:
            # Retrieval remains usable if the cross-encoder cannot be downloaded.
            print(f"[warn] Reranker unavailable; using retrieval scores: {exc}")
            scores = [doc.get("score", 0.0) for doc in documents]

        scored = sorted(zip(scores, documents), key=lambda item: float(item[0]), reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(doc.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, doc) in enumerate(scored[:top_k])
        ]


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [{"id": index, "text": doc.get("text", ""), "meta": doc}
                        for index, doc in enumerate(documents)]
            ranked = self._model.rerank(RerankRequest(query=query, passages=passages))
            return [
                RerankResult(
                    text=result["text"],
                    original_score=float(documents[result["id"]].get("score", 0.0)),
                    rerank_score=float(result["score"]),
                    metadata=dict(documents[result["id"]].get("metadata", {})),
                    rank=index,
                )
                for index, result in enumerate(ranked[:top_k])
            ]
        except Exception as exc:
            print(f"[warn] FlashRank unavailable; using retrieval scores: {exc}")
            ordered = sorted(documents, key=lambda doc: float(doc.get("score", 0.0)), reverse=True)[:top_k]
            return [RerankResult(doc.get("text", ""), float(doc.get("score", 0.0)),
                                 float(doc.get("score", 0.0)), dict(doc.get("metadata", {})), index)
                    for index, doc in enumerate(ordered)]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
