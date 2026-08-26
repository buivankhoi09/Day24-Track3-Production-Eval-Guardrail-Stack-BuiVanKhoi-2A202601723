from __future__ import annotations

"""Phase A: RAGAS evaluation, bottom-10 diagnostics and failure clusters."""

import json
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ANSWERS_PATH, TEST_SET_PATH

Distribution = str
METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
GROUPS = ("factual", "multi_hop", "adversarial")
DIAGNOSTIC_TREE = {
    "faithfulness": ("LLM answer is not grounded in retrieved context", "Tighten the grounded-answer prompt and reduce answer scope"),
    "answer_relevancy": ("Answer does not directly address the question", "Improve the answer prompt and include question intent"),
    "context_precision": ("Retrieved context contains too much noise", "Tune top-k, add metadata filters, or strengthen reranking"),
    "context_recall": ("Relevant evidence was not retrieved", "Improve chunking, query expansion, or hybrid retrieval coverage"),
}


@dataclass
class RagasResult:
    question_id: int
    distribution: Distribution
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float

    @property
    def avg_score(self) -> float:
        return sum(float(getattr(self, metric)) for metric in METRICS) / len(METRICS)

    @property
    def worst_metric(self) -> str:
        return min(METRICS, key=lambda metric: float(getattr(self, metric)))


def load_test_set_50q(path: str = TEST_SET_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_answers(path: str = ANSWERS_PATH) -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing answers file: {path}. Run setup_answers.py first.")
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def group_by_distribution(test_set: list[dict]) -> dict[str, list[dict]]:
    groups = {name: [] for name in GROUPS}
    for item in test_set:
        distribution = item.get("distribution")
        if distribution not in groups:
            raise ValueError(f"Unknown distribution: {distribution!r}")
        groups[distribution].append(item)
    return groups


def _metric_value(row, metric: str) -> float:
    value = row.get(metric) if isinstance(row, dict) else getattr(row, metric, None)
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def run_ragas_50q(answers: list[dict]) -> list[RagasResult]:
    """Evaluate all answers through Day 18's evaluate_ragas API."""
    if not answers:
        return []
    from src.m4_eval import evaluate_ragas

    questions = [str(item.get("question", "")) for item in answers]
    answer_texts = [str(item.get("answer", "")) for item in answers]
    contexts = [list(item.get("contexts", [])) for item in answers]
    ground_truths = [str(item.get("ground_truth", "")) for item in answers]
    raw = evaluate_ragas(questions, answer_texts, contexts, ground_truths)
    rows = raw.get("per_question", []) if isinstance(raw, dict) else []
    results = []
    for index, item in enumerate(answers):
        row = rows[index] if index < len(rows) else {}
        results.append(RagasResult(
            question_id=int(item.get("id", index + 1)),
            distribution=str(item.get("distribution", "factual")),
            question=questions[index], answer=answer_texts[index], contexts=contexts[index],
            ground_truth=ground_truths[index],
            faithfulness=_metric_value(row, "faithfulness"),
            answer_relevancy=_metric_value(row, "answer_relevancy"),
            context_precision=_metric_value(row, "context_precision"),
            context_recall=_metric_value(row, "context_recall"),
        ))
    return results


def bottom_10(results: list[RagasResult]) -> list[dict]:
    output = []
    for rank, result in enumerate(sorted(results, key=lambda item: item.avg_score)[:10], start=1):
        diagnosis, suggested_fix = DIAGNOSTIC_TREE[result.worst_metric]
        output.append({
            "rank": rank, "question_id": result.question_id,
            "distribution": result.distribution, "question": result.question,
            "avg_score": round(result.avg_score, 4), "worst_metric": result.worst_metric,
            "diagnosis": diagnosis, "suggested_fix": suggested_fix,
        })
    return output


def cluster_analysis(results: list[RagasResult]) -> dict:
    matrix = {metric: {group: 0 for group in GROUPS} for metric in METRICS}
    for result in results:
        if result.worst_metric in matrix and result.distribution in GROUPS:
            matrix[result.worst_metric][result.distribution] += 1
    dominant_distribution = max(GROUPS, key=lambda group: sum(matrix[m][group] for m in METRICS))
    dominant_metric = max(METRICS, key=lambda metric: sum(matrix[metric].values()))
    insight = (
        f"Distribution '{dominant_distribution}' has the most low-score cases; "
        f"'{dominant_metric}' is the dominant failure. "
        f"Recommended action: {DIAGNOSTIC_TREE[dominant_metric][1]}."
    )
    return {
        "matrix": matrix,
        "dominant_failure_distribution": dominant_distribution,
        "dominant_failure_metric": dominant_metric,
        "insight": insight,
    }


def save_phase_a_report(results: list[RagasResult], clusters: dict,
                        path: str = "reports/ragas_50q.json") -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    per_distribution = {}
    for group in GROUPS:
        subset = [result for result in results if result.distribution == group]
        if subset:
            per_distribution[group] = {"count": len(subset), **{
                metric: round(sum(getattr(result, metric) for result in subset) / len(subset), 4)
                for metric in METRICS
            }, "avg_score": round(sum(result.avg_score for result in subset) / len(subset), 4)}
    report = {
        "total_questions": len(results), "per_distribution": per_distribution,
        "failure_clusters": clusters, "bottom_10": bottom_10(results),
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Phase A report saved -> {path}")


if __name__ == "__main__":
    test_set = load_test_set_50q()
    groups = group_by_distribution(test_set)
    print("Distribution counts:", {key: len(value) for key, value in groups.items()})
    results = run_ragas_50q(load_answers())
    save_phase_a_report(results, cluster_analysis(results))
    print(f"Evaluated {len(results)} questions")
