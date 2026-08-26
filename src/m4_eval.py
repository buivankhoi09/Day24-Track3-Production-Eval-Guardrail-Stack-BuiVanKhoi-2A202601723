from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import json
import math
import os
import re
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    if not (len(questions) == len(answers) == len(contexts) == len(ground_truths)):
        raise ValueError("questions, answers, contexts, and ground_truths must have equal lengths")

    zero_rows = [EvalResult(question, answer, context, ground_truth, 0.0, 0.0, 0.0, 0.0)
                 for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths)]
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict({"question": questions, "answer": answers,
                                     "contexts": contexts, "ground_truth": ground_truths})
        frame = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                           context_precision, context_recall]).to_pandas()
        def value(row, key):
            raw = row.get(key, 0.0)
            value = float(raw)
            return value if math.isfinite(value) else 0.0
        rows = [EvalResult(str(row["question"]), str(row["answer"]), list(row["contexts"]),
                           str(row["ground_truth"]), value(row, "faithfulness"),
                           value(row, "answer_relevancy"), value(row, "context_precision"),
                           value(row, "context_recall")) for _, row in frame.iterrows()]
        metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        return {metric: sum(getattr(row, metric) for row in rows) / len(rows) if rows else 0.0
                for metric in metrics} | {"per_question": rows}
    except Exception as exc:
        print(f"[warn] RAGAS evaluation unavailable: {exc}")
        def tokens(value: str) -> set[str]:
            return {word.casefold() for word in re.findall(r"\w+", value, flags=re.UNICODE) if len(word) > 1}
        rows = []
        for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths):
            question_words, answer_words, truth_words = tokens(question), tokens(answer), tokens(ground_truth)
            context_words = tokens(" ".join(context))
            answer_relevancy = len(question_words & answer_words) / max(len(question_words), 1)
            faithfulness = len(answer_words & context_words) / max(len(answer_words), 1)
            context_recall = len(truth_words & context_words) / max(len(truth_words), 1)
            precision = sum(bool(tokens(item) & truth_words) for item in context) / max(len(context), 1)
            rows.append(EvalResult(question, answer, context, ground_truth,
                                   min(faithfulness, 1.0), min(answer_relevancy, 1.0),
                                   min(precision, 1.0), min(context_recall, 1.0)))
        metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
        return {metric: sum(getattr(row, metric) for row in rows) / len(rows) if rows else 0.0
                for metric in metrics} | {"per_question": rows}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    tree = {
        "faithfulness": ("LLM answer is not grounded in retrieved context", "Tighten the grounded-answer prompt and reduce answer scope."),
        "context_recall": ("Relevant evidence was not retrieved", "Improve chunking, query expansion, or hybrid retrieval coverage."),
        "context_precision": ("Retrieved context contains too much noise", "Tune top-k, add metadata filters, or strengthen reranking."),
        "answer_relevancy": ("Answer does not directly address the question", "Improve the answer prompt and include question intent."),
    }
    ranked = []
    for result in eval_results:
        values = {metric: float(getattr(result, metric)) for metric in tree}
        worst = min(values, key=values.get)
        diagnosis, suggested_fix = tree[worst]
        ranked.append({"question": result.question, "expected": result.ground_truth,
                       "answer": result.answer, "worst_metric": worst,
                       "score": sum(values.values()) / len(values),
                       "diagnosis": diagnosis, "suggested_fix": suggested_fix})
    return sorted(ranked, key=lambda item: item["score"])[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
