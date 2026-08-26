from __future__ import annotations

"""Phase B: pairwise judging, swap consistency, Cohen's kappa and bias."""

import json
import os
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import HUMAN_LABELS_PATH, JUDGE_MODEL, OPENAI_API_KEY


@dataclass
class JudgeResult:
    question: str
    answer_a: str
    answer_b: str
    winner_pass1: str
    winner_pass2: str
    final_winner: str
    reasoning_pass1: str
    reasoning_pass2: str
    position_consistent: bool
    scores_pass1: dict = field(default_factory=dict)
    scores_pass2: dict = field(default_factory=dict)


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in re.findall(r"\w+", text, flags=re.UNICODE) if len(token) > 1}


def _local_scores(question: str, answer_a: str, answer_b: str) -> tuple[float, float]:
    query_tokens = _tokens(question)

    def score(answer: str) -> float:
        answer_tokens = _tokens(answer)
        overlap = len(query_tokens & answer_tokens) / max(len(query_tokens), 1)
        useful = min(len(answer_tokens) / 30.0, 1.0)
        penalty = 0.15 if not answer.strip() else 0.0
        return max(0.0, min(1.0, 0.65 * overlap + 0.35 * useful - penalty))

    return score(answer_a), score(answer_b)


def _valid_judge(payload: dict) -> dict | None:
    if not isinstance(payload, dict) or payload.get("winner") not in {"A", "B", "tie"}:
        return None
    scores = payload.get("scores", {})
    try:
        scores = {"A": max(0.0, min(1.0, float(scores.get("A", 0.0)))),
                  "B": max(0.0, min(1.0, float(scores.get("B", 0.0))))}
    except (TypeError, ValueError, AttributeError):
        return None
    return {"winner": payload["winner"], "reasoning": str(payload.get("reasoning", "")), "scores": scores}


def pairwise_judge(question: str, answer_a: str, answer_b: str) -> dict:
    """Return a JSON-compatible judge result; use OpenAI when configured."""
    if (OPENAI_API_KEY and os.getenv("RAG_DISABLE_LLM") != "1"
            and not OPENAI_API_KEY.startswith(("sk-...", "your-", "replace-"))):
        try:
            from openai import OpenAI
            prompt = (
                "Evaluate two HR-policy answers for accuracy, completeness and conciseness. "
                "Return JSON only with winner A/B/tie, brief reasoning, and scores A/B in [0,1].\n\n"
                f"Question: {question}\nAnswer A: {answer_a}\nAnswer B: {answer_b}"
            )
            response = OpenAI(api_key=OPENAI_API_KEY).chat.completions.create(
                model=JUDGE_MODEL, temperature=0,
                messages=[{"role": "system", "content": "You are a strict RAG evaluator."},
                          {"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
            )
            result = _valid_judge(json.loads(response.choices[0].message.content or "{}"))
            if result:
                return result
        except Exception as exc:
            print(f"[warn] LLM judge unavailable; using deterministic fallback: {exc}")

    score_a, score_b = _local_scores(question, answer_a, answer_b)
    margin = abs(score_a - score_b)
    winner = "tie" if margin < 0.05 else ("A" if score_a > score_b else "B")
    reasoning = "Hai câu trả lời có chất lượng tương đương theo heuristic offline." if winner == "tie" else (
        f"Answer {winner} có mức liên quan và độ đầy đủ cao hơn theo đánh giá offline."
    )
    return {"winner": winner, "reasoning": reasoning, "scores": {"A": round(score_a, 4), "B": round(score_b, 4)}}


def swap_and_average(question: str, answer_a: str, answer_b: str) -> JudgeResult:
    pass1 = _valid_judge(pairwise_judge(question, answer_a, answer_b)) or {
        "winner": "tie", "reasoning": "Invalid judge result", "scores": {"A": 0.0, "B": 0.0}}
    pass2_raw = _valid_judge(pairwise_judge(question, answer_b, answer_a)) or {
        "winner": "tie", "reasoning": "Invalid judge result", "scores": {"A": 0.0, "B": 0.0}}
    winner_pass2 = {"A": "B", "B": "A", "tie": "tie"}[pass2_raw["winner"]]
    consistent = pass1["winner"] == winner_pass2
    final = pass1["winner"] if consistent else "tie"
    return JudgeResult(
        question=question, answer_a=answer_a, answer_b=answer_b,
        winner_pass1=pass1["winner"], winner_pass2=winner_pass2, final_winner=final,
        reasoning_pass1=pass1["reasoning"], reasoning_pass2=pass2_raw["reasoning"],
        position_consistent=consistent, scores_pass1=pass1["scores"],
        scores_pass2={"A": pass2_raw["scores"]["B"], "B": pass2_raw["scores"]["A"]},
    )


def cohen_kappa(judge_labels: list[int], human_labels: list[int]) -> float:
    if len(judge_labels) != len(human_labels) or not judge_labels:
        raise ValueError("judge_labels and human_labels must have equal non-empty lengths")
    n = len(judge_labels)
    observed = sum(judge == human for judge, human in zip(judge_labels, human_labels)) / n
    expected = sum(
        (judge_labels.count(label) / n) * (human_labels.count(label) / n)
        for label in set(judge_labels) | set(human_labels)
    )
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return max(-1.0, min(1.0, (observed - expected) / (1.0 - expected)))


def bias_report(judge_results: list[JudgeResult]) -> dict:
    total = len(judge_results)
    inconsistent = sum(not result.position_consistent for result in judge_results)
    decisive = [result for result in judge_results if result.final_winner in {"A", "B"}]
    a_longer = sum(result.final_winner == "A" and len(result.answer_a) > len(result.answer_b) for result in decisive)
    b_longer = sum(result.final_winner == "B" and len(result.answer_b) > len(result.answer_a) for result in decisive)
    verbosity = (a_longer + b_longer) / len(decisive) if decisive else 0.0
    position_rate = inconsistent / total if total else 0.0
    return {
        "total_judged": total, "position_bias_rate": round(position_rate, 3),
        "position_bias_count": inconsistent, "verbosity_bias": round(verbosity, 3),
        "verbosity_details": {"a_wins_a_longer": a_longer, "b_wins_b_longer": b_longer,
                               "total_decisive": len(decisive)},
        "interpretation": (
            "Position bias cao; tiếp tục dùng swap-and-average."
            if position_rate > 0.3 else "Position bias thấp; judge tương đối ổn định."
        ),
    }


def _json_result(result: JudgeResult) -> dict:
    return {"question": result.question, "answer_a": result.answer_a, "answer_b": result.answer_b,
            "winner_pass1": result.winner_pass1, "winner_pass2": result.winner_pass2,
            "final_winner": result.final_winner, "reasoning_pass1": result.reasoning_pass1,
            "reasoning_pass2": result.reasoning_pass2, "position_consistent": result.position_consistent,
            "scores_pass1": result.scores_pass1, "scores_pass2": result.scores_pass2}


def run_phase_b(answers_path: str = "answers_50q.json", report_path: str = "reports/judge_results.json") -> dict:
    with open(answers_path, encoding="utf-8") as handle:
        answers = json.load(handle)
    pairs = []
    for item in answers[:10]:
        pairs.append(swap_and_average(item["question"], item.get("answer", ""), item.get("ground_truth", "")))
    with open(HUMAN_LABELS_PATH, encoding="utf-8") as handle:
        human_data = json.load(handle)
    by_id = {int(item.get("id", item.get("question_id"))): item for item in answers}
    judge_labels = []
    human_labels = []
    for item in human_data:
        answer = by_id.get(int(item["question_id"]), {})
        judged = swap_and_average(item["question"], answer.get("answer", ""), item.get("model_answer", ""))
        judge_labels.append(1 if judged.final_winner == "B" else 0)
        human_labels.append(int(item["human_label"]))
    report = {
        "pairwise_samples": [_json_result(result) for result in pairs],
        "bias_report": bias_report(pairs),
        "human_agreement": {"n": len(human_labels), "cohen_kappa": round(cohen_kappa(judge_labels, human_labels), 4),
                            "judge_labels": judge_labels, "human_labels": human_labels},
    }
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Phase B report saved -> {report_path}")
    return report


if __name__ == "__main__":
    run_phase_b()
