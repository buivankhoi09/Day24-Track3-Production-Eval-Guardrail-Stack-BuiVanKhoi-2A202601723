from __future__ import annotations

"""Phase C: local PII detection plus optional NeMo Guardrails integration."""

import asyncio
import json
import os
import re
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (ADVERSARIAL_SET_PATH, GUARDRAILS_CONFIG_DIR,
                    LATENCY_BUDGET_P95_MS, PRESIDIO_LANGUAGE)

_PII_PATTERNS = (
    ("VN_CCCD", re.compile(r"(?<!\d)\d{12}(?!\d)"), 0.90),
    ("VN_CCCD", re.compile(r"(?<!\d)\d{9}(?!\d)"), 0.70),
    ("VN_PHONE", re.compile(r"(?<!\d)0[3-9]\d{8}(?!\d)"), 0.90),
    ("EMAIL", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), 0.95),
)
_JAILBREAK_TERMS = (
    "ignore your previous", "ignore your system", "forget your system", "system override",
    "unrestricted ai", "pretend you are dan", "do anything now", "bỏ qua tất cả",
    "ai không có giới hạn",
    "đóng vai ai không có giới hạn", "admin command", "priority 1", "previous instructions",
    "system instructions", "training data",
)
_PII_REQUEST_TERMS = (
    "cccd của", "số điện thoại của", "email của nhân viên", "thông tin cá nhân của",
    "tiết lộ thông tin", "bảng lương chi tiết", "lương của nhân viên cụ thể",
    "employee salaries", "confidential hr data", "employee records", "mật khẩu admin",
)
_OFF_TOPIC_TERMS = (
    "bài thơ", "nấu phở", "bitcoin", "ethereum", "giá cổ phiếu", "recommend phim",
    "giải phương trình", "dy/dx", "thời tiết", "tin tức", "marvel",
)
_SAFE_REFUSAL = "Xin lỗi, tôi không thể xử lý yêu cầu này theo chính sách bảo mật và phạm vi HR."


def setup_presidio():
    """Create Presidio engines when optional dependencies are installed."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[Pattern("CCCD", r"\b\d{12}\b", 0.9), Pattern("CMND", r"\b\d{9}\b", 0.7)],
    ))
    registry.add_recognizer(PatternRecognizer(
        supported_entity="VN_PHONE", patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    ))
    return AnalyzerEngine(registry=registry), AnonymizerEngine()


def _regex_entities(text: str) -> list[dict]:
    found = []
    for entity_type, pattern, score in _PII_PATTERNS:
        for match in pattern.finditer(text):
            found.append({"type": entity_type, "text": match.group(), "score": score,
                          "start": match.start(), "end": match.end()})
    return sorted(found, key=lambda item: (item["start"], item["end"]))


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    entities = _regex_entities(text)
    if analyzer is not None:
        try:
            results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            for result in results:
                entity_type = {"EMAIL_ADDRESS": "EMAIL", "PHONE_NUMBER": "PHONE_NUMBER"}.get(
                    result.entity_type, result.entity_type)
                candidate = {"type": entity_type, "text": text[result.start:result.end],
                             "score": round(float(result.score), 3), "start": result.start, "end": result.end}
                if not any(candidate["start"] == item["start"] and candidate["end"] == item["end"] for item in entities):
                    entities.append(candidate)
        except Exception as exc:
            print(f"[warn] Presidio scan unavailable; regex fallback used: {exc}")
    entities.sort(key=lambda item: (item["start"], item["end"]))
    if not entities:
        return {"has_pii": False, "entities": [], "anonymized": text}
    anonymized = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        anonymized = anonymized[:entity["start"]] + f"<{entity['type']}>" + anonymized[entity["end"]:]
    return {"has_pii": True, "entities": entities, "anonymized": anonymized}


def setup_nemo_rails():
    from nemoguardrails import LLMRails, RailsConfig
    return LLMRails(RailsConfig.from_path(GUARDRAILS_CONFIG_DIR))


def _local_input_reason(text: str) -> str | None:
    lowered = text.casefold()
    if any(term in lowered for term in _JAILBREAK_TERMS):
        return "jailbreak_or_prompt_injection"
    if any(term in lowered for term in _PII_REQUEST_TERMS):
        return "sensitive_personal_data_request"
    if any(term in lowered for term in _OFF_TOPIC_TERMS):
        return "off_topic"
    return None


def _response_text(response) -> str:
    if isinstance(response, str):
        return response
    if isinstance(response, dict):
        return str(response.get("response") or response.get("content") or "")
    messages = getattr(response, "messages", None)
    if messages:
        return str(messages[-1].get("content", "")) if isinstance(messages[-1], dict) else str(messages[-1])
    return str(response or "")


async def check_input_rail(text: str, rails=None) -> dict:
    reason = _local_input_reason(text)
    if reason:
        return {"allowed": False, "blocked_reason": reason, "response": _SAFE_REFUSAL}
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            raw = _response_text(response)
            lowered = raw.casefold()
            blocked = any(term in lowered for term in ("xin lỗi", "không thể", "i cannot", "i'm sorry"))
            return {"allowed": not blocked, "blocked_reason": "nemo_input_rail" if blocked else None,
                    "response": raw}
        except Exception as exc:
            return {"allowed": False, "blocked_reason": "nemo_input_rail_error", "response": str(exc)}
    return {"allowed": True, "blocked_reason": None, "response": ""}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    pii = pii_scan(answer)
    sensitive = any(term in answer.casefold() for term in (
        "mật khẩu hệ thống là", "thông tin bí mật", "cccd của nhân viên là",
        "số điện thoại cá nhân của", "employee salary", "confidential data",
    ))
    if pii["has_pii"] or sensitive:
        return {"safe": False, "flagged_reason": "sensitive_output", "final_answer": _SAFE_REFUSAL}
    if rails is not None:
        try:
            response = await rails.generate_async(messages=[
                {"role": "user", "content": question}, {"role": "assistant", "content": answer}
            ])
            raw = _response_text(response)
            blocked = raw.casefold() != answer.casefold() and any(term in raw.casefold() for term in ("xin lỗi", "không thể", "i cannot"))
            return {"safe": not blocked, "flagged_reason": "nemo_output_rail" if blocked else None,
                    "final_answer": raw if blocked else answer}
        except Exception as exc:
            return {"safe": False, "flagged_reason": "nemo_output_rail_error", "final_answer": _SAFE_REFUSAL}
    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(adversarial_set: list[dict], rails=None,
                          analyzer=None, anonymizer=None) -> list[dict]:
    async def run_all():
        output = []
        for item in adversarial_set:
            pii = pii_scan(item["input"], analyzer, anonymizer)
            blocked_by = "presidio" if pii["has_pii"] else None
            rail = await check_input_rail(item["input"], rails) if blocked_by is None else None
            if blocked_by is None and rail and not rail["allowed"]:
                blocked_by = "nemo_input"
            actual = "blocked" if blocked_by else "allowed"
            output.append({"id": item["id"], "category": item["category"],
                           "input": item["input"][:120], "expected": item["expected"],
                           "actual": actual, "blocked_by": blocked_by,
                           "passed": actual == item["expected"]})
        return output
    results = asyncio.run(run_all())
    print(f"Adversarial suite: {sum(item['passed'] for item in results)}/{len(results)} passed")
    return results


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(values)
    def percentile(percent: float) -> float:
        position = (len(ordered) - 1) * percent
        lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
        return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 2)
    return {"p50": percentile(0.50), "p95": percentile(0.95), "p99": percentile(0.99)}


def measure_p95_latency(test_inputs: list[str], n_runs: int = 20,
                        rails=None, analyzer=None, anonymizer=None) -> dict:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")
    inputs = test_inputs or [""]
    presidio_times, nemo_times, total_times = [], [], []
    async def measure():
        for index in range(n_runs):
            text = inputs[index % len(inputs)]
            start = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio = (time.perf_counter() - start) * 1000
            start = time.perf_counter()
            await check_input_rail(text, rails)
            nemo = (time.perf_counter() - start) * 1000
            presidio_times.append(presidio); nemo_times.append(nemo); total_times.append(presidio + nemo)
    asyncio.run(measure())
    total = _percentiles(total_times)
    return {"presidio_ms": _percentiles(presidio_times), "nemo_ms": _percentiles(nemo_times),
            "total_ms": total, "latency_budget_ok": total["p95"] < LATENCY_BUDGET_P95_MS,
            "budget_ms": LATENCY_BUDGET_P95_MS}


def run_phase_c(report_path: str = "reports/guard_results.json") -> dict:
    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as handle:
        adversarial_set = json.load(handle)
    suite = run_adversarial_suite(adversarial_set)
    latency = measure_p95_latency([item["input"] for item in adversarial_set], n_runs=20)
    report = {"adversarial_results": suite,
              "adversarial_summary": {"passed": sum(item["passed"] for item in suite), "total": len(suite)},
              "latency": latency}
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(f"Guard report saved -> {report_path}")
    return report


if __name__ == "__main__":
    run_phase_c()
