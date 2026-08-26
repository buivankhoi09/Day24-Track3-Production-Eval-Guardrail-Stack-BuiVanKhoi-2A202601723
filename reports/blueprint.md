# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh viên:** Bùi Văn Khôi  
**Ngày chạy online:** 2026-08-26  

## Guard Stack Architecture

`Input → PII scan → input rail → Day 18 RAG → output rail → response`

| Layer | Tool | Latency P95 | Failure action |
|---|---|---:|---|
| PII detection | Presidio-compatible VN regex layer | 0.01 ms | Reject and log |
| Topic/jailbreak | Input rail rules | 0.01 ms | Reject with reason |
| RAG pipeline | Day 18 BM25 + dense Qdrant + reranker + GPT-4o-mini | online | Fallback / retry |
| Output check | PII and sensitive-output rail | local check | Redact and log |

## Online Evaluation Results

- Answers: **50/50** generated with OpenAI API after indexing 116 chunks in Qdrant.
- Retrieval: BM25 + dense embedding + reranker completed successfully; M5 enrichment was disabled to avoid 116 extra API calls.
- RAGAS: **200 online metric evaluations** completed.
- Adversarial suite: **20/20 passed (100%)**.
- Guard P50/P95/P99: **0.01 / 0.02 / 0.03 ms**.
- Cohen's κ: **0.4000** against the 10 human labels.

| Distribution | Faithfulness | Answer relevancy | Context precision | Context recall | Avg |
|---|---:|---:|---:|---:|---:|
| factual | 0.9250 | 0.7931 | 0.9792 | 0.9250 | 0.9056 |
| multi_hop | 0.5942 | 0.6634 | 0.8917 | 0.7625 | 0.7279 |
| adversarial | 0.6000 | 0.5038 | 0.9417 | 0.7167 | 0.6905 |

## CI/CD Gates

```yaml
- name: Tests
  run: pytest tests/ -q
- name: RAGAS quality gate
  run: python src/phase_a_ragas.py
- name: Guardrail gate
  run: python src/phase_c_guard.py
```

Merge gates: all tests pass, faithfulness ≥ 0.75, adversarial pass rate ≥ 90%, and total guard P95 < 500 ms. This run passes the guardrail and latency gates; factual faithfulness passes, while multi-hop and adversarial faithfulness need improvement.

## Monitoring

Alert on faithfulness < 0.70, adversarial pass rate < 80%, guard P95 > 600 ms, or PII detections >10/hour. Store IDs, rail reason, policy version, latency and redaction outcome; never log raw PII.

## Production assessment

Enabling Qdrant, dense retrieval and reranking improved context precision/recall compared with the BM25-only run. The remaining weakness is faithful handling of multi-hop and version-conflict questions. Strengthen version-aware filters, require citations or source snippets in the answer prompt, and consider enabling M5 enrichment after measuring its quality gain against its API cost.
