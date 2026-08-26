# Failure Cluster Analysis — Phase A (Online + Qdrant)

**Sinh viên:** Bùi Văn Khôi  
**Ngày:** 2026-08-26  

## 1. Aggregate RAGAS scores

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.9250 | 0.5942 | 0.6000 |
| answer_relevancy | 0.7931 | 0.6634 | 0.5038 |
| context_precision | 0.9792 | 0.8917 | 0.9417 |
| context_recall | 0.9250 | 0.7625 | 0.7167 |
| **avg_score** | **0.9056** | **0.7279** | **0.6905** |

## 2. Bottom 10 questions

| Rank | ID | Distribution | avg_score | Worst metric |
|---:|---:|---|---:|---|
| 1 | 39 | multi_hop | 0.0000 | faithfulness |
| 2 | 33 | multi_hop | 0.2083 | faithfulness |
| 3 | 48 | adversarial | 0.3125 | faithfulness |
| 4 | 43 | adversarial | 0.4167 | faithfulness |
| 5 | 50 | adversarial | 0.4167 | faithfulness |
| 6 | 44 | adversarial | 0.4583 | faithfulness |
| 7 | 9 | factual | 0.5000 | faithfulness |
| 8 | 21 | multi_hop | 0.5833 | answer_relevancy |
| 9 | 31 | multi_hop | 0.6207 | faithfulness |
| 10 | 30 | multi_hop | 0.6250 | answer_relevancy |

## 3. Failure cluster matrix

| Worst metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 2 | 10 | 4 | 16 |
| answer_relevancy | 15 | 5 | 1 | 21 |
| context_precision | 1 | 0 | 0 | 1 |
| context_recall | 2 | 5 | 5 | 12 |

**Dominant metric:** answer_relevancy (21 cases).  
**Lowest distribution score:** adversarial (0.6905).  
**Most failure-count cases:** factual (20 cases).

## 4. Root-cause analysis

Qdrant dense retrieval and reranking produced strong context precision and recall, especially on factual questions. Answer relevancy is now the dominant worst-metric cluster, while faithfulness remains the main source of the lowest individual scores. Multi-hop and adversarial questions are weaker because they require combining documents or distinguishing v2023/v2024 and v1/v2. The most difficult examples are the password comparison, trial-period benefits, Manager allowance/leave calculation and personal VPN question.

## 5. Suggested fixes

| Weak metric | Root cause | Suggested fix |
|---|---|---|
| faithfulness | Generator adds unsupported policy details | Add citations, strict grounded prompt and abstention rule |
| context_recall | Version or second-hop evidence missed | Add query expansion and version-aware parent retrieval |
| context_precision | Similar policy variants | Keep reranker and add source/version metadata filters |
| answer_relevancy | Answer misses sub-question intent | Require a direct answer for every requested field and calculation |

## 6. Adversarial distribution

Adversarial remains the weakest group despite high context precision. IDs 48, 43, 50 and 44 are in the bottom 10; they represent eligibility, password and personal-VPN policy traps. The next iteration should prioritize current-policy version metadata, contradiction checks and grounded refusal when evidence is ambiguous.
