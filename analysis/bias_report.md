# LLM Judge Bias Report — Phase B (Online + Qdrant)

**Sinh viên:** Bùi Văn Khởi  
**Ngày:** 2026-08-26  
**Judge model:** gpt-4o-mini

## 1. Pairwise samples

| # | Question | Final winner | Pass 1 / Pass 2 |
|---:|---|---|---|
| 1 | Nghỉ bao nhiêu ngày khi kết hôn? | B | B / B |
| 2 | Hạn mức bảo hiểm PVI? | B | B / B |
| 3 | Phụ cấp ăn trưa? | B | B / B |
| 4 | Mentor và buddy có thể cùng người? | tie | B / tie |
| 5 | Thiết bị 55 triệu cần ai duyệt? | B | B / B |

The complete 10-sample output, scores and reasoning are stored in `reports/judge_results.json`.

## 2. Swap-and-average

| Metric | Result |
|---|---:|
| Total judged | 10 |
| Position-inconsistent cases | 1 |
| Position bias rate | 10.0% |
| Decisive cases | 9 |

The second-pass winner was mapped back to the original A/B coordinate system. One case changed after swapping and became an inconclusive tie.

## 3. Cohen's κ

Human labels were loaded for IDs 1, 5, 12, 21, 23, 29, 33, 41, 46 and 50. Online judge labels were `[1, 0, 0, 1, 0, 0, 1, 0, 1, 1]`; human labels were `[1, 0, 1, 1, 1, 0, 1, 0, 1, 0]`.

**Cohen's κ: 0.2308 — fair/slight agreement.**

The judge should support triage and regression analysis, not act as the only production quality gate. Continue calibration with a larger balanced human-labeled set.

## 4. Verbosity bias

| Measure | Count |
|---|---:|
| A wins and A is longer | 0 |
| B wins and B is longer | 9 |
| Total decisive | 9 |
| Verbosity bias rate | 100.0% |

The judge preferred the longer answer in every decisive case. This is a material risk because retrieved answers may include source context that is verbose but not more correct. Keep accuracy, completeness and conciseness as separate criteria, and require a short final conclusion.

## 5. Overall assessment

Swap-and-average reduced positional uncertainty to 10%, below the 30% warning threshold. Agreement with human labels is only slight/fair (κ=0.2308), so human review remains necessary. The high verbosity signal should be monitored with length-matched answer pairs and a judge prompt that explicitly penalizes irrelevant context.
