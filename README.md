# PolicyLens: Grounded Health Insurance Policy Understanding & RAG

PolicyLens is a production-grade, citation-grounded RAG (Retrieval-Augmented Generation) application designed for Indian retail health insurance policies.

## Key Features
- **Strict Citation-or-Abstain**: Every factual claim cites retrieved policy clauses and exact page numbers. If evidence is missing, PolicyLens explicitly abstains.
- **Auto-Extracted Policy Summary**: Extracts waiting periods, room-rent limits, co-payments, and sub-limits with clause attribution.
- **Scenario Comparison**: Side-by-side policy comparison for medical procedures and conditions.
- **Red-Flag Analysis**: Highlights restrictive sub-limits, high co-pays, and disease waiting periods.
- **Continuous Evaluation**: Built-in benchmark suite reporting Hit Rate@5, MRR, citation precision, and RAGAS faithfulness.

## Disclaimer
> **Informational only, not insurance advice. Refer to the official policy document.** PolicyLens never predicts claim approval or provides legal/medical advice.

*Full architecture and benchmark tables will be generated during project phases.*
