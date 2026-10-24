# PolicyLens Resume Bullets — Sample Evaluation

Metrics below are taken from the current JSON files in this directory. The
available artifacts evaluate the generated four-page `sample_policy.pdf`;
they do not establish performance on 15 real insurance policies.

## Bullet templates with measured values

- Built an end-to-end RAG system over **[NOT MEASURED]** insurance policy PDFs
  with clause-aware chunking; on the four-page sample PDF, clause-boundary and
  section-context preservation were both **100.0%**. The implementation has
  dense + BM25 hybrid search, but Hit@5 improvement from **[NOT MEASURED]** to
  **[NOT MEASURED]** was not reported in the evaluation JSON. Cross-encoder
  reranking was **not implemented in the checked retrieval path**.
- Developed Pydantic-based structured extraction, policy comparison, and
  red-flag detection with citation-grounded answers and abstention guardrails;
  field accuracy on a hand-labeled set: **[NOT MEASURED]**.
- RAGAS faithfulness: **[NOT MEASURED]**; RAGAS answer relevance:
  **[NOT MEASURED]**. A FastAPI REST API and Docker Compose configuration are
  present, but deployment and active Langfuse tracing are **not measured or
  evidenced by these result files**.

## Other measured sample results

From `chunking_ablation.json`:

| Metric | Clause-aware | Fixed-size baseline |
|---|---:|---:|
| Chunks produced | 13 | 9 |
| Clause-boundary preservation | 100.0% | 0.0% |
| Section-context preservation | 100.0% | 0.0% |

These are the evaluation script's structural preservation metrics on a single
generated sample PDF, not retrieval Hit@5 or end-to-end answer quality.

From `parser_comparison.json`, PyMuPDF processed the four-page sample in
**1.74 seconds**, produced **13 clauses**, detected **1 table**, and recorded a
**1.0** preservation score. The Unstructured parser was unavailable, so this
does not establish a comparative parser win.

## Bullets not ready to use as originally claimed

1. **15 PDFs and Hit@5 improvement:** corpus size and Hit@5 are absent from the
   current JSON results. Cross-encoder reranking is not implemented in the
   checked retrieval path.
2. **Z% extraction accuracy:** no measured extraction-accuracy result appears
   in the JSON files.
3. **RAGAS and deployed/traced claims:** faithfulness, answer relevance, actual
   Docker deployment, and active Langfuse traces are not established by the
   JSON files. The repository contains configuration, not runtime evidence.

Do not replace any **[NOT MEASURED]** value with an estimate. Re-run an
appropriate evaluation and save its results before using those claims.
