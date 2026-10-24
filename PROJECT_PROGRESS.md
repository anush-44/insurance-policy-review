# PolicyLens Project Progress

## Starting state and preservation

- Audited the repository, current Git status, local changes, parser/retrieval paths, configuration, tests, and evaluation artifacts before editing.
- Preserved the user's pre-existing edits in `app/cli.py`, `app/rag/pipeline.py`, `eval/results/parser_comparison.json`, `data/icici_complete_health.pdf`, `docs/PROJECT_OVERVIEW.md`, `docs/RESUME_METRICS.md`, and `eval/results/RESUME_BULLETS.md`.
- Replaced synthetic/unverified QA labels with manually inspected PDF-grounded questions. Legacy comparison scenarios remain labeled structurally valid only, not as gold evidence.

## Completed implementation and executed work

### Acquisition and ingestion

- Added an HTTPS source allowlist, retries, PDF/product/UIN/text/page validation, SHA-256, no-overwrite behavior, manifest updates, and acquisition reports.
- Acquired and validated five insurer or insurer-linked policy PDFs. Their hashes, UINs, and page counts are in `data/policies.csv` and `data/acquisition_report.json`.
- The local ICICI PDF passes content checks but source provenance remains unverified. The manifest-driven ingestion correctly skipped it.
- Indexed the five validated policies with cached BGE embeddings. After a parser repair and safe reingestion of Star, counts are: Care Supreme 234 chunks; HDFC Optima Secure 329; Niva ReAssure 2.0 254; Aditya Activ One 420; Star Comprehensive 237 (1,474 total).
- The old local ICICI registry entry remains at 188 chunks and was not revalidated against the local PDF version.
- Star's split-line numbered headings were initially missed. Parser support was added, raising detection from 14 to 186 clause units; reingestion replaced the old vectors and pruned 153 stale chunks.

### Parsing, indexing, retrieval, and evidence handling

- Added single-level/nested clause recognition, parser UIN and selectable-text metadata, and page spans for fixed-size chunks derived from clause overlap.
- Added rollback-capable Qdrant/BM25/registry persistence, stale-vector pruning, policy vector deletion, safe policy IDs, and per-policy ingestion reports.
- Added real CrossEncoder reranking after hybrid candidate retrieval, configurable candidate depth/disable behavior, and observable fallback status. Reranker weights are not cached and quality remains unverified.
- Added selectable BM25, dense, and hybrid retrieval modes plus a real-embedding retrieval/chunking ablation harness. Ran the six BM25/dense/hybrid × clause-aware/fixed-size configurations on four answerable Care Supreme questions. Hit@5 was 0.75 across all; MRR@5 ranged 0.425–0.75. This is a preliminary small-sample result, with no general improvement claim. Cross-encoder reranking was not run because its weights are not cached.
- Red-flag fallback emits typed evidence-backed records. Summary fallbacks leave unsupported fields unavailable, and comparison fallbacks abstain rather than rank by keyword overlap.

### Evaluation datasets and workflows

- `eval/qa_set.jsonl` has 33 questions across six manifest policies: 13 local ICICI questions (11 answerable, 2 unanswerable) and 4 questions for each of five officially validated policies. All PDF-page/reference checks pass. This is a small manually checked benchmark, not a general accuracy claim.
- `eval/extraction_labels.json` has 7 verified fields for one policy, plus conditional, schedule-dependent, and unlabeled fields. References pass; predictions and accuracy are not available.
- Added extraction scoring and optional RAGAS workflows. RAGAS/`datasets` are absent and `OPENAI_API_KEY` is not configured, so no judge scores were generated.
- Gemini is configured, but automatic approval review rejected the live run because it would send local policy text and questions to an external service without explicit approval of that payload and destination. The run was not executed; no live result or API cost is claimed.

### Deployment, documentation, and observability

- Updated Compose metadata persistence, health checks, and Langfuse environment passthrough. Docker is unavailable; Compose was not built or started.
- Langfuse has configuration variables but no tracing calls in the retrieval/generation path. No traces were emitted.
- Rewrote README and data/evaluation documentation to reflect actual workflows and limitations. Added `RESULTS.md`, `RESUME_BULLETS.md`, and this progress log. Fixed the Makefile evaluation target.

## Verification

- `.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=E:\Insurance\pytest-temp` — **43 passed in 6.36 seconds**.
- `.venv\Scripts\python.exe -m eval.validate_eval_data` — all 33 QA references and 7 verified extraction references passed; 20 comparison cases pass schema only.
- Targeted Ruff checks on changed/new modules passed.
- `.venv\Scripts\python.exe -m compileall -q app data eval tests` completed successfully.
- `.venv\Scripts\python.exe -m app.cli list-policies` verified the five newly ingested entries and the pre-existing ICICI entry.

## Remaining evaluation and environment gaps

1. Expand real-embedding retrieval/chunking ablations across the other validated policies; reranker is unavailable locally.
2. Investigate Star's low clause detection before using it for citation-sensitive evaluation.
3. Produce structured extraction predictions and score them; add manually verified labels for additional policies if needed.
4. Add evaluated RAG answer outputs; RAGAS needs dependencies and an OpenAI judge key.
5. Build/run Compose when Docker is available, and implement/test actual Langfuse tracing if observability remains in scope.
