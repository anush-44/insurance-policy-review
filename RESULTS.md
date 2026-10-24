# PolicyLens Evaluation Results

Evaluation date: 2026-10-24 (Asia/Kolkata). This report records executed checks and keeps content validation, indexing, and answer-quality evaluation separate.

## Policy acquisition and ingestion

Five of six policy wording PDFs were acquired from insurer or insurer-linked HTTPS sources and validated for PDF integrity, product/UIN, selectable text, page count, and SHA-256. The sixth, `data/icici_complete_health.pdf`, passed content checks (63 pages; UIN `ICIHLIP25035V082425`; SHA-256 `b5425731688630ea194018b1c195ee6d313a27d06316d73b10811a187f81d8dc`) but its source provenance is unverified, so the manifest-driven ingestion command skipped it.

| Policy | Pages | Parsed clause units | Indexed chunks | SHA-256 |
|---|---:|---:|---:|---|
| Care Supreme | 45 | 113 | 234 | `2ed123866ffe89b0ddc455ebe867273450b1dbe1d5e3f3040427b0c1ba4d0a4a` |
| HDFC ERGO my:Optima Secure | 69 | 209 | 329 | `694c0540cb341ec9254c08a41668174b60d4a7ebc4833d78505052367c0b6ab3` |
| Niva Bupa ReAssure 2.0 | 36 | 237 | 254 | `8b00f5307e46553b955c3e21d0adea417e206a635b936f0623109fbec2f8c1bb` |
| Aditya Birla Activ One | 56 | 278 | 420 | `e94f3734458ececa858a7dd51a78e832a6db333a851fdbd3d5f330ac6b1b60dd` |
| Star Comprehensive | 48 | 186 | 237 | `b1dbe8fb78646f75566d47c32b7ebfa27c4071941c8f548224c461ee35a8021f` |

The five official documents now contribute 1,474 chunks, each embedded with the cached `BAAI/bge-base-en-v1.5` model and committed to local Qdrant plus BM25/registry files. Star initially exposed a split-line heading issue; parser support was added and Star was reingested, replacing 155 chunks with 237 and pruning 153 stale vectors. The refreshed per-policy report is `data/ingestion_reports/star-comprehensive.json`; the initial manifest run report remains a historical record of the first pass.

The existing registry also contains the pre-existing `icici_complete_health` entry with 188 chunks. It was not rebuilt or validated against the local ICICI PDF version.

## Gold QA set and extraction labels

`eval/qa_set.jsonl` now has 33 manually referenced questions across six manifest policies: 13 for the local ICICI wording (11 answerable, 2 schedule-dependent abstentions) and 4 for each of the five officially validated documents. The reference validator checked IDs, pages, and evidence text against the PDF pages. All 33 references passed. The set includes simple facts, exceptions and conditional rules, multi-reference questions, and unanswerable schedule-specific questions. It remains a small manually labeled benchmark; no answer accuracy has been measured.

`eval/extraction_labels.json` contains one ICICI policy with 7 verified fields and conditional, schedule-dependent, or unlabeled fields. Gold references pass. No extraction predictions were produced, so accuracy/precision/recall/F1 are not measured.

## Retrieval ablation

One real-embedding ablation was run for Care Supreme using its four answerable QA records (four questions; no unanswerable cases). There were no query failures. Results below are preliminary and too small to support general quality claims. The report with PDF/dataset hashes is `eval/results/retrieval_ablation_care-supreme_20261024_125400.json`.

| Chunker | Retrieval | Hit@5 / Recall@5 | MRR@5 | nDCG@5 | Median latency |
|---|---|---:|---:|---:|---:|
| Clause-aware | BM25 | 0.75 / 0.75 | 0.750 | 0.750 | 2.31 ms |
| Clause-aware | Dense | 0.75 / 0.75 | 0.425 | 0.504 | 89.04 ms |
| Clause-aware | Hybrid RRF | 0.75 / 0.75 | 0.425 | 0.504 | 95.38 ms |
| Fixed-size | BM25 | 0.75 / 0.75 | 0.750 | 0.750 | 1.91 ms |
| Fixed-size | Dense | 0.75 / 0.75 | 0.458 | 0.533 | 82.73 ms |
| Fixed-size | Hybrid RRF | 0.75 / 0.75 | 0.583 | 0.625 | 81.32 ms |

The controlled harness also supports optional cross-encoder reranking; it was not run because reranker weights are not cached. These results are one-policy diagnostics, not evidence that any retrieval configuration is generally better. Mock embeddings remain diagnostic only.

## RAGAS, answer quality, and operational checks

- RAGAS and `datasets` are not installed; `OPENAI_API_KEY` is not configured. No RAGAS judge scores were generated.
- Gemini is configured, but the live answer-quality run was not permitted: automatic approval review rejected sending local policy text and questions to the external Gemini service because that payload/destination lacked explicit approval. The request was not executed; no live result or cost is claimed.
- Reranker weights are not cached; cross-encoder quality/latency remains unverified.
- Red-flag, summary, and comparison fallbacks were updated to require evidence or abstain on unavailable facts. Real-policy response quality has not been scored.
- Docker is unavailable in the environment, so Compose was not built or started. Langfuse has no tracing integration and no traces were emitted.

## Tests and code checks

- `.venv\Scripts\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=E:\Insurance\pytest-temp`: **43 passed in 4.72 seconds** on the final run after the parser repair.
- `.venv\Scripts\python.exe -m eval.validate_eval_data`: QA and extraction schemas/references passed (33 QA questions; 7 verified extraction fields). The 20 legacy comparison cases validate structurally only and are not gold evidence.
- Targeted Ruff checks on changed/new application, data, evaluation, and test modules passed.
- `.venv\Scripts\python.exe -m compileall -q app data eval tests` completed successfully.
- `.venv\Scripts\python.exe -m app.cli list-policies` showed all five new policy IDs plus the pre-existing ICICI registry entry.

## Reproduction

```powershell
.venv\Scripts\python.exe -m data.acquire_policies
.venv\Scripts\python.exe -m data.ingest_policies
.venv\Scripts\python.exe -m eval.validate_eval_data
.venv\Scripts\python.exe -m eval.run_retrieval_ablation --policy-id care-supreme --pdf data/raw/care_supreme.pdf
```

The retrieval harness requires the configured embedding model. `--reranker` additionally requires the cross-encoder weights. RAGAS needs the optional evaluation dependencies and evaluator credentials. `README.md`, `data/README.md`, and `eval/README.md` describe the workflows and constraints.
