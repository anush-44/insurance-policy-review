# PolicyLens Evaluation Results

Evaluation updated: 2026-11-21 (Asia/Kolkata). This report records executed checks and keeps content validation, indexing, and answer-quality evaluation separate.

## Policy acquisition and ingestion

Eleven policy wording PDFs were acquired from insurer or insurer-linked HTTPS sources and validated for PDF integrity, product/UIN, selectable text, page count, and SHA-256. The twelfth, `data/icici_complete_health.pdf`, passed content checks (63 pages; UIN `ICIHLIP25035V082425`; SHA-256 `b5425731688630ea194018b1c195ee6d313a27d06316d73b10811a187f81d8dc`) but its source provenance is unverified, so the manifest-driven ingestion command skipped it.

| Policy | Pages | Parsed clause units | Indexed chunks | SHA-256 |
|---|---:|---:|---:|---|
| Care Supreme | 45 | 113 | 234 | `2ed123866ffe89b0ddc455ebe867273450b1dbe1d5e3f3040427b0c1ba4d0a4a` |
| HDFC ERGO my:Optima Secure | 69 | 209 | 329 | `694c0540cb341ec9254c08a41668174b60d4a7ebc4833d78505052367c0b6ab3` |
| Niva Bupa ReAssure 2.0 | 36 | 237 | 254 | `8b00f5307e46553b955c3e21d0adea417e206a635b936f0623109fbec2f8c1bb` |
| Aditya Birla Activ One | 56 | 278 | 420 | `e94f3734458ececa858a7dd51a78e832a6db333a851fdbd3d5f330ac6b1b60dd` |
| Star Comprehensive | 48 | 186 | 237 | `b1dbe8fb78646f75566d47c32b7ebfa27c4071941c8f548224c461ee35a8021f` |
| Care Supreme Enhance | 49 | 329 | 359 | `cac3d77cbd7fd3e518433844c14f905598f1b0e14d844875bf66fde406e0abab` |
| HDFC ERGO Optima Restore | 53 | 156 | 253 | `0058f205f552a171befda30071681626d19b4f8f116baa986d6318b3dae66907` |
| ACKO Health III | 22 | 114 | 130 | `ee30b58696e255594e0c76d792bcb687df84bab1815c1d9d4603e2a5c0ebf5e6` |
| TATA AIG MediCare Select | 44 | 298 | 281 | `0739595ac7cb51b0e8a5edad7968d49a3185ca18b5f3754ec72e4f453049cd54` |
| Care Ultimate Care | 121 | 400 | 467 | `8fc212016eb59115e81a7bde8bc4ad1915fdc10e297f40627ace7e9c137d0b09` |
| Care Ultimate Joy | 101 | 354 | 397 | `22ab0ac6cb3d86fda1fb126f9b0d9fe804ca4b653ad7762ddb161efa0f09d331` |

The eleven official documents contribute 3,361 chunks across 644 PDF pages from seven insurers, embedded with the cached `BAAI/bge-base-en-v1.5` model and committed to local Qdrant plus BM25/registry files. Ultimate Care and Ultimate Joy were downloaded from the [insurer's policy wording downloads page](https://www.careinsurance.com/other-downloads.html); their PDFs contain UINs `CHIHLIP26058V022526` and `CHIHLIP27060V012627`. Star initially exposed a split-line heading issue; parser support was added and Star was reingested, replacing 155 chunks with 237 and pruning 153 stale vectors. Per-policy reports are in `data/ingestion_reports/`; the acquisition report is `data/acquisition_report.json`.

The existing registry also contains the pre-existing `icici_complete_health` entry with 188 chunks. It was not rebuilt or validated against the local ICICI PDF version.

## Gold QA set and extraction labels

`eval/qa_set.jsonl` has 57 manually referenced questions across all 12 manifest policies: 13 for local ICICI (11 answerable, 2 abstentions), 44 for the eleven official documents (40 answerable, 4 abstentions). IDs, page bounds, and evidence references pass validation. This remains a small, exploratory labeled set; it was not reserved as a held-out test, and answer accuracy has not been measured.

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

### Indexed-corpus retrieval ablation (exploratory)

The new `eval/evaluate_indexed_retrieval.py` evaluates the existing local BM25 and Qdrant indexes without re-embedding PDFs. It used the 40 answerable questions for the eleven verified policies (46 gold references, all represented somewhere in the index); the four official-policy abstention questions were excluded from retrieval scoring. It used real BGE query embeddings and RRF `k=60`. The JSON report, including dataset/manifest/index hashes, per-question ranks, latency, and deterministic bootstrap intervals, is `eval/results/indexed_retrieval_20261121_170414.json`.

| Setup | Hit@1 | Hit@5 | Recall@5 | Recall@10 | MRR@10 | Median retrieval latency |
|---|---:|---:|---:|---:|---:|---:|
| BM25, whole corpus | 12.5% | 35.0% | 35.0% | 55.0% | 0.238 | 20.52 ms |
| BGE dense, whole corpus | 22.5% | 37.5% | 37.5% | 50.0% | 0.298 | 129.29 ms |
| Hybrid RRF, whole corpus | 22.5% | 42.5% | 42.5% | 48.8% | 0.312 | 112.47 ms |
| Hybrid RRF, supplied policy ID filter | 30.0% | 80.0% | 77.5% | 87.5% | 0.507 | 147.18 ms |

The bootstrap 95% interval for filtered hybrid Recall@5 is **63.8–88.8%** (40 questions); for unfiltered hybrid it is **27.5–57.5%**. The filter row uses the gold policy ID supplied by the caller, not a predicted insurer or policy route. The gain therefore shows the value of known policy context, not automatic routing accuracy. The questions are often generic (for example, asking for a waiting period without naming a policy), which makes whole-corpus search ambiguous. The dense index also contains 188 legacy ICICI chunks, while BM25 in this harness includes only the eleven validated policies; that affects the unfiltered comparison. Exact evidence matching requires both the reference text and PDF page range in a chunk; repeated headers such as UINs can still make some items easier. These are exploratory observations on a small set, not the held-out benchmark in the reference specification. Latency was measured on a CPU while other local checks ran, so it is a workload observation rather than an isolated service benchmark.

In the filtered hybrid row, eight of 40 questions had no matching reference in the first five results. Three had one at ranks 8–10; five had none in the first ten. This is a rank-based error count, not a diagnosed cause. The per-question chunk IDs in the JSON report support later inspection of clause splitting, table extraction, and ranking separately.

## RAGAS, answer quality, and operational checks

- RAGAS and `datasets` are not installed; `OPENAI_API_KEY` is not configured. No RAGAS judge scores were generated.
- After the user explicitly approved sending policy excerpts and evaluation questions to Gemini, a live run used `gemini-3.5-flash-lite` on all 44 questions for the eleven verified policies. The previous auto-review rejection applied to the earlier unapproved attempt; no data was sent in that attempt.
- Reranker weights are not cached; cross-encoder quality/latency remains unverified.
- Red-flag, summary, and comparison fallbacks were updated to require evidence or abstain on unavailable facts. Red-flag precision/recall and summary/comparison quality have not been scored.
- Docker is unavailable in the environment, so Compose was not built or started. Langfuse has no tracing integration and no traces were emitted.

### Live Gemini answer-quality evaluation (exploratory)

`eval/results/live_answers_20261121.json` contains 44 model predictions, gold answers, citations, response flags, model versions, final-attempt latency, and API token usage. `eval/results/live_answer_review_20261121.json` records an explicit assistant review of each response; `eval/results/live_answer_scores_20261121.json` contains reproducible scores and per-citation checks. The review is **not** independent human annotation, and these questions were not held out from development.

| Check | Observed result | Denominator / meaning |
|---|---:|---|
| Assistant-reviewed answer correctness | 90.0% | 36/40 answerable; Wilson 95% interval 77.0–96.0% |
| Partial answers | 1 | 1/40 answerable |
| Incorrect answers | 2 | 2/40 answerable |
| False refusals | 2.5% | 1/40 answerable |
| Refusal flag accuracy | 75.0% | 3/4 unanswerable; Wilson 95% interval 30.1–95.4% |
| Strict quote found on cited PDF page | 73.6% | 39/53 citations across all 44 responses |
| Citation on one of the labeled gold pages | 60.0% | 24/40 answerable questions; an alternate page may still be valid |
| End-to-end latency, p50 / p95 | 2.65 s / 5.02 s | Final attempt per question, local retrieval plus Gemini |

The model produced 42 cited responses and two uncited abstentions. A refusal can include a citation explaining why the requested member-specific value is unavailable. The four substantive answer failures were: one false refusal on the Care road-ambulance question, one Niva grace-period duration missed after retrieval, one TATA waiting-period duration missed despite a cited response, and one partial TATA ambulance answer that omitted conditions. On the fourth unanswerable question, Gemini correctly pointed to the Policy Schedule but set `is_abstention=false`; this is the refusal-flag failure. Fourteen of 53 citations failed the **strict exact quote on cited page** check: 11 quotes were found elsewhere in the same PDF, and three were not exact substrings anywhere in that PDF after Unicode and alphanumeric normalization. This check is a conservative source-verification measure, not a full faithfulness score.

The 44 recorded successful responses used 62,349 input tokens and 9,010 output tokens. At Google's [published Gemini 3.5 Flash-Lite standard paid rates](https://aistudio.google.com/docs/pricing#gemini-3.5-flash-lite) of $0.30 and $2.50 per million input/output tokens, that is an **estimated $0.0412 total, or $0.00094 per recorded answer**. This is a rate-based estimate, not a billing statement; failed attempts and separate connection tests are not included. There were 51 API attempts for the final 44 records because transient 503/429 responses required retries.

## Tests and code checks

- The October full-suite run passed 43 tests in 4.72 seconds after the parser repair. On 2026-11-21, the 33 unit tests passed in 2.03 seconds. The full suite was started again but stalled at the first FastAPI `TestClient` health test; a minimal plain FastAPI `TestClient` reproduction also stalled in this environment. No new full-suite pass is claimed.
- `.venv\Scripts\python.exe -m eval.validate_eval_data`: QA and extraction schemas/references passed (57 QA questions across 12 policy IDs; 7 verified extraction fields). The 20 legacy comparison cases validate structurally only and are not gold evidence.
- `.venv\Scripts\python.exe -m pytest tests/unit -q -p no:cacheprovider --basetemp=E:\Insurance\pytest-unit-final-20261121`: **34 passed in 2.30 seconds**, including a regression check for Gemini's authentication header and structured JSON Schema request.
- Targeted Ruff checks on changed/new application, data, evaluation, and test modules passed.
- `.venv\Scripts\python.exe -m compileall -q app data eval tests` completed successfully.
- `.venv\Scripts\python.exe -m app.cli list-policies` showed all eleven official policy IDs plus the pre-existing legacy ICICI registry entry. Ultimate Care and Ultimate Joy completed indexing on 2026-11-21; their per-policy ingestion reports record 467 and 397 chunks respectively.

## Reproduction

```powershell
.venv\Scripts\python.exe -m data.acquire_policies
.venv\Scripts\python.exe -m data.ingest_policies
.venv\Scripts\python.exe -m eval.validate_eval_data
.venv\Scripts\python.exe -m eval.run_retrieval_ablation --policy-id care-supreme --pdf data/raw/care_supreme.pdf
$env:HF_HUB_OFFLINE='1'; .venv\Scripts\python.exe -m eval.evaluate_indexed_retrieval
.venv\Scripts\python.exe -m eval.run_live_answers --out eval/results/live_answers_20261121.json
.venv\Scripts\python.exe -m eval.score_live_answers --predictions eval/results/live_answers_20261121.json --review eval/results/live_answer_review_20261121.json --out eval/results/live_answer_scores_20261121.json
```

The retrieval harness requires the configured embedding model. `--reranker` additionally requires the cross-encoder weights. RAGAS needs the optional evaluation dependencies and evaluator credentials. `README.md`, `data/README.md`, and `eval/README.md` describe the workflows and constraints.

## Reference-project completion check

The supplied reference numbers are illustrative targets, not observed PolicyLens results. Current measured status:

| Area | Reference | PolicyLens now |
|---|---|---|
| Corpus | 10–12 policies; 6–8 insurers; 3,000–4,500 chunks; one table-heavy/scanned PDF | 11 verified policies; 7 insurers; 3,361 chunks; no scanned-PDF/OCR validation |
| Labeled QA | 120–150; dev/test split; ~20% unanswerable | 57 source-checked questions; no frozen held-out split; 6 unanswerable (10.5%) |
| Retrieval | Five-way held-out ablation through reranker | Four indexed-corpus setups measured on 40 official-policy answerable questions; filtered hybrid Recall@5 77.5% (95% bootstrap interval 63.8–88.8%); no reranker measurement |
| Chunking | Fixed, overlap, clause-aware, parent-context ablations | Single-policy, four-question fixed versus clause-aware diagnostic only; parent-section context exists in chunks but no measured parent-context ablation |
| Generation/safety | Correctness, faithfulness, citation, refusal, red-flag metrics | 44 live Gemini answers scored with assistant review; 36/40 correct, 3/4 refusal flags correct, 39/53 strict quote-on-cited-page checks pass; no independent human review, faithfulness judge, or red-flag labels |
| Engineering | 80–120 tests; Docker, eval CI gate, deployed demo | Last full test run: 43 passed in October; Docker not available here; no real-corpus eval CI gate or deployed demo verified |

The next evidence-producing steps are to expand and freeze a manually reviewed QA set, run held-out retrieval/chunking/reranker comparisons, then score generated answers and citations with human review. No target percentage above is presented as an achieved result.
