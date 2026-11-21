# Evaluation Data and Commands

## QA data schema

Each JSONL record in `qa_set.jsonl` contains:

- `id`: unique question ID
- `policy_id`: ID from `data/policies.csv`
- `question`, `gold_answer`
- `gold_clause_ids`, `gold_pages`: paired evidence references; unanswerable items may cite a wording limitation
- `answerability`: `answerable` or `unanswerable`
- `category`, `notes`: question type and explanation of the label

Run `python -m eval.validate_eval_data` to validate JSONL, IDs, policy references, page bounds, and clause/reference text against the local PDFs. It reports errors rather than editing labels.

The current set is 57 questions across the local ICICI wording and eleven officially acquired policy wordings. The ICICI source file's provenance is unverified; the other eleven policy files passed source/content validation. There are 13 ICICI questions (11 answerable, 2 unanswerable), 32 questions across the first nine official policies, and 6 each for Ultimate Care and Ultimate Joy (4 answerable, 2 unanswerable per product). All labeled PDF-page references pass validation. This remains a small manually checked development set with no frozen held-out split. Previous synthetic records are not included in it.

## Extraction labels

`extraction_labels.json` uses `schema_version: 1`, with per-policy `fields`. Each field includes `gold_value`, `status`, gold clause/page references when verified, and notes for conditional, schedule-dependent, or unlabeled values. Current labels cover one policy and 7 verified fields; they contain no predictions.

Create predictions with a `predictions` list keyed by `policy_id`. Each field prediction has `value`, `status`, and optional `evidence` entries with `clause_id` and `page`. Score with:

```powershell
.venv\Scripts\python.exe -m eval.score_extraction --predictions path\to\predictions.json --out eval/results/extraction.json
```

Only exact normalized structured values are counted as matches. Schedule-dependent gold values require an unknown/missing prediction, not an invented cap. Single-valued metrics are reported as accuracy; no unsupported PRF1 claim is made.

## Retrieval ablation

```powershell
.venv\Scripts\python.exe -m eval.run_retrieval_ablation --policy-id care-supreme --pdf data/raw/care_supreme.pdf
```

Uses the same corpus and gold QA records for BM25-only, dense-only, hybrid RRF, hybrid plus cross-encoder, and both clause-aware/fixed-size chunking. Reports Hit@5/Recall@5, MRR@5, nDCG@5, retrieval latency, missing-evidence rate, query failures, sample sizes, data hashes, and model details. If a reranker model cannot load, its fallback status is recorded. Add `--mock-embeddings` only for plumbing diagnostics; resulting scores are not retrieval-quality evidence.

For an exploratory evaluation of the already indexed official-policy corpus, run:

```powershell
.venv\Scripts\python.exe -m eval.evaluate_indexed_retrieval
```

This loads serialized BM25 chunks and local Qdrant vectors; it does not rebuild document embeddings. It compares whole-corpus BM25, dense, and hybrid retrieval with hybrid retrieval given the correct policy ID. The last row measures a known-policy filter, not automated query routing. Output includes per-question ranks, Recall@5/10, Hit@1/5, MRR@10, deterministic bootstrap intervals, latency, and source hashes. The current questions are exploratory and must not be described as held-out performance.

## RAGAS

Install optional dependencies with `pip install -e ".[eval]"`. Run:

```powershell
.venv\Scripts\python.exe -m eval.run_ragas --policy-id icici-complete-health --out eval/results/ragas.json
```

The workflow uses Faithfulness, Answer Relevancy, Context Precision, and Context Recall when supported by the installed RAGAS API. It requires an indexed corpus, a real app LLM provider, and `OPENAI_API_KEY` for RAGAS' default judge. Evaluator calls may incur cost. Dependency, credential, and mock-provider failures stop the run without producing scores. RAGAS is not installed and has not run in this environment.

## Live Gemini answers

After explicit approval to send official-policy excerpts and evaluation questions to Gemini:

```powershell
.venv\Scripts\python.exe -m eval.run_live_answers --out eval/results/live_answers_20261121.json
.venv\Scripts\python.exe -m eval.score_live_answers --predictions eval/results/live_answers_20261121.json --review eval/results/live_answer_review_20261121.json --out eval/results/live_answer_scores_20261121.json
```

The first command checkpoints each model response and distinguishes model answers from local fallbacks. Re-running the same output path resumes failures and skips completed responses. The scorer requires a complete, explicit review-label map and checks citation quotes against the original PDFs. The included review was performed by the project assistant, not independent human annotators. Exact quote-on-cited-page rate is stricter than the prompt's near-verbatim allowance and is separate from answer faithfulness.

## Existing synthetic artifacts

`compare_cases.jsonl` and the older parser/chunking outputs predate the verified one-policy set. They are synthetic or insufficiently validated and must remain separate from real-policy results.
