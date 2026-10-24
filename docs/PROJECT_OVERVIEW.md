# PolicyLens Project Overview

## Purpose

PolicyLens is a Python application for exploring Indian health-insurance policy
wording. It ingests PDF policy documents, indexes clause-sized text, and offers
question answering and policy-analysis endpoints with citations and
abstention-oriented behavior.

This document describes the current repository state. It distinguishes
implemented features from measured quality and from work that is not yet
complete. The project is a prototype; repository configuration alone does not
prove production deployment or production-level accuracy.

## What is implemented

### Policy ingestion

- PDF text and table extraction with PyMuPDF.
- Clause-aware chunking that retains policy, section, clause, and page metadata.
- Dense embeddings using `BAAI/bge-base-en-v1.5`.
- Local Qdrant vector indexing and a BM25 keyword index.
- CLI commands to ingest a PDF, list indexed policies, and remove registry
  metadata.
- FastAPI PDF upload endpoint for policy ingestion.
- An indexed-policy registry and serialized BM25 data under the configured
  local Qdrant storage directory.

### Retrieval and answer generation

- Dense Qdrant retrieval and BM25 retrieval.
- Reciprocal-rank fusion (RRF) combines the two ranked lists.
- A provider abstraction is present for Gemini and mock/offline operation.
- Q&A response schemas include answer text, citations, confidence, abstention
  status, caveats, and retrieved chunk IDs.
- A low-evidence fallback can cite a matching retrieved clause or abstain.
- The API appends an informational-use disclaimer to Q&A responses.

**Not implemented in the checked retrieval path:** cross-encoder reranking.
Although a reranker model name exists in configuration, the retrieval code does
not load or execute a cross-encoder.

### Policy analysis

- Pydantic response schemas and pipeline methods for summary cards, red flags,
  and side-by-side scenario comparison.
- API routes for summary and red-flag analysis on an indexed policy.
- API route for comparing two indexed policies for a scenario.
- Deterministic fallback logic exists for some missing or invalid model
  responses.

The presence of these features does not establish high factual accuracy.
Fallback outputs, especially red flags and summaries, should be verified
against the source policy.

### API and user interface

- FastAPI application with health endpoints and automatically generated API
  documentation at `/docs`.
- REST endpoints:

  | Method | Endpoint | Purpose |
  |---|---|---|
  | `POST` | `/api/v1/policies/ingest` | Upload and index a PDF |
  | `GET` | `/api/v1/policies` | List indexed policies |
  | `DELETE` | `/api/v1/policies/{policy_id}` | Remove policy registry entry |
  | `POST` | `/api/v1/qa` | Ask a policy-grounded question |
  | `GET` | `/api/v1/analyse/summary/{policy_id}` | Get a structured summary |
  | `GET` | `/api/v1/analyse/red-flags/{policy_id}` | Get potential restrictive clauses |
  | `POST` | `/api/v1/analyse/compare` | Compare two indexed policies |

- Streamlit UI with pages for PDF ingestion, Q&A, summary, red flags, and
  comparison.

### Packaging and operations

- Python package configuration and optional development/evaluation dependency
  groups are defined in `pyproject.toml`.
- Dockerfiles exist for the API and UI.
- Docker Compose defines API, UI, and Qdrant services.
- Local mode uses embedded Qdrant storage by default; the Compose setup uses a
  Qdrant server.
- Environment configuration includes fields for Gemini, OpenAI, Anthropic,
  Qdrant, and optional Langfuse settings. Gemini is the implemented live LLM
  client; OpenAI and Anthropic are not implemented by the current client
  factory, which falls back to the mock client for other provider selections.

Langfuse configuration and an optional dependency are present, but active
tracing instrumentation or collected traces have not been verified. Docker
configuration exists, but an actual deployment has not been verified.

## Current local data

At the last verification in this workspace:

- `data/sample_policy.pdf` is generated sample/test data, not an official
  insurance policy.
- `data/icici_complete_health.pdf` is a 63-page ICICI Lombard policy PDF.
- The local registry listed the ICICI policy as `icici_complete_health`, with
  188 indexed chunks. The parser reported 12 clauses for that PDF.
- The additional policies mentioned outside this workspace are not included
  in the verified local corpus.

The parser's clause count and segmentation should be manually checked for long
or complex policies; a high chunk count does not itself demonstrate complete
or correct clause extraction.

## Evaluation and test status

### Automated tests

The project test suite was run successfully:

- **35 passed**
- **1 Pydantic serializer warning** in the red-flags integration test.

The integration tests use a synthetic in-memory policy; they do not establish
live Gemini answer quality for every indexed document.

### Checked-in evaluation artifacts

- `eval/qa_set.jsonl`: 50 labeled questions for `care-adv-sample`.
- `eval/extraction_labels.json`: 5 policy label records.
- `eval/compare_cases.jsonl`: comparison scenarios.
- `eval/results/chunking_ablation.json`: chunking measurements on the generated
  four-page sample.
- `eval/results/parser_comparison.json`: parser comparison output. Unstructured
  was unavailable during the latest run, so this is not a valid head-to-head
  parser comparison.

The sample chunking evaluation reported:

| Metric | Clause-aware | Fixed-size baseline |
|---|---:|---:|
| Chunks | 13 | 9 |
| Clause-boundary preservation | 100.0% | 0.0% |
| Section-context preservation | 100.0% | 0.0% |

These are structural chunking metrics on one generated sample policy, not
retrieval or answer-quality scores.

A separate offline prototype check found BM25-only Hit@5 of 40/40 (100%) and
dense + BM25/RRF Hit@5 of 40/40 (100%) across the 40 answerable questions for
that same sample policy. The measured gain was zero percentage points. The
offline deterministic summary fallback matched 4/13 labeled fields (30.8%)
for the one matching sample-policy label. These small, single-policy prototype
checks do not represent live-model performance or the full real-policy corpus.
Details and scoring caveats are in `docs/RESUME_METRICS.md` and
`eval/results/RESUME_BULLETS.md`.

### Evaluation gaps

- RAGAS faithfulness and answer-relevance metrics have not been measured.
- No reproducible corpus-wide Hit@5 comparison or 15-policy evaluation has
  been produced.
- No live-model extraction-accuracy score has been produced.
- Cross-encoder reranking has not been implemented or evaluated.
- The Makefile's `eval` target refers to `eval.run_all`, which is absent. Thus
  `make eval` is not currently a working full-evaluation command. The individual
  scripts `eval.run_chunking_ablation` and `eval.run_parser_comparison` are
  available; `eval.run_eval` is also present but has not been established as a
  complete replacement for `run_all`.

## Known gaps and risks

1. **Reranking:** model configuration exists, but no cross-encoder execution.
2. **Evaluation completeness:** no active RAGAS runner, no measured extraction
   accuracy evaluator, and no full `run_all` module.
3. **Limited evidence scope:** the reproducible retrieval figures cover one
   generated sample policy; the external 15-PDF corpus has not been validated
   in this workspace.
4. **Fallback accuracy:** offline deterministic summary and red-flag fallbacks
   are useful for exercising the pipeline but are not substitutes for verified
   model analysis.
5. **PDF parsing:** image-only/scanned PDFs need OCR support, which is not
   included. Complex layouts and tables may need manual validation.
6. **Observability and deployment:** Langfuse instrumentation and real Docker
   deployment are not evidenced by the current repository artifacts.
7. **Red-flag response typing:** the test suite currently emits a Pydantic
   serializer warning because fallback red flags are assembled as dictionaries
   instead of `RedFlag` model instances.
8. **Policy deletion:** the current CLI/API documentation notes that deletion
   removes the registry entry but does not delete the associated Qdrant points.

## Running locally

From the repository root on Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
Copy-Item .env.example .env
# Add the selected provider's key to .env when live LLM calls are desired.
python -m app.cli ingest "path\to\policy.pdf" --policy-id policy_name --insurer "Insurer"
python -m app.cli list-policies
python -m uvicorn app.main:app --reload --port 8000
```

In a second terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m streamlit run ui/app.py --server.port 8501
```

To run the tests:

```powershell
python -m pytest -q
```

The default Q&A provider is Gemini when a Gemini key is configured; without a
key, the application falls back to its mock client. Offline/mock behavior is
appropriate for tests and local plumbing checks, not for asserting live model
quality.

## Safety note

PolicyLens is an informational prototype, not insurance, legal, or financial
advice. Users should verify every conclusion against the official policy
wording and consult a qualified advisor for coverage decisions.
