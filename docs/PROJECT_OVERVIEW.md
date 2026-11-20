# PolicyLens Project Overview

## Purpose

PolicyLens is a Python prototype for grounded exploration of Indian
health-insurance policy wording. It parses policy PDFs, indexes clause-level
chunks, combines dense and keyword retrieval, and exposes question answering,
summaries, red flags, and policy comparison through FastAPI and Streamlit.

This overview describes repository state and verified evidence. It does not
turn configuration or code presence into a claim about live answer accuracy,
deployment, or production readiness.

## Implemented capabilities

### Ingestion

- PyMuPDF extraction for selectable text and tables.
- Clause-aware chunking plus a fixed-size chunker for controlled evaluation.
- UIN and page-span metadata, policy/insurer metadata, and per-policy ingestion
  reports.
- A deterministic policy acquisition workflow with an HTTPS source allowlist,
  retries, PDF/UIN/text/page validation, SHA-256 hashes, and no-overwrite
  behavior.
- CLI and FastAPI ingestion, policy listing, and policy deletion.
- Local Qdrant storage, a serialized BM25 index, and an indexed-policy registry.
- Re-ingestion replaces stale Qdrant points and BM25/registry data with
  rollback-oriented cleanup.

### Retrieval and generation

- Dense retrieval with `BAAI/bge-base-en-v1.5`.
- BM25 keyword retrieval.
- Reciprocal-rank fusion (RRF) for dense and keyword results.
- Optional `BAAI/bge-reranker-base` cross-encoder reranking.
- Cross-encoder errors fall back to RRF order and record an observable status.
- Gemini generation and an explicit mock provider.
- Configurable OpenAI and Anthropic fields exist, but the current LLM factory
  only implements Gemini and mock. Unsupported providers fail visibly.
- Q&A schemas support clause/page/quote citations, confidence, abstention,
  caveats, and retrieved chunk IDs.
- Summary, red-flag, and comparison pipelines are evidence-oriented:
  unsupported values abstain or remain unavailable, and source evidence is
  required where claims are returned.

### API and UI

- FastAPI routes for ingestion, listing, deletion, Q&A, summaries, red flags,
  and side-by-side comparison.
- Interactive OpenAPI documentation at `/docs`.
- Streamlit pages for ingestion, Q&A, summaries, red flags, and comparison.
- Liveness health endpoints for API/container use.
- Dockerfiles and Docker Compose configuration for API, UI, and Qdrant.

## Local corpus and registry

The repository manifests and/or validates five insurer-hosted or
insurer-linked policy PDFs plus one pre-existing local ICICI PDF:

- Care Supreme
- HDFC ERGO my:Optima Secure
- Niva Bupa ReAssure 2.0
- Aditya Birla Activ One
- Star Comprehensive
- ICICI Lombard Complete Health, with unverified local acquisition provenance

The verified official documents account for 1,474 indexed chunks in the local
evaluation state. The separate ICICI entry has 188 chunks and was not rebuilt
as part of the manifest ingestion. Policy acquisition and ingestion reports are
kept under `data/` and should be checked before citing document provenance.

## Evaluation status

### QA dataset

`eval/qa_set.jsonl` contains 33 manually referenced questions across the local
ICICI document and five validated policy documents. PDF and page-reference
validation passes. This is a small benchmark and not evidence of general answer
accuracy.

The extraction-label file contains 7 verified fields for the local ICICI policy.
Its references pass, but no live extraction predictions or measured accuracy
have been produced.

### Retrieval

A real-embedding, controlled retrieval/chunking ablation was run for Care
Supreme over four answerable questions. Across BM25, dense, hybrid RRF,
clause-aware chunks, and fixed-size chunks:

- Hit@5: 0.75 for all evaluated configurations
- MRR@5: 0.425–0.750
- nDCG@5: 0.504–0.750

These are one-policy diagnostic results. They do not support a claim of
improvement across the full corpus. Cross-encoder reranking was not included
because reranker weights are not cached locally.

### Tests

The latest baseline suite ran successfully with:

- **43 tests passed**
- Pydantic and parser paths covered
- FastAPI TestClient integration coverage
- Hybrid retrieval and reranking fallback unit coverage

The test suite uses mock or deterministic components in many cases and does not
prove live Gemini quality or real-policy factual accuracy.

### Not measured

- RAGAS faithfulness, answer relevance, context precision, and context recall
- Live Gemini answer accuracy
- Extraction field accuracy and field-level precision/recall
- 15-policy or corpus-wide retrieval improvement
- Cross-encoder reranking quality
- Langfuse trace volume or latency
- Docker Compose build/start/health behavior
- Real deployment uptime or scaling

## Operational gaps

1. RAGAS requires optional dependencies and an evaluator API key.
2. Live Gemini evaluation requires explicit approval for sending local policy
   text and questions to the external service.
3. Langfuse settings and Compose environment passthrough exist, but no active
   tracing calls are implemented.
4. Docker and Docker Compose were unavailable during verification.
5. OCR is not implemented for scanned or image-only PDFs.
6. The Star parser repair and re-ingestion improved clause detection, but
   citation-sensitive quality across every policy still needs manual review.
7. Summary field-level citations and end-to-end answer scoring remain areas for
   future improvement.

## Run locally

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `LLM_PROVIDER=mock` for explicit offline operation, or configure Gemini
for live generation:

```dotenv
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
```

Start the API and UI:

```powershell
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --reload --port 8000
.\\.venv\\Scripts\\python.exe -m streamlit run ui/app.py
```

Validate the project and evaluation assets:

```powershell
.\\.venv\\Scripts\\python.exe -m pytest tests/ -q -p no:cacheprovider --basetemp=E:\Insurance\pytest-temp
.\\.venv\\Scripts\\python.exe -m eval.validate_eval_data
```

## Safety note

PolicyLens is informational only. It is not insurance, legal, or financial
advice. Every answer, red flag, summary, and comparison must be checked against
the official policy wording and a qualified adviser where coverage or claim
decisions matter.
