# PolicyLens: Project and Implementation Guide

## What this project does

PolicyLens is a Python prototype for exploring Indian health-insurance policy
wordings. It turns selectable PDF text into structured clauses and searchable
chunks, retrieves relevant passages using dense and keyword search, and offers
question answering and policy analysis through a FastAPI service and a
Streamlit interface.

The application is intended to help users find and understand wording in a
policy document. It is not insurance, legal, or financial advice. Answers and
summaries must be checked against the insurer's official wording.

This guide describes code that exists in the repository. A feature being
implemented does not mean it has been validated for every policy, deployed, or
proven accurate in live use.

## End-to-end flow

```text
Policy wording PDF
    |
    +--> Acquisition and content validation (optional manifest workflow)
    |
    +--> PyMuPDF text/table extraction
            |
            +--> Sections, clauses, UIN and page metadata
                    |
                    +--> Clause-aware chunks
                            |
                            +--> BGE text embeddings --> Qdrant vector store
                            |
                            +--> BM25 token index --> serialized local JSON
                                    |
                                    +--> dense search + BM25 search
                                              |
                                              +--> Reciprocal Rank Fusion
                                                      |
                                                      +--> optional CrossEncoder
                                                              |
                                                              +--> RAG operations
                                                                      |
                                                                      +--> Gemini or explicit mock LLM
                                                                              |
                                                                              +--> FastAPI --> Streamlit
```

In the normal API path, the application loads a shared RAG pipeline at startup.
The vector collection is managed by Qdrant; BM25 chunks and the policy
registry are stored as local JSON files under `QDRANT_LOCAL_PATH`.

## Implemented capabilities

### Policy acquisition and local data

- `data/policies.csv` is the policy manifest. It records insurer, product, UIN,
  source URL, page count, local path, download date, SHA-256, and validation
  status.
- `python -m data.acquire_policies` downloads PDFs only from HTTPS insurer
  hosts configured in an allowlist. It validates PDF readability, selectable
  text, product-name evidence, page count, and UIN where available.
- Downloads use temporary files, retry transient failures, check redirected
  hosts, record hashes, and avoid replacing an existing different version
  unless `--refresh` is specified.
- Locally supplied PDFs can be recognized as content-valid while their source
  provenance remains explicitly unverified.
- `python -m data.ingest_policies` ingests only validated manifest rows by
  default. `--include-unverified-local` is required to include a local PDF
  whose source provenance is unverified.
- The acquisition and ingestion commands write JSON reports. Per-policy
  ingestion reports include page, clause, and chunk counts, parser warnings,
  and status.
- `data/create_sample_pdf.py` generates a synthetic sample PDF for tests and
  parser demonstrations; it is not an official policy document.

The manifest currently describes eleven insurer-hosted or insurer-linked PDFs
with validated source/content information and one local ICICI Lombard PDF
whose source provenance is unverified. The `data/raw/` PDFs and
`data/ingestion_reports/` are Git-ignored local data; they are not guaranteed
to be present in a fresh clone. The checked-in `data/icici_complete_health.pdf`
also has unverified download provenance.

### PDF parsing and data model

- `PyMuPDFParser` extracts selectable text page by page and attempts table
  extraction with PyMuPDF's table finder.
- Tables are represented as structured rows and headers and also rendered as
  Markdown for indexing.
- The parser detects common numbered clause, section, annexure, schedule, and
  lettered heading patterns, including some headings whose number and title
  appear on separate lines.
- It tracks page spans, parent section, clause identifiers, table data, and
  insurer policy UIN when detected in the first pages.
- Clause types are assigned using insurance-specific text heuristics:
  definition, coverage, exclusion, waiting period, sub-limit, table, claims,
  or other.
- If recognizable structure is missing, a page-oriented fallback creates
  clauses from available page text.
- Pydantic models in `app/core/schemas.py` define policy documents, sections,
  clauses, table blocks, searchable chunks, citations, and grounded answers.
- `UnstructuredParser` is an optional parser used by the parser comparison
  evaluation. The optional `unstructured` dependency is not required for the
  production PyMuPDF ingestion path.

Scanned/image-only documents are not OCR-processed. Extraction quality depends
on the text and tables available in the PDF.

### Chunking and indexing

- `ClauseAwareChunker` is the ingestion chunker. It preserves clause and
  section context, merges short clauses, and splits long clauses into
  overlapping windows. Chunk metadata includes policy ID, insurer, clause,
  page span, chunk type, UIN, and detected cross-references.
- `FixedSizeChunker` creates fixed character windows with page-span metadata.
  It is an evaluation baseline, not the default ingestion chunker.
- `BGEEmbeddingModel` wraps Sentence Transformers and the configured BGE model.
  BGE queries receive the model's retrieval instruction prefix, and vectors
  are normalized by default.
- `DeterministicMockEmbeddingModel` provides repeatable pseudo-vectors for
  tests and diagnostic workflows; it is not suitable evidence of retrieval
  quality.
- `QdrantVectorStore` supports local embedded storage, a Qdrant server, and an
  in-memory mode used by tests/evaluation. It stores dense vectors and chunk
  payloads and filters searches by policy and chunk type.
- Chunk IDs are mapped to deterministic Qdrant point IDs. The vector store
  supports policy deletion, snapshots/restoration for rollback, and removal
  of stale chunks after re-ingestion.
- `BM25Index` tokenizes insurance text while retaining useful codes and clause
  numbers. It supports policy and chunk-type filters.
- BM25 chunks are serialized per policy and reloaded at API startup. The
  policy registry stores identifying metadata and index counts.
- Ingestion validates policy IDs, refuses empty chunk output, writes BM25 and
  registry files through temporary files, and attempts to restore previous
  Qdrant and file state if persistence fails. This is rollback-oriented
  handling, not a distributed transaction.

### Retrieval and ranking

- Retrieval mode is configurable as `hybrid`, `dense`, or `bm25`.
- Dense search uses cosine similarity in Qdrant; keyword search uses BM25.
- Hybrid mode merges ranked results with Reciprocal Rank Fusion (RRF), using a
  configurable smoothing constant.
- Optional CrossEncoder reranking scores a candidate pool and returns the
  highest-ranked chunks. It can be disabled; if it is enabled but cannot load
  or score, the retriever logs an observable fallback status and preserves
  RRF order.
- Search results can be restricted to one policy or selected clause types.

### RAG operations and evidence handling

`RAGPipeline` implements four application operations:

1. **Question answering** retrieves policy passages and asks the configured
   LLM for a `GroundedAnswer`: answer text, clause/page/quote citations,
   confidence, abstention state, caveats, and retrieved chunk IDs. Prompts
   direct the model to use supplied context only and to abstain when evidence
   is missing. If the model fails or returns no usable citations, the pipeline
   has a short retrieved-text fallback or abstains when keyword overlap is
   insufficient. The API appends the informational disclaimer.
2. **Policy summary** retrieves a broad set of typed policy clauses and asks
   for structured values such as waiting periods, room-rent limits, co-pay,
   restoration, no-claim bonus, exclusions, sub-limits, and claim process.
   The response schema has no field-level citations; extracted values should
   be independently verified.
3. **Red-flag scan** retrieves exclusions, waiting periods, and sub-limits,
   then returns candidate restrictive clauses with severity, page, and
   evidence. A deterministic text-based fallback is used if no flags are
   returned by generation. Results are not a substitute for reviewing the
   source wording.
4. **Policy comparison** retrieves evidence separately from two policies for
   a scenario, requests structured comparison points, and checks whether
   returned clause/page references match retrieved passages. Missing support
   is marked as insufficient evidence rather than treated as a supported
   comparison.

Prompts and response models live in `app/rag/prompts.py` and
`app/rag/schemas.py`. Pydantic schemas define the API response shape.

### LLM providers

- `GeminiLLMClient` calls the Gemini REST API and supports JSON-schema-based
  structured generation. It requires `GEMINI_API_KEY`.
- `MockLLMClient` provides deterministic, offline behavior for tests and
  development. Mock output is not evidence of answer quality.
- The configured LLM provider defaults to Gemini; setting
  `LLM_PROVIDER=mock` explicitly selects offline/mock behavior.
- OpenAI and Anthropic keys/settings are present for configuration
  compatibility, but their provider clients are not implemented. Selecting
  them fails visibly.

### HTTP API

The FastAPI application is defined in `app/main.py`. OpenAPI documentation is
served at `/docs` and ReDoc at `/redoc`.

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Service status, name, version, and disclaimer |
| `GET` | `/health` | Liveness response for container orchestration |
| `POST` | `/api/v1/policies/ingest` | Upload a PDF and index it |
| `GET` | `/api/v1/policies` | List indexed policy registry entries |
| `DELETE` | `/api/v1/policies/{policy_id}` | Remove policy vectors, BM25 data, and registry entry |
| `POST` | `/api/v1/qa` | Answer a question, optionally restricted to a policy |
| `GET` | `/api/v1/analyse/summary/{policy_id}` | Extract a structured policy summary |
| `GET` | `/api/v1/analyse/red-flags/{policy_id}` | Scan a policy for restrictive clauses |
| `POST` | `/api/v1/analyse/compare` | Compare two policies for a scenario |

The ingestion endpoint accepts multipart PDF data and optional policy ID and
insurer fields. The QA endpoint validates a minimum question length and
`top_k` range. Analysis endpoints require policies to exist in the registry.
Routes return typed Pydantic response objects and HTTP errors for invalid,
missing, or failed operations.

### Streamlit interface

`ui/app.py` provides a browser UI with five screens:

- **Ingest Policy:** upload a PDF, optionally specify its ID and insurer, and
  display page/chunk counts after indexing.
- **Ask Questions:** choose an indexed policy, enter a question, select the
  retrieval context count, and view answer, confidence, abstention, clause
  citations, quotes, and caveats.
- **Policy Summary:** display the extracted waiting periods, room rent, co-pay,
  restoration, no-claim bonus, sub-limits, exclusions, and claim process.
- **Red Flags:** display restrictive clauses grouped by severity.
- **Compare:** select two policies and a preset or custom scenario to view
  retrieved comparison points and the overall evidence-limited verdict.

The interface calls the FastAPI endpoints over HTTP and includes an
informational disclaimer. Its current API base URL is hard-coded as
`http://localhost:8000`; although Compose supplies an `API_URL` value to the
UI container, the current UI code does not read that variable. Container
networking for the UI-to-API call therefore needs verification/follow-up.

### Evaluation and quality checks

- `eval/qa_set.jsonl` contains 57 manually referenced questions across twelve
  policy IDs in the current local dataset: 51 answerable and 6 unanswerable.
- `eval/extraction_labels.json` contains seven verified field references for
  one policy; it does not contain model predictions.
- `eval/compare_cases.jsonl` contains 20 comparison scenarios. Some older
  comparison/parser/chunking artifacts are synthetic or not sufficiently
  validated and must not be presented as real-policy quality evidence.
- `python -m eval.validate_eval_data` validates dataset structure and checks
  labeled page/clause references against local PDFs.
- `python -m eval.run_eval` runs the RAG pipeline against the QA set and
  reports citation recall, abstention accuracy, confidence, non-empty answers,
  and latency. It supports explicit mock mode for plumbing/CI use.
- `python -m eval.run_retrieval_ablation` compares BM25-only, dense-only, and
  hybrid RRF retrieval with clause-aware and fixed-size chunking. It reports
  Hit@5, Recall@5, MRR@5, nDCG@5, latency, missing evidence, failures, hashes,
  and model/reranker status. Mock embeddings are diagnostic only.
- `python -m eval.evaluate_indexed_retrieval` compares whole-corpus BM25,
  dense, hybrid, and hybrid with a supplied policy-ID filter over the verified
  policy QA records. It records per-question ranks, bootstrap intervals, and
  latency without rebuilding document vectors.
- `python -m eval.run_live_answers` collects checkpointed Gemini responses;
  `python -m eval.score_live_answers` scores reviewed labels, refusal flags,
  citation quotes, latency, and token usage. These calls send retrieved policy
  excerpts and questions to Gemini and require explicit approval.
- `python -m eval.run_chunking_ablation` evaluates chunking alternatives.
- `python -m eval.run_parser_comparison` compares PyMuPDF with the optional
  Unstructured parser and reports latency, clause counts, and table
  preservation metrics.
- `python -m eval.score_extraction` scores supplied predictions against manual
  extraction labels using exact normalized values; it avoids unsupported
  precision/recall claims for single-valued fields.
- `python -m eval.run_ragas` is an optional evaluator for faithfulness,
  answer-relevance, context-precision, and context-recall metrics. It requires
  optional dependencies, an indexed corpus, a real app LLM provider, and an
  OpenAI judge key; it has not been run for the current project state.

The documented retrieval ablation is a small one-policy diagnostic over four
answerable Care Supreme questions: Hit@5 was 0.75 for the measured
configurations, MRR@5 ranged from 0.425 to 0.750, and nDCG@5 ranged from 0.504
to 0.750. These results do not establish corpus-wide improvement. The
CrossEncoder quality was not measured in that run because model weights were
not cached.

The newer indexed-corpus evaluation covered 40 answerable questions from
eleven verified policies. Hybrid Recall@5 was 42.5% across the whole corpus
and 77.5% when the correct policy ID was supplied by the caller. This is an
exploratory development-set result, not automatic query-routing accuracy or
held-out performance. The live Gemini run answered 44 official-policy
questions. Assistant review marked 36 of 40 answerable responses correct;
three of four unanswerable questions were correctly flagged for refusal.
Citation and failure details are in the root `RESULTS.md`.

### Tests, lint, and CI

- Tests are under `tests/unit/` and `tests/integration/`. They cover chunking,
  parsing, BM25 and hybrid retrieval, RRF, indexing, evaluation metrics, RAG
  behavior, and API routes.
- The latest local unit run completed with **34 tests passed**. The latest
  full-suite rerun stalled at a FastAPI `TestClient` test in this environment;
  an earlier full-suite run passed 43 tests.
- The evaluation-data validator completed successfully for the current local
  dataset: 57 QA items, 20 comparison cases, and seven verified extraction
  fields with references passing.
- GitHub Actions (`.github/workflows/ci.yml`) installs Python 3.11 and
  development dependencies, runs `ruff check .`, `ruff format --check .`, and
  `pytest -v`.

### Deployment and runtime configuration

- `Dockerfile` builds the FastAPI service; `Dockerfile.ui` builds the
  Streamlit service.
- `docker-compose.yml` defines Qdrant, API, and UI containers, persistent
  Qdrant and policy-metadata volumes, API/UI port mappings, and health checks.
- Qdrant can run locally embedded or as the Compose server. The Compose API
  configuration selects server mode and connects to the `qdrant` service.
- `.env.example` documents LLM, embedding, reranker, Qdrant, database URL,
  logging, and Langfuse-related settings. Secrets must be supplied locally and
  not committed.
- Langfuse configuration is present, but there are no active tracing calls in
  the current request/generation path.
- `DATABASE_URL` is configured, but the current application does not implement
  a relational persistence workflow; policy registry/BM25 metadata uses local
  JSON and vectors use Qdrant.
- Docker build/start/health and real deployment behavior have not been
  verified for the current project state.

## Repository map

| Path | Responsibility |
|---|---|
| `app/api/` | FastAPI routes and shared pipeline initialization |
| `app/core/` | Settings, core Pydantic models, LLM clients, logging |
| `app/indexing/` | Chunking, embeddings, BM25, and Qdrant storage |
| `app/ingestion/` | PyMuPDF parsing and optional parser comparison |
| `app/rag/` | Prompts, response schemas, and RAG orchestration |
| `app/retrieval/` | Hybrid retrieval and reciprocal-rank fusion |
| `data/` | Policy manifest, acquisition/ingestion scripts, sample data |
| `eval/` | Gold datasets, validators, evaluation runners, scoring |
| `tests/` | Unit and API/dense-search integration tests |
| `ui/` | Streamlit user interface |
| `Dockerfile*`, `docker-compose.yml` | Container and local multi-service setup |
| `.github/workflows/ci.yml` | GitHub Actions lint, format, and test workflow |

## Local setup and commands

Requires Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `LLM_PROVIDER=mock` for explicit offline development, or configure
`GEMINI_API_KEY` for live Gemini generation. The embedding model is downloaded
by Sentence Transformers when first needed.

Start the services locally:

```powershell
python -m uvicorn app.main:app --reload --port 8000
python -m streamlit run ui/app.py
```

Useful checks and workflows:

```powershell
ruff check .
ruff format --check .
pytest -v
python -m eval.validate_eval_data
python -m data.acquire_policies
python -m data.ingest_policies
python -m app.cli list-policies
docker compose up --build
```

See the root [README](../README.md) for user-facing setup and workflow examples,
the [evaluation guide](../eval/README.md) for dataset formats and evaluation
details, and [decisions](./DECISIONS.md) for design notes.

## Known limitations and unverified behavior

- No OCR is implemented for scanned/image-only PDFs.
- Only Gemini and explicit mock LLM providers are implemented.
- Summaries do not provide field-level source citations.
- Model-generated analysis and mock results are not proof of factual accuracy.
- The labeled QA set is small and manually checked; it does not establish
  general answer accuracy.
- Extraction predictions have not been scored and RAGAS has not been run.
- The retrieval and live-answer evaluations are small exploratory samples;
  they have no held-out split or independent human answer review.
- CrossEncoder reranker quality, Docker behavior, Langfuse tracing, and
  production deployment/uptime have not been verified.
- The Streamlit UI currently hard-codes its API URL, which may prevent the
  Compose UI container from reaching the Compose API service.
- The system is informational only and is not insurance, legal, or financial
  advice.
