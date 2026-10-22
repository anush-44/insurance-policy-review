# PolicyLens Architecture Decision Records (ADRs)

## ADR 001: PDF Parsing Engine (PyMuPDF vs. Unstructured vs. LlamaParse)
- **Context:** Insurance policy documents are dense 30–70 page PDFs containing complex nested clauses, tables (waiting period matrices, day-care lists, sub-limits), and multi-column layouts.
- **Decision:** PyMuPDF (`fitz`) is chosen as the foundational baseline engine. We implement a comparative benchmarking harness (`eval/run_parser_comparison.py`) to test against `unstructured` on table-dense pages.
- **Trade-offs:**
  - *PyMuPDF:* Ultra-fast (milliseconds per page), zero external API dependency, highly reliable vector bounding boxes and built-in table extraction (`page.find_tables()`). It requires custom logic for hierarchical clause detection.
  - *Unstructured / OCR:* Slower, heavier dependency tree, but handles complex multi-column layouts well.
  - *LlamaParse:* Cloud API; high accuracy on exotic layouts but introduces external latency, cost, and data privacy considerations for local deployments.

## ADR 002: Dual Storage Mode for Qdrant (Docker Server vs. Local Embedded)
- **Context:** Developers may run the system in containerized environments (Docker Compose) or directly in local Python environments where Docker may not be installed.
- **Decision:** Implement configurable `QDRANT_MODE`:
  - `server`: Connects via HTTP/gRPC to Qdrant Docker container.
  - `local`: Uses Qdrant's embedded storage (`path="./qdrant_storage"` or `":memory:"`).
- **Trade-offs:** Server mode is production-identical and supports multi-worker concurrency; embedded mode allows instant development and zero-dependency local test suites.

## ADR 003: LLM Provider Abstraction
- **Context:** Default LLM is Google Gemini, but the project must allow switching to OpenAI, Claude, or local Ollama models without breaking downstream retrieval or extraction code.
- **Decision:** Design an abstract `LLMClient` base class in `app/core/llm.py` with structured output capabilities (Pydantic models) and prompt templating.
- **Trade-offs:** Adds a lightweight abstraction layer, but decouples prompt engineering from vendor-specific client libraries.

## ADR 004: Embedding & Cross-Encoder Model Selection
- **Context:** Indian insurance text includes domain-specific clauses, statutory terms, and numeric codes.
- **Decision:** BAAI's `bge-base-en-v1.5` for dense vector search and `bge-reranker-base` for cross-encoder reranking.
- **Trade-offs:** `bge-base-en-v1.5` offers top-tier retrieval performance on MTEB benchmarks while remaining efficient enough to run CPU-side locally. `bge-small-en-v1.5` is supported as a lightweight alternative via configuration.
