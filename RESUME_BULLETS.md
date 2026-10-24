# Resume Bullet Options

- Improved a PyMuPDF-based insurance policy ingestion pipeline with single-level and nested clause detection, UIN extraction, page-span preservation, and per-policy ingestion reporting.
- Added rollback-safe Qdrant/BM25/registry updates and policy deletion, plus configurable sentence-transformer cross-encoder reranking with observable RRF fallback.
- Built reproducible QA, extraction, and retrieval/chunking evaluation tooling with gold-reference validation, Hit@5/MRR/nDCG metrics, and explicit handling for missing evidence; quality improvements remain unclaimed pending multi-policy runs.
