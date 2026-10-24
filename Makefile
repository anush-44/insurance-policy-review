.PHONY: test lint format eval eval-validate eval-retrieval run-api run-ui clean sample-pdf ingest list-policies

PYTHON := python
VENV_PYTHON := .venv/Scripts/python
VENV_PYTEST := .venv/Scripts/pytest
VENV_RUFF := .venv/Scripts/ruff

# Run unit and integration tests
test:
	$(VENV_PYTEST) -v tests/

# Lint with ruff
lint:
	$(VENV_RUFF) check .

# Format code with ruff
format:
	$(VENV_RUFF) format .

# Generate sample policy PDF for testing
sample-pdf:
	$(VENV_PYTHON) data/create_sample_pdf.py

# Run parser comparison benchmark
parser-eval:
	$(VENV_PYTHON) -m eval.run_parser_comparison --pdf data/sample_policy.pdf

# Validate manually labeled evaluation assets (no external API calls)
eval: eval-validate

eval-validate:
	$(VENV_PYTHON) -m eval.validate_eval_data

# Controlled ablation (set POLICY_ID and PDF; requires real embedding model files)
eval-retrieval:
	$(VENV_PYTHON) -m eval.run_retrieval_ablation --policy-id $(POLICY_ID) --pdf $(PDF) --reranker

# Run backend API locally
run-api:
	$(VENV_PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Run Streamlit UI locally
run-ui:
	$(VENV_PYTHON) -m streamlit run ui/app.py --server.port 8501

# Clean up cached files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

# Ingest a PDF: make ingest PDF=path/to/policy.pdf INSURER="HDFC ERGO"
ingest:
	$(VENV_PYTHON) -m app.cli ingest $(PDF) --insurer "$(INSURER)"

# List all indexed policies
list-policies:
	$(VENV_PYTHON) -m app.cli list-policies
