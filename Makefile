.DEFAULT_GOAL := help
.PHONY: help install corpus stats chunks questions index reindex eval analyze generate ask serve gate docker-build docker-run test lint fmt check clean

PYTHON ?= python

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Install pinned dependencies
	$(PYTHON) -m pip install -r requirements.txt

corpus:  ## Download the pinned PostgreSQL docs corpus
	bash data/download.sh

stats:  ## Print corpus statistics
	$(PYTHON) -m src.ingestion.corpus_stats

chunks:  ## Compare the two chunking strategies
	$(PYTHON) -m src.ingestion.chunk_stats

questions:  ## Verify the question set against the corpus
	$(PYTHON) -m src.eval.verify_questions

index:  ## Build the BM25 and dense retrieval indexes
	$(PYTHON) -m src.ingestion.indexer

reindex:  ## Rebuild the dense index from scratch
	$(PYTHON) -m src.ingestion.indexer --rebuild

eval:  ## Run all questions through all retrievers, then report
	$(PYTHON) -m src.eval.runner
	$(PYTHON) -m src.eval.report

analyze:  ## Explain the evaluation results question by question
	$(PYTHON) -m src.eval.analyze

generate:  ## Run the full pipeline over every question, then report
	$(PYTHON) -m src.eval.generation_runner
	$(PYTHON) -m src.eval.generation_report

ask:  ## Ask one question through the full pipeline
	$(PYTHON) -m src.generation.demo

serve:  ## Run the API locally with reload
	$(PYTHON) -m uvicorn src.api:app --reload --port 8000

gate:  ## Check committed results against their recorded floors
	$(PYTHON) scripts/check_thresholds.py

docker-build:  ## Build the container image
	docker build -t technical-docs-rag .

docker-run:  ## Run the image, mounting the corpus and indexes
	docker run --rm -p 8000:8000 		-v "$(CURDIR)/data:/app/data" 		--add-host=host.docker.internal:host-gateway 		technical-docs-rag

test:  ## Run the test suite
	$(PYTHON) -m pytest tests/ -q

lint:  ## Check formatting and lint rules
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m ruff format --check src tests scripts

fmt:  ## Apply formatting and autofixable lint rules
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m ruff format src tests scripts

check: lint test gate  ## Everything CI runs

clean:  ## Remove caches (keeps the downloaded corpus and indexes)
	rm -rf .pytest_cache .ruff_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
