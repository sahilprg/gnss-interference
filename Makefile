.PHONY: install test lint sample ingest coverage

DATE ?= 2026-10-26

install:
	pip install -e ".[dev,fast]"

test:
	python -m pytest -q

lint:
	ruff check .

# ~2 minutes: first 5,000 aircraft of one day, to check everything works
sample:
	gnssguard ingest --date $(DATE) --max-traces 5000 --out data/sample --force
	gnssguard coverage --data data/sample

# one full day into the main store (streams ~4-6 GB, writes tens of MB)
ingest:
	gnssguard ingest --date $(DATE) --out data/processed

coverage:
	gnssguard coverage --data data/processed --save reports/coverage
