.PHONY: install test lint sample ingest coverage cells label synth gold gold-tool maps phase2 train evaluate phase3

DATE ?= 2026-10-26
REGION ?= baltic

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

# Phase 2: labels
cells:
	gnssguard cells

label:
	gnssguard label

synth:
	gnssguard synth --region central_europe --date $(DATE) --seeds 0,1,2,3,4

gold:
	gnssguard gold-sample --dates $(DATE) --n 300

gold-tool:
	gnssguard gold-tool

maps:
	gnssguard map --region $(REGION) --date $(DATE) --gold labels/gold.csv

phase2: cells label synth maps

# Phase 3: detection model
train:
	gnssguard train

evaluate:
	gnssguard evaluate

phase3: train evaluate
