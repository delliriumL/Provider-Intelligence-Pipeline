PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
STREAMLIT := $(VENV)/bin/streamlit
RUFF := $(VENV)/bin/ruff
PYTEST := $(VENV)/bin/pytest

.PHONY: install demo demo-llm demo-compare app test lint clean

install:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -e ".[dev]"

demo:
	$(PY) -m provider_intelligence.cli run-all

demo-llm:
	LLM_MODE=auto $(PY) -m provider_intelligence.cli run-all

demo-compare:
	LLM_MODE=off $(PY) -m provider_intelligence.cli generate-demo-data
	LLM_MODE=off $(PY) -m provider_intelligence.cli run-pipeline
	mkdir -p outputs/rule_based
	cp outputs/recommendations.json outputs/rule_based/
	cp outputs/synthetic_ground_truth.csv outputs/rule_based/ 2>/dev/null || true
	LLM_MODE=auto $(PY) -m provider_intelligence.cli run-pipeline
	mkdir -p outputs/adaptive
	cp outputs/recommendations.json outputs/adaptive/
	$(PY) -m provider_intelligence.cli evaluate --compare-llm \
		--rule-based-dir outputs/rule_based \
		--adaptive-dir outputs/adaptive
	$(PY) -m provider_intelligence.cli estimate-cost

app:
	$(STREAMLIT) run app/streamlit_app.py

test:
	LLM_MODE=off $(PYTEST)

lint:
	$(RUFF) check src tests app

clean:
	rm -rf .pytest_cache .ruff_cache
	rm -rf outputs/*.json outputs/*.csv outputs/rule_based outputs/adaptive
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
