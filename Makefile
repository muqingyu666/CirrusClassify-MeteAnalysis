PYTHON ?= python
PACKAGE := cirrusclassify

.PHONY: setup lint format test coverage docs

setup:
	$(PYTHON) -m pip install -U pip
	$(PYTHON) -m pip install -e .[develop]

lint:
	ruff check src tests scripts

format:
	black src tests scripts
	ruff check src tests scripts --fix

unit:
	pytest -q

coverage:
	pytest --cov=$(PACKAGE) --cov-report=term-missing

docs:
	sphinx-build -b html docs docs/_build
