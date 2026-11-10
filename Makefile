# Each target is one step. `make all` rebuilds every number, model, chart and notebook from data.zip.
# Inside Docker, PY and DATA_ZIP are set by the image, so the same targets work there unchanged.
PY ?= .venv/bin/python
DATA_ZIP ?= ../data.zip
EXAMPLE := outputs/worked_example_SP0267_2024-12-12

setup:
	uv venv --python 3.12 && uv pip install -r requirements.lock && uv pip install -e . --no-deps

data:
	$(PY) scripts/unpack_data.py $(DATA_ZIP)

audit:
	$(PY) -m quote_followup audit --zip $(DATA_ZIP)

scan:
	@mkdir -p outputs
	$(PY) scripts/explore.py | tee outputs/problem_scan.txt

model:
	$(PY) -m quote_followup train --as-of 2024-12-12

example:
	@mkdir -p outputs
	$(PY) -m quote_followup board --sp SP0267 --as-of 2024-12-12 --hindsight --out $(EXAMPLE).csv | tee $(EXAMPLE).txt

backtest:
	$(PY) -m quote_followup backtest

ablations:
	$(PY) scripts/ablations.py

sensitivity:
	$(PY) scripts/sensitivity.py

figures: backtest
	$(PY) scripts/figures.py
	$(PY) scripts/animate.py

notebooks:
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:
	$(PY) -m pytest

all: data test audit scan model example ablations sensitivity figures notebooks

.PHONY: setup data audit scan model example backtest ablations sensitivity figures notebooks test all
