.PHONY: install test lint docker-build smoke benchmark-small benchmark-full report clean-artifacts

PYTHON ?= python3.11
CONFIG ?= configs/synthetic.yaml
FORMATS ?= rlds,hdf5,lerobot,robodm

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests

docker-build:
	docker build -t robot-data-format-benchmark .

smoke:
	$(PYTHON) -m robot_bench.cli run-all --config configs/synthetic.yaml --profile small --formats $(FORMATS)

benchmark-small:
	$(PYTHON) -m robot_bench.cli run-all --config $(CONFIG) --profile small --formats $(FORMATS)

benchmark-full:
	$(PYTHON) -m robot_bench.cli run-all --config configs/datasets.yaml --profile full --formats $(FORMATS)

report:
	$(PYTHON) -m robot_bench.cli report --results artifacts/results/results.csv

clean-artifacts:
	find artifacts -mindepth 1 -type f ! -name .gitkeep -delete
	find artifacts -mindepth 1 -type d -empty -delete
