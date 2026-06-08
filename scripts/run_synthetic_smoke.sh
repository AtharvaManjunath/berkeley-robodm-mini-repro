#!/usr/bin/env bash
set -euo pipefail

python -m robot_bench.cli run-all --config configs/synthetic.yaml --profile small
