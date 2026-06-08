# robot-data-format-benchmark

[![CI](../../actions/workflows/ci.yml/badge.svg)](../../actions/workflows/ci.yml)
![Python 3.11](https://img.shields.io/badge/python-3.11-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

Reproducible storage and loading benchmarks for robot-learning dataset formats.

`robot-data-format-benchmark` is a research-systems benchmark harness inspired by the storage/loading benchmark component of the Berkeley Robo-DM paper. It compares robot dataset representations on disk size, peak RAM, and loading plus dummy training-loop throughput across deterministic synthetic fixtures and a configured N=8 set of public robot-learning dataset targets.

This is not an official Robo-DM reproduction, and it does not claim to exactly recreate the paper. The project is designed to make replication attempts auditable: exact/source adapters, local `*-style` adapters, fallbacks, failed dataset loads, and implementation tradeoffs are recorded explicitly.

## What It Compares

| Format | Adapter status | Implementation note |
|---|---|---|
| RLDS / RLDS-style | source or style | Source TFDS/RLDS loading is intended when available; CI uses a local RLDS-style fixture. |
| HDF5 | style | Lossless, uncompressed, one `.h5` file per episode by default. |
| LeRobot-style | style | Local Parquet low-dimensional data plus MP4/array-backed visual streams. |
| Robo-DM-style fallback | fallback | Self-contained `.rdm` episode containers. This is not official Robo-DM. |

## Metrics

| Metric | What is measured |
|---|---|
| Disk size | Recursive format directory size plus approximate per-episode statistics. |
| Peak RSS RAM | Main process and child-process RSS sampled with `psutil`. |
| Dummy training-loop throughput | Episode loading, image decode to arrays, and deterministic Torch CPU forward workload. |
| Batch latency | Mean, p50, and p95 measured batch latency after warmup batches. |

## Quickstart

The quickstart is fully offline and uses synthetic data only. It does not download public datasets.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
make install
make smoke
make report
```

Outputs are generated under `artifacts/`:

- `artifacts/results/results.csv`
- `artifacts/results/results.jsonl`
- `artifacts/reports/benchmark_report.md`
- `artifacts/reports/replication_matrix.md`
- `artifacts/reports/tradeoffs.md`
- `artifacts/plots/*.png`
- `artifacts/manifests/episode_selection_<profile>.json`

`artifacts/` is ignored by Git except for `.gitkeep` placeholders, so benchmark outputs do not get committed accidentally.

## Why N=8

`configs/datasets.yaml` lists eight public robot-learning dataset targets requested for the benchmark:

1. `bridge`
2. `berkeley_cable_routing`
3. `nyu_door_opening_surprising_effectiveness`
4. `berkeley_autolab_ur5`
5. `nyu_franka_play_dataset_converted_externally_to_rlds`
6. `cmu_play_fusion`
7. `bridge_data_msr`
8. `utokyo_xarm_pick_and_place_converted_externally_to_rlds`

Public-data runs are intentionally guarded. The harness does not silently replace a failed public dataset with another dataset. Failures are represented in `results.csv` with `status=failed`, `error_type`, and `error_message`.

## Common Commands

```bash
make install          # install package and dev dependencies
make test             # run synthetic/unit/integration tests
make lint             # run ruff
make smoke            # offline end-to-end synthetic benchmark
make benchmark-small  # small profile, synthetic config by default
make report           # regenerate reports from artifacts/results/results.csv
make clean-artifacts  # remove generated outputs, keep .gitkeep files
```

The Makefile defaults to `python3.11`. If your Python 3.11 binary has another name:

```bash
make PYTHON=/path/to/python3.11 smoke
```

## CLI

```bash
python3.11 -m robot_bench.cli prepare --config configs/synthetic.yaml --profile small
python3.11 -m robot_bench.cli convert --config configs/synthetic.yaml --formats rlds,hdf5,lerobot,robodm --profile small
python3.11 -m robot_bench.cli benchmark --config configs/synthetic.yaml --formats rlds,hdf5,lerobot,robodm --profile small
python3.11 -m robot_bench.cli report --results artifacts/results/results.csv
python3.11 -m robot_bench.cli run-all --config configs/synthetic.yaml --profile small
```

## Public Dataset Attempts

Public downloads are never automatic. To attempt a small public-data run, configure local caches or source access first, then explicitly opt in:

```bash
python3.11 -m robot_bench.cli run-all \
  --config configs/datasets.yaml \
  --profile small \
  --allow-downloads
```

Full mode removes the default episode cap unless explicitly configured and uses 200 measured batches by default:

```bash
python3.11 -m robot_bench.cli run-all \
  --config configs/datasets.yaml \
  --profile full \
  --allow-downloads
```

Expect full public-data runs to require substantial storage, CPU time, RAM, and dataset-specific setup. `--max-download-gb` guards are configured to prevent accidental very large downloads.

## Docker

A Dockerfile is included for reproducible smoke runs, but the Docker build is not currently verified in CI.

```bash
make docker-build
docker run --rm \
  -v "$PWD/artifacts:/app/artifacts" \
  -v "${TFDS_DATA_DIR:-$PWD/.cache/tensorflow_datasets}:/data/tensorflow_datasets" \
  robot-data-format-benchmark
```

## Repository Map

| Path | Purpose |
|---|---|
| `src/robot_bench/` | Package code: CLI, adapters, benchmark runner, reports. |
| `configs/` | Synthetic and N=8 public dataset configs. |
| `tests/` | CI-safe synthetic tests; no large downloads. |
| `docs/` | Methodology, reproduction notes, dataset notes, tradeoffs, result schema. |
| `artifacts/` | Runtime outputs, ignored except `.gitkeep`. |
| `.github/workflows/ci.yml` | Lint, tests, and synthetic smoke workflow. |

## Current Verification

The repository is intended to support:

```bash
make install
make test
make lint
make smoke
make benchmark-small
make report
```

Docker build depends on a local Docker installation and is not required for CI, tests, or smoke.

## Limitations

- Sample-capped `small` results are workstation-friendly, not full paper-scale measurements.
- RLDS source loading depends on available TFDS/RLDS datasets or local caches; synthetic tests use RLDS-style fixtures.
- LeRobot-style conversion may flatten or omit modalities that do not map naturally to Parquet/video.
- Robo-DM-style fallback is not official Robo-DM.
- Video codec availability affects disk size and decode throughput.
- Hardware, filesystem cache state, and CPU video decode strongly affect throughput and RAM.

For more detail, see:

- `docs/METHODOLOGY.md`
- `docs/REPRODUCTION_NOTES.md`
- `docs/TRADEOFFS.md`
- `docs/DATASETS.md`
- `docs/RESULT_SCHEMA.md`
