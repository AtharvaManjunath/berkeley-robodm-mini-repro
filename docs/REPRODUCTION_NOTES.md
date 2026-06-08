# Reproduction Notes

Recommended synthetic reproduction:

```bash
make install
make smoke
```

Public-data attempts require explicit permission:

```bash
python3.11 -m robot_bench.cli run-all --config configs/datasets.yaml --profile small --allow-downloads
```

The current scaffold intentionally records unavailable public loaders as failed rows unless a local TFDS/RLDS integration is added. This keeps failed replications visible.
