# Methodology

## Dataset Selection

The public target config contains the requested N=8 datasets. CI and default smoke runs use deterministic synthetic fixtures so tests do not download large public data.

## Episode Sampling

Episode selection is deterministic per dataset using the configured seed. The selected episode IDs and lengths are written to `artifacts/manifests/episode_selection_<profile>.json`.

## Conversion Details

- RLDS: source where available; local RLDS-style NPZ fixture for synthetic and non-source exports.
- HDF5: uncompressed lossless `.h5` file per episode.
- LeRobot-style: Parquet low-dimensional data plus MP4 video streams; unsupported modalities are flattened when practical.
- Robo-DM-style fallback: self-contained `.rdm` container with JSON header, numeric NPZ payloads, video payloads, and simple seeking through the zip index.

## Timing Boundaries

Measured headline latency includes complete episode loading, image decode into arrays, and deterministic dummy Torch CPU forward compute. Warmup batches are ignored.

## Memory Measurement

Peak RSS is sampled with psutil for the main process and recursive child processes at a default 50 ms interval.

## Dummy Training Loop

The loop normalizes image arrays, computes image statistics, flattens action/state arrays, feeds an 8-feature vector through a tiny Torch module, and optionally runs backward when enabled.

## Reproducibility Controls

Python, NumPy, PyTorch, and TensorFlow seeds are set when those libraries are present. Dataset selection manifests, config files, package versions, Python version, platform, and hostname are recorded.

## Limitations

Sample-capped runs are intentionally workstation-friendly and are not full paper-scale measurements. Filesystem cache state, ffmpeg codec support, CPU model, RAM, and storage device can materially affect results.
