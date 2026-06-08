# Result Schema

Each CSV/JSONL row records:

- dataset_name
- format
- adapter_kind
- codec
- compression
- profile
- episodes_requested
- episodes_loaded
- steps_loaded
- frames_loaded
- batch_size
- num_batches
- workers
- disk_bytes
- mean_episode_bytes
- median_episode_bytes
- p95_episode_bytes
- throughput_episodes_per_sec
- throughput_frames_per_sec
- mean_batch_latency_ms
- p50_batch_latency_ms
- p95_batch_latency_ms
- peak_rss_mb
- conversion_seconds
- benchmark_seconds
- pure_load_time_seconds
- seed
- hostname/platform/python/package versions
- status
- error_type
- error_message
- notes
