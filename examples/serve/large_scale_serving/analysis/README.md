# analysis

Per-request and engine-side metrics for one serving run, from what an attempt directory holds:
`request_trace/` (client-facing requests and responses), `perf_metrics/` (proxy and worker timing),
`adp_route_trace-ctx-N.jsonl` (routing decisions) and `ctx-N.log` / `gen-N.log` (iteration lines).

```
python3 analysis/report.py <attempt_dir>                        # whole run
python3 analysis/report.py <attempt_dir>/request_trace/<hour>   # one UTC hour
```

Writes `requests.csv`, `ctx_iters.csv`, `ctx_rank_iters.csv`, `gen_iters.csv`, `gen_rank_iters.csv`
and `REPORT.html` under `_reports/<run>[_<hour>]/`. Everything joins on `disagg_request_id`.

| file | role |
|---|---|
| `requests_table.py` | one row per request: identity, usage tokens, latency phases, KV blocks, routing |
| `engine_iters.py` | iteration lines → per-rank rows and per-instance (ranks pooled) rows, pads removed |
| `charts.py` | SVG line charts and histograms, no dependencies |
| `report.py` | entry point; CSVs plus the HTML |
| `common.py` | time parsing, percentiles, CSV writer |

## requests.csv

| column | definition |
|---|---|
| `rid` | `disagg_request_id`; the join key to perf_metrics and the routing trace |
| `started_at` | request arrival: middleware `server_arrival_time` (steady clock) plus the wall-clock offset recovered as `min(recorded_at − server_arrival_time)` over the attempt |
| `handler_entry_at` | the trace's `recorded_at`, stamped after the body was read and validated |
| `finished_at` | response `finished_at` |
| `parse_ms` | `handler_entry_at − started_at`: body read and validation before the handler ran (about 1.5 s per MB) |
| `proxy_dispatch_ms` | proxy `disagg_ctx_dispatch_time − disagg_server_arrival_time` |
| `status` | `rejected_400`, or the response status: `completed`, `error`, `client_disconnected`, `no_response` |
| `isl_total`, `isl_cached`, `osl` | usage the client received (chat: last chunk; responses: `response.completed`) |
| `isl_new` | `isl_total − isl_cached` |
| `cache_hit_ratio` | `isl_cached / isl_total` |
| `ttft_ms` | proxy `disagg_server_first_token_time − disagg_server_arrival_time` |
| `e2e_ms` | `finished_at − started_at` of the same trace_id |
| `decode_ms` | `e2e_ms − ttft_ms` |
| `prefill_queue_ms` | ctx worker `first_scheduled_time − arrival_time` |
| `prefill_ms` | ctx worker `first_token_time − first_scheduled_time` |
| `gpu_prefill_ms` | ctx worker `time_breakdown_metrics.ctx_gpu_forward_time` |
| `kv_transfer_ms`, `kv_transfer_bytes` | gen worker `kv_cache_transfer_end − start`, `kv_cache_size` |
| `gen_decode_ms` | gen worker `last_token_time − first_token_time` |
| `ctx_blocks_total/new/reused`, `gen_blocks_total` | worker `kv_cache_metrics` |
| `ctx_instance`, `routed_rank`, `route_phase`, `route_iter`, `route_log_iter` | routing decision (`best_rank`, `phase`, batch iteration) |

Caveat: on `/v1/responses` the server currently reports `cached_tokens == input_tokens`; the value is
recorded as received. `ctx_blocks_reused × tokens_per_block` is the engine-side measurement.

## ctx_rank_iters.csv / gen_rank_iters.csv (one row per worker, iteration, rank)

| column | definition |
|---|---|
| `engine_instance` | engine lifetime inside the log (the KV-sizing dry run is 0, the real engine 1) |
| `ctx_tokens` | `states.num_ctx_tokens` as logged |
| `is_adp_pad` | attention-DP dummy: all four `kv_*_blocks` counters unchanged since this rank's previous iteration and `ctx_tokens ≥ 0.9 × max_num_tokens` |
| `ctx_tokens_real` | `ctx_tokens`, or 0 for a pad |
| `decode_requests_real`, `gen_tokens_real` | gen: `scheduled_requests` / `gen_tokens`, or 0 on an idle rank (`is_adp_pad` on gen = one scheduled request with `kv_cache_util == 0`) |
| `token_budget_util` | `ctx_tokens_real / max_num_tokens` |
| `kv_hit_rate_iter` | `Δkv_reused_blocks / (Δreused + Δmissed)` for this iteration |
| `kv_hit_rate_cum` | the log's `kv_hit_rate`: the same ratio over the engine lifetime |
| `kv_cache_util` | as logged: share of blocks pinned by in-flight requests (`1 − available / max`) |
| `kv_capacity_blocks` | `max(kv_free_blocks + kv_evictable_blocks)` over the engine lifetime |
| `kv_pool_filled_ratio` | `1 − kv_free_blocks / kv_capacity_blocks`: blocks holding content, pinned or reusable |
| `kv_{offload,onboard,host_dropped}_blocks_delta/_total` | cross-tier movement this iteration / cumulative |
| `host_step_ms`, `device_step_ms` | `host_step_time`, `prev_device_step_time` |

## ctx_iters.csv / gen_iters.csv (ranks pooled per iteration)

| column | definition |
|---|---|
| `busy_ranks`, `has_prefill` | ranks with `ctx_tokens_real > 0`; any |
| `ctx_tokens_mean` | mean of `ctx_tokens_real` over all ranks, pads included as 0 |
| `token_budget_util` | `ctx_tokens_mean / max_num_tokens` |
| `*_rank_skew` | `(max − mean) / mean` across ranks; 0 = even, `ranks − 1` = one rank has everything |
| `idle_ranks` | ranks flagged as pads this iteration |
| `decode_requests_sum`, `has_decode` | Σ `decode_requests_real`; any (gen) |
| `batch_occupancy` | mean `decode_requests_real` / `max_batch_size` (gen) |
| `tokens_per_request` | `Σ gen_tokens_real / Σ decode_requests_real` (gen; accepted draft length + 1 under MTP) |
| `kv_hit_rate_iter` / `kv_hit_rate_cum` | counters summed over ranks first, then divided |
| `kv_pool_filled_mean` | mean of the per-rank ratio |
