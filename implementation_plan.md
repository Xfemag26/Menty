# Implementation Plan - Benchmark Harness & LLM Profiling

## Overview
Add a local benchmarking harness (`src/bin/bench.rs`, `scripts/benchmark.sh`), extend `Cargo.toml` with `clap` and `csv`, and instrument `src/llm.rs` with non-invasive timing hooks, telemetry calls, and local CSV performance logging (`/tmp/inference_profile.csv`).

## Types
- `Args` (in `src/bin/bench.rs`): CLI arguments (`profile`, `prompt_file`, `iterations`, `max_tokens`).

## Files
- `Cargo.toml`: Add `clap = { version = "4", features = ["derive"] }`, `csv = "1.2"`, and update `tracing-subscriber` features (`fmt`).
- `src/bin/bench.rs`: New bench binary loading model profiles, running streaming inference, and writing JSON summaries (`/tmp/bench_out.json`).
- `scripts/benchmark.sh`: Shell script running iterations via `time` to measure elapsed time, peak RSS, and token counts.
- `src/llm.rs`: Add timing instrumentation around model loading steps, tokenization, prefill, generation, and CSV output logging.
- `src/main.rs`: Ensure `telemetry::init()` is called at startup with warning handling.

## Functions
- `main()` in `src/bin/bench.rs`: Parses CLI args, initializes telemetry, loads/prefetches model profile, executes benchmark iterations.
- Instrumented blocks in `src/llm.rs`:
  - `InferenceRuntime::start_with_profile`: time gguf open, tokenizer init, model init, and per-layer CUDA upload.
  - `complete_stream`: time tokenization, prompt prefill, generation loop, append metrics to `/tmp/inference_profile.csv`, write `/tmp/bench_out.json`.

## Dependencies
- `clap = { version = "4", features = ["derive"] }`
- `csv = "1.2"`
- `tracing-subscriber` (`fmt` feature)

## Testing
- Run `cargo build --release` and test the bench binary / script against local prompt files. Verify `/tmp/inference_profile.csv` and `/tmp/bench_out.json` output.

## Implementation Order
1. Update `Cargo.toml` with bench dependencies.
2. Create `src/bin/bench.rs`.
3. Create `scripts/benchmark.sh` and make executable (`chmod +x`).
4. Instrument `src/llm.rs` with timing hooks and CSV/JSON logging.
5. Verify startup telemetry call in `src/main.rs`.
6. Run release build and benchmark script.
