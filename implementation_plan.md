# Implementation Plan

[Overview]
Update `momentum/src/llm.rs` to add robust local inference features: async prefetch and layer warmup, batch generation API (`complete_batch` and `complete_stream_batch`), safer model path handling, and robust memory-only warmup/prompt fallback support matching the requested architecture.

[Types]
- `GpuStatus`: Existing struct retained and validated.
- `InferenceRuntime`: Existing struct retained and validated.
- `LoadedModel`: Existing struct retained.
- `PREFETCH_TASK`: Added `OnceLock<Mutex<Option<JoinHandle<()>>>>` to manage async prefetch tasks safely without blocking.

[Files]
- `/home/mehtculos-xfemag/AI_project/momentum/src/llm.rs`: Replaced with the complete enhanced implementation containing async prefetch, prewarm, batch generation APIs, and robust model loading/fallback.

[Functions]
- `async_prefetch_model(model_path: &str)`: Spawns non-blocking file access task to prefetch model weights.
- `async_prewarm_server()`: Spawns non-blocking warmup task.
- `complete_batch(prompts: &[String], max_tokens: u32) -> anyhow::Result<Vec<String>>`: Batch inference utility.
- `complete_stream_batch(prompts: &[String], max_tokens: u32, mut on_token: impl FnMut(&str)) -> anyhow::Result<Vec<String>>`: Batch streaming inference utility.
- Existing `complete`, `complete_stream`, `embed_text`, `cosine_sim` retained and fully functional.

[Classes]
- None new; existing structures updated.

[Dependencies]
- Existing dependencies (`infer-rs`, `tokio`, `anyhow`) in `Cargo.toml`.

[Testing]
- Run `cargo check` and `cargo test` to ensure successful compilation and test suites pass.

[Implementation Order]
1. Write the updated `momentum/src/llm.rs` using the editor tool.
2. Run `cargo check` and `cargo test` to verify compilation and correctness.

