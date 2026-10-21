# Implementation Plan

[Overview]
Implement codebase-wide optimizations, dead code cleanup, robust error handling, and performance refinements across Momentum.

[Types]
- No major structural type changes required; minor additions to error handling types and telemetry wrappers if needed.

[Files]
- `src/action_log.rs`: Clean up or utilize unused structs (`PendingAction`) and methods (`list_pending`, `mark_done`, `mark_rejected`) or annotate with `#[allow(dead_code)]`.
- `src/llm.rs`: Optimize prompt processing and streamline fallback error handling.
- `src/config.rs`: Verify profile configurations and mmap settings.
- `src/tools.rs`: Enhance error propagation and resource bounds checking.
- `src/reflection.rs`: Tune idle reflection polling and audit heuristics.

[Functions]
- `src/action_log.rs`: Expose or annotate unused dead-code methods to eliminate compiler warnings.
- `src/llm.rs`: Ensure robust error handling on token decoding and batch execution streams.

[Classes]
- None.

[Dependencies]
- No new dependencies; existing Cargo.toml dependencies are sufficient.

[Testing]
- Run `cargo check` and `cargo test` to validate zero warnings and passing test suites.
- Verify release build via `cargo build --release`.

[Implementation Order]
1. Clean up unused dead code and compiler warnings across `src/action_log.rs` and other modules.
2. Refine error handling and logging across `src/tools.rs` and `src/llm.rs`.
3. Run verification tests (`cargo test`, `cargo build --release`).
 callers in `src/main.rs`.
3. Validate build and benchmark across profiles.

