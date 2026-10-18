# Implementation Plan

## Overview
Wire `ServerConnect` mode into real OpenAI-compatible remote chat completion proxying, and clean up confirmed unused constants and unread fields across the codebase.

## Types
- Clean up unread fields in `GuardianAlert` in `src/guardian.rs` if safe.

## Files
- `/home/mehtculos-xfemag/AI_project/momentum/src/config.rs`: Remove unused constants (`MAX_TOKENS`, `TEMPERATURE`, `REFLECTION_BUDGET`, `CONTEXT_MAX_SESSIONS`, `PERSONA`).
- `/home/mehtculos-xfemag/AI_project/momentum/src/main.rs`: Implement remote chat request forwarding via `ureq` when `AppMode::ServerConnect` is selected.

## Functions
- `send_message` in `src/main.rs`: Add branch for `AppMode::ServerConnect` to invoke remote endpoint (`/chat/completions`) and receive assistant responses.

## Dependencies
- None added (reusing `ureq` already present).

## Testing
- Run `cargo test --workspace` to ensure successful compilation and test pass.

## Implementation Order
1. Remove unused constants in `src/config.rs`.
2. Add remote chat forwarding logic for `ServerConnect` in `src/main.rs`.
3. Run `cargo test --workspace`.
