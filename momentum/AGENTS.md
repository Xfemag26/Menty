# Whisper + Speak Tools (Momentum)

## Overview
This session wired local whisper.cpp transcription (`transcribe`) and basic TTS (`speak`) to the existing Momentum tool framework. Both tools were configured to fit the Rust CLI-only rule, avoiding in-process bindings or X11 dependencies. Permissions were aligned with existing `Policy` classification, and new paths were added in `tools.rs` and `config.rs`.

### What was added:
1. **Transcribe (`whisper-cli`)**
   - Reads local flac/mp3/ogg/wav, no downloads.
   - `Policy::check` -> Safe unless path is `Denied`.
   - Uses whisper.cpp with `ggml-base.en.bin`.

2. **Speak (`spd-say`)**
   - Reads short strings (max 500 chars).
   - Outputs audio locally via `spd-say --wait`.
   - Risk level: Safe in permissions.

### Key Decisions
- CLI-only maintained: No in-process whisper bindings, X11, or Python deps.
- Default deny fallback kept intact for unknown tools.
- Piper and other TTS deferred to future slices (YAGNI).

### Next Steps
- Smoke both paths: `transcribe` JFK wav; short `spd-say` message.
- Resolve missing `Cargo.toml` at `momentum` root.
- X11 input (blocked on missing packages).

---
**Files Edited**
- `permissions.rs`: Added policy for `transcribe`/`speak`.
- `tools.rs`: Integrated CLI for both tools.
- `config.rs`: Updates to `needs_tools` triggers and persona lines.
- AGENTS.md: Wrote this summary.