# Menty

Local Rust assistant (`momentum`) plus its GGUF inference engine (`infer-rs`).

No model weights are in this repo. Point Momentum at your own GGUF blobs.

## Layout

- `momentum/` — egui chat app, memory, tools, guardian
- `infer-rs/` — GGUF load, quant, CUDA/CPU kernels, tokenizer

`momentum` depends on `infer-rs` via `path = "../infer-rs"`.

## Build

```bash
cd momentum
cargo build --release
```

Needs Rust 2021, a display (X11), and `libcuda.so.1` if you want GPU layers.
Whisper and `spd-say` are optional external tools.
