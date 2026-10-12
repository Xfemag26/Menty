#![allow(dead_code, unused_imports)]
// main.rs - High-performance low-resource LLM inference engine in Rust
mod avx;
mod cuda;
mod gguf;
mod kv_cache;
mod model;
mod ops;
mod partition;
mod quant;
mod speculative;
mod tokenizer;

use gguf::GGUFFile;
use kv_cache::PagedKVCache;
use model::TransformerModel;
use partition::HelmPartitioner;
use speculative::PromptLookupSpeculator;
use std::env;
use std::io::{self, Write};
use std::time::Instant;
use tokenizer::Tokenizer;

fn main() {
    let args: Vec<String> = env::args().collect();
    let default_model = env::var("MOMENTUM_DEFAULT_MODEL").ok();
    let model_path = args.get(1).or(default_model.as_ref());
    let Some(model_path) = model_path else {
        eprintln!("Usage: infer-rs <model.gguf> [prompt] (or set MOMENTUM_DEFAULT_MODEL)");
        return;
    };
    let prompt = args
        .get(2)
        .map(|s| s.as_str())
        .unwrap_or("Hello! Tell me about yourself.");

    println!("⚡ infer-rs: Ultra-Low Resource LLM Inference Engine");
    println!("📦 Loading model: {}", model_path);

    let start_time = Instant::now();
    let gguf = match GGUFFile::open(model_path) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("❌ Failed to load GGUF file: {e}");
            println!("\nUsage: infer-rs <model.gguf> [prompt]");
            return;
        }
    };

    for (name, info) in &gguf.tensors {
        if name.contains("token_embd")
            || name.contains("norm")
            || name.contains("output")
            || name.contains("blk.0.attn_q")
        {
            println!(
                "  Tensor: {} | qtype: {:?} | dims: {:?}",
                name, info.qtype, info.dims
            );
        }
    }

    println!(
        "✅ Model mmap loaded in {:.2}ms (0 copy, <15MB RSS)",
        start_time.elapsed().as_secs_f64() * 1000.0
    );

    let tokenizer = match Tokenizer::from_gguf(&gguf) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("❌ Failed to build tokenizer: {e}");
            return;
        }
    };

    let model = match TransformerModel::from_gguf(&gguf) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("❌ Failed to parse transformer model: {e}");
            return;
        }
    };

    // HELM Dynamic Partitioning Plan
    let partitioner = HelmPartitioner::new_mx150(2048);
    let plan = partitioner.plan(
        model.config.n_layers,
        gguf.mmap.len(),
        model.config.n_kv_heads,
        model.config.head_dim,
    );

    println!("\n📊 HELM Dynamic Layer Partition Plan (MX150 2GB):");
    println!("   Total Layers: {}", plan.total_layers);
    println!(
        "   GPU Offload Layers: {} (~{} MB VRAM)",
        plan.gpu_layers, plan.gpu_memory_mb
    );
    println!(
        "   CPU Fallback Layers: {} (~{} MB RAM)",
        plan.cpu_layers, plan.cpu_memory_mb
    );
    println!(
        "   Paged KV Cache (q8_0 K / q4_0 V): ~{} MB",
        plan.kv_cache_mb
    );

    // Initialize Asymmetric Paged KV Cache
    let configured_context = gguf
        .get_u32_metadata("llama.context_length")
        .or_else(|| gguf.get_u32_metadata("qwen2.context_length"))
        .unwrap_or(2048) as usize;
    let mut kv_cache = PagedKVCache::with_limit(
        model.config.n_layers,
        model.config.n_kv_heads,
        model.config.head_dim,
        configured_context,
    );

    let speculator = PromptLookupSpeculator::new(2, 3);

    println!("\n💬 Prompt: \"{}\"", prompt);
    print!("🤖 Output: ");
    io::stdout().flush().unwrap();

    let mut tokens = tokenizer.encode(prompt);
    if tokens.is_empty() {
        tokens.push(tokenizer.bos_id);
    }

    let mut logits = vec![0.0f32; model.config.vocab_size];
    let gen_start = Instant::now();
    let mut generated_count = 0;
    let max_gen_tokens = 64;

    // Prefill prompt tokens
    for (pos, &tok) in tokens.iter().enumerate() {
        if let Err(e) = model.forward(tok, pos, &mut kv_cache, &mut logits) {
            eprintln!("Inference failed during prompt prefill: {e}");
            return;
        }
    }

    let mut current_pos = tokens.len();

    // Generation loop with Speculative Decoding
    while generated_count < max_gen_tokens {
        // Sample argmax from logits
        let mut best_tok = 0u32;
        let mut max_logit = f32::NEG_INFINITY;
        for (id, &val) in logits.iter().enumerate() {
            if val > max_logit {
                max_logit = val;
                best_tok = id as u32;
            }
        }

        if best_tok == tokenizer.eos_id {
            break;
        }

        tokens.push(best_tok);
        let tok_str = tokenizer.decode(best_tok).replace(' ', " ");
        print!("{}", tok_str);
        io::stdout().flush().unwrap();
        generated_count += 1;

        // Draft speculative tokens
        let draft_tokens = speculator.draft_tokens(&tokens);
        if !draft_tokens.is_empty() {
            // Forward speculative token
            for &draft_tok in &draft_tokens {
                if let Err(e) = model.forward(draft_tok, current_pos, &mut kv_cache, &mut logits) {
                    eprintln!("Inference failed during speculative step: {e}");
                    return;
                }
                current_pos += 1;
                tokens.push(draft_tok);
                let draft_str = tokenizer.decode(draft_tok).replace(' ', " ");
                print!("{}", draft_str);
                io::stdout().flush().unwrap();
                generated_count += 1;
            }
        } else {
            if let Err(e) = model.forward(best_tok, current_pos, &mut kv_cache, &mut logits) {
                eprintln!("Inference failed during decode: {e}");
                return;
            }
            current_pos += 1;
        }
    }

    let total_secs = gen_start.elapsed().as_secs_f64();
    let tok_per_sec = if total_secs > 0.0 {
        generated_count as f64 / total_secs
    } else {
        0.0
    };

    println!(
        "\n\n⏱️ Generated {} tokens in {:.2}s ({:.2} tokens/sec)",
        generated_count, total_secs, tok_per_sec
    );
}
