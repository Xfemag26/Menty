#![allow(dead_code, unused_imports)]
// lib.rs - infer-rs high-performance inference engine library

pub mod avx;
pub mod cuda;
pub mod gguf;
pub mod kv_cache;
pub mod model;
pub mod ops;
pub mod partition;
pub mod quant;
pub mod speculative;
pub mod tokenizer;

pub use gguf::GGUFFile;
pub use kv_cache::PagedKVCache;
pub use model::{ModelConfig, TransformerModel};
pub use partition::HelmPartitioner;
pub use speculative::PromptLookupSpeculator;
use std::sync::Arc;
pub use tokenizer::Tokenizer;

/// High-level in-process inference engine
pub struct InferEngine {
    pub gguf: Arc<GGUFFile>,
    pub tokenizer: Tokenizer,
    pub model_path: String,
}

impl InferEngine {
    pub fn load(model_path: &str) -> Result<Self, String> {
        let gguf = GGUFFile::open(model_path).map_err(|e| format!("{e}"))?;
        let tokenizer = Tokenizer::from_gguf(&gguf)?;
        Ok(Self {
            gguf: Arc::new(gguf),
            tokenizer,
            model_path: model_path.to_string(),
        })
    }

    /// Complete prompt with optional streaming callback
    pub fn complete<F: FnMut(&str) -> bool>(
        &self,
        prompt: &str,
        max_tokens: usize,
        mut on_token: F,
    ) -> Result<String, String> {
        let mut tokens = self.tokenizer.encode(prompt);
        if tokens.is_empty() {
            tokens.push(1); // default BOS if available
        }

        let model = TransformerModel::from_gguf(&self.gguf)?;
        let mut kv_cache = PagedKVCache::with_limit(
            model.config.n_layers,
            model.config.n_kv_heads,
            model.config.head_dim,
            self.gguf
                .get_u32_metadata("llama.context_length")
                .or_else(|| self.gguf.get_u32_metadata("qwen2.context_length"))
                .unwrap_or(2048) as usize,
        );
        let speculator = PromptLookupSpeculator::new(2, 3);
        let mut logits = vec![0.0f32; model.config.vocab_size];

        // Prefill prompt
        for (pos, &tok) in tokens.iter().enumerate() {
            model.forward(tok, pos, &mut kv_cache, &mut logits)?;
        }

        let mut current_pos = tokens.len();
        let mut generated_count = 0;
        let mut output_text = String::new();

        // Generation loop with Speculative Decoding
        while generated_count < max_tokens {
            let mut best_tok = 0u32;
            let mut max_logit = f32::NEG_INFINITY;
            for (id, &val) in logits.iter().enumerate() {
                if val > max_logit {
                    max_logit = val;
                    best_tok = id as u32;
                }
            }

            if best_tok == self.tokenizer.eos_id {
                break;
            }

            tokens.push(best_tok);
            let tok_str = self.tokenizer.decode(best_tok).replace(' ', " ");
            output_text.push_str(&tok_str);
            let keep_going = on_token(&tok_str);
            generated_count += 1;
            if !keep_going {
                break;
            }

            let draft_tokens = speculator.draft_tokens(&tokens);
            if !draft_tokens.is_empty() {
                for &draft_tok in &draft_tokens {
                    model.forward(draft_tok, current_pos, &mut kv_cache, &mut logits)?;
                    current_pos += 1;
                    tokens.push(draft_tok);
                    let draft_str = self.tokenizer.decode(draft_tok).replace(' ', " ");
                    output_text.push_str(&draft_str);
                    let keep = on_token(&draft_str);
                    generated_count += 1;
                    if !keep || generated_count >= max_tokens {
                        break;
                    }
                }
            } else {
                model.forward(best_tok, current_pos, &mut kv_cache, &mut logits)?;
                current_pos += 1;
            }
        }

        Ok(output_text)
    }
}
