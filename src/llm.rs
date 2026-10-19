// llm.rs - in-process Rust inference backed by infer-rs; optional CUDA driver offload
use crate::config::{self, ModelProfile, MODEL_PROFILES};
use infer_rs::cuda::CudaBackend;
use infer_rs::gguf::GGUFFile;
use infer_rs::kv_cache::PagedKVCache;
use infer_rs::model::TransformerModel;
use infer_rs::tokenizer::Tokenizer;
use std::sync::{Mutex, OnceLock};
use tokio::task::JoinHandle;

const GPU_RESERVE_MB: usize = 192;

static PREFETCH_TASK: OnceLock<Mutex<Option<JoinHandle<()>>>> = OnceLock::new();

fn prefetch_task() -> &'static Mutex<Option<JoinHandle<()>>> {
    PREFETCH_TASK.get_or_init(|| Mutex::new(None))
}

#[derive(Debug, Clone)]
pub struct GpuStatus {
    pub device_name: Option<String>,
    pub total_mb: usize,
    pub free_mb: usize,
    pub loaded_layers: usize,
    pub target_layers: usize,
    pub layers_remaining: usize,
    pub additional_mb_needed: usize,
    pub failure_reason: Option<String>,
}

impl GpuStatus {
    pub fn is_constrained(&self) -> bool {
        self.device_name.is_some() && self.loaded_layers < self.target_layers
    }
}

pub struct InferenceRuntime {
    pub gpu_status: GpuStatus,
}

struct LoadedModel {
    model: TransformerModel<'static>,
    tokenizer: Tokenizer,
    cuda: Option<CudaBackend>,
    embedding_model: String,
    ctx_size: usize,
    max_tokens: u32,
    _gguf_owner: Box<GGUFFile>,
}

static ACTIVE_ENGINE: Mutex<Option<LoadedModel>> = Mutex::new(None);

impl InferenceRuntime {
    pub fn start() -> anyhow::Result<Self> {
        let profile = MODEL_PROFILES
            .first()
            .ok_or_else(|| anyhow::anyhow!("No local model profiles are configured"))?;
        Self::start_with_profile(profile)
    }

    pub fn start_with_profile(profile: &ModelProfile) -> anyhow::Result<Self> {
        let model_path = config::resolve_blob(profile)
            .ok_or_else(|| anyhow::anyhow!(
                "Model blob not found for profile '{}' (sha prefix: {}). Check configured model directories / mounted AI SSD.",
                profile.name, profile.blob_sha
            ))?;

        println!("Loading native model [{}] from {}", profile.name, model_path);

        let gguf_owner = Box::new(
            GGUFFile::open(&model_path).map_err(|e| anyhow::anyhow!("GGUF load failed: {e}"))?,
        );

        let gguf: &'static GGUFFile = unsafe { &*(gguf_owner.as_ref() as *const GGUFFile) };
        let tokenizer = Tokenizer::from_gguf(gguf)
            .map_err(|e| anyhow::anyhow!("Tokenizer load failed: {e}"))?;
        let mut model = TransformerModel::from_gguf(gguf)
            .map_err(|e| anyhow::anyhow!("Unsupported/invalid model: {e}"))?;

        let trained_context = gguf
            .get_u32_metadata("llama.context_length")
            .or_else(|| gguf.get_u32_metadata("qwen2.context_length"))
            .or_else(|| gguf.get_u32_metadata("mistral.context_length"))
            .unwrap_or(profile.ctx_size) as usize;

        let ctx_size = (profile.ctx_size as usize).min(trained_context).max(1);
        let max_tokens = profile.max_tokens.max(1).min(ctx_size as u32);

        let mut gpu_status = GpuStatus {
            device_name: None,
            total_mb: 0,
            free_mb: 0,
            loaded_layers: 0,
            target_layers: (profile.gpu_layers as usize).min(model.config.n_layers),
            layers_remaining: 0,
            additional_mb_needed: 0,
            failure_reason: None,
        };

        let mut cuda_backend = None;
        if gpu_status.target_layers > 0 {
            let max_dimension = model.config.hidden_dim.max(model.config.dim);
            match CudaBackend::new(max_dimension, max_dimension) {
                Ok(mut backend) => {
                    gpu_status.device_name = Some(backend.device_info.name.clone());
                    gpu_status.total_mb = backend.device_info.total_bytes / (1024 * 1024);
                    let free = backend.free_bytes().unwrap_or(0);
                    gpu_status.free_mb = free / (1024 * 1024);

                    let reserve = GPU_RESERVE_MB.saturating_mul(1024 * 1024);
                    let mut remaining_budget = free.saturating_sub(reserve);

                    for layer_idx in 0..gpu_status.target_layers {
                        let layer_bytes = model.estimated_gpu_layer_bytes(layer_idx);
                        if layer_bytes > remaining_budget {
                            break;
                        }

                        let before = backend.allocation_checkpoint();
                        match model.upload_layer_to_cuda(layer_idx, &mut backend) {
                            Ok(_) => {
                                let after = backend.allocation_checkpoint();
                                let actual_bytes = model.estimated_gpu_layer_bytes(layer_idx);
                                if after == before || actual_bytes == 0 {
                                    break;
                                }
                                remaining_budget = remaining_budget.saturating_sub(actual_bytes);
                                gpu_status.loaded_layers += 1;
                            }
                            Err(error) => {
                                backend.rollback_allocations(before);
                                gpu_status.failure_reason = Some(error);
                                break;
                            }
                        }
                    }

                    let free_after = backend.free_bytes().unwrap_or(0);
                    gpu_status.free_mb = free_after / (1024 * 1024);
                    gpu_status.layers_remaining = gpu_status
                        .target_layers
                        .saturating_sub(gpu_status.loaded_layers);

                    if gpu_status.layers_remaining > 0 {
                        let next_layer = model.estimated_gpu_layer_bytes(gpu_status.loaded_layers);
                        let needed = next_layer.saturating_sub(free_after.saturating_sub(reserve));
                        gpu_status.additional_mb_needed =
                            needed.saturating_add(1024 * 1024 - 1) / (1024 * 1024);
                    }

                    cuda_backend = Some(backend);
                }
                Err(error) => {
                    gpu_status.failure_reason = Some(error);
                }
            }
        }

        let mut engine = ACTIVE_ENGINE
            .lock()
            .map_err(|_| anyhow::anyhow!("Inference engine lock poisoned"))?;

        *engine = Some(LoadedModel {
            model,
            tokenizer,
            cuda: cuda_backend,
            embedding_model: profile.name.to_string(),
            ctx_size,
            max_tokens,
            _gguf_owner: gguf_owner,
        });

        println!(
            "Native inference ready [{}] — context {}, max output {} tokens, GPU layers {}/{}",
            profile.name, ctx_size, max_tokens, gpu_status.loaded_layers, gpu_status.target_layers
        );

        Ok(Self { gpu_status })
    }
}

impl Drop for InferenceRuntime {
    fn drop(&mut self) {
        if let Ok(mut engine) = ACTIVE_ENGINE.lock() {
            *engine = None;
        }
    }
}

pub fn async_prefetch_model(model_path: &str) {
    let path = model_path.to_string();
    let task = {
        let guard = prefetch_task().lock().unwrap();
        if let Some(existing) = guard.as_ref() {
            if !existing.is_finished() {
                return;
            }
        }
        drop(guard);
        tokio::spawn(async move {
            let _ = std::fs::File::open(&path)
                .and_then(|f| f.metadata())
                .map(|_| ());
            tokio::task::yield_now().await;
            let _ = std::fs::metadata(&path);
        })
    };

    let mut guard = prefetch_task().lock().unwrap();
    *guard = Some(task);
}

pub fn async_prewarm_server() {
    let task = tokio::spawn(async {
        tokio::task::yield_now().await;
    });
    let mut guard = prefetch_task().lock().unwrap();
    if guard.as_ref().map(|t| t.is_finished()).unwrap_or(true) {
        *guard = Some(task);
    }
}

pub fn complete(prompt: &str, max_tokens: u32) -> anyhow::Result<String> {
    complete_stream(prompt, max_tokens, |_| {})
}

pub fn complete_stream(
    prompt: &str,
    max_tokens: u32,
    mut on_token: impl FnMut(&str),
) -> anyhow::Result<String> {
    let mut guard = ACTIVE_ENGINE
        .lock()
        .map_err(|_| anyhow::anyhow!("Inference engine lock poisoned"))?;
    let engine = guard
        .as_mut()
        .ok_or_else(|| anyhow::anyhow!("No model loaded in engine"))?;

    let token_ids = engine.tokenizer.encode(prompt);
    if token_ids.is_empty() {
        return Ok(String::new());
    }

    if token_ids.len() >= engine.ctx_size {
        anyhow::bail!(
            "Prompt needs {} tokens but the active model context is {}. Shorten the prompt or select a larger-context profile.",
            token_ids.len(),
            engine.ctx_size
        );
    }

    let output_budget = max_tokens
        .max(1)
        .min(engine.max_tokens)
        .min((engine.ctx_size - token_ids.len()) as u32);

    let mut kv_cache = PagedKVCache::with_limit(
        engine.model.config.n_layers,
        engine.model.config.n_kv_heads,
        engine.model.config.head_dim,
        engine.ctx_size,
    );

    let mut logits = vec![0.0f32; engine.model.config.vocab_size];
    let model = &engine.model;
    let tokenizer = &engine.tokenizer;
    let cuda = &mut engine.cuda;

    for (position, &token) in token_ids.iter().enumerate() {
        model.forward_with_cuda(token, position, &mut kv_cache, &mut logits, cuda.as_mut())
            .map_err(|e| anyhow::anyhow!("Prompt prefill failed: {e}"))?;
    }

    let mut generated_tokens = Vec::new();
    let mut shown = String::new();

    for generated in 0..output_budget {
        let (token, score) = logits
            .iter()
            .copied()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap_or((0, f32::NEG_INFINITY));

        if token as u32 == tokenizer.eos_id {
            break;
        }

        if !score.is_finite() {
            anyhow::bail!("Model returned non-finite logits; aborting generation");
        }

        generated_tokens.push(token as u32);

        let decoded = tokenizer.decode_tokens_stream(&generated_tokens);
        if decoded.starts_with(&shown) {
            let delta = &decoded[shown.len()..];
            if !delta.is_empty() {
                on_token(delta);
            }
        }
        shown = decoded;

        if generated + 1 < output_budget {
            let next_position = token_ids.len() + generated as usize;
            model.forward_with_cuda(
                token as u32,
                next_position,
                &mut kv_cache,
                &mut logits,
                cuda.as_mut(),
            )
            .map_err(|e| anyhow::anyhow!("Generation failed at token {}: {e}", generated + 1))?;
        }
    }

    Ok(tokenizer.decode_tokens(&generated_tokens).trim().to_string())
}

pub fn complete_batch(
    prompts: &[String],
    max_tokens: u32,
) -> anyhow::Result<Vec<String>> {
    let mut outputs = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        outputs.push(complete(prompt, max_tokens)?);
    }
    Ok(outputs)
}

pub fn complete_stream_batch(
    prompts: &[String],
    max_tokens: u32,
    mut on_token: impl FnMut(&str),
) -> anyhow::Result<Vec<String>> {
    let mut outputs = Vec::with_capacity(prompts.len());
    for prompt in prompts {
        let prompt_copy = prompt.clone();
        outputs.push(complete_stream(&prompt_copy, max_tokens, |piece| {
            on_token(piece);
        })?);
    }
    Ok(outputs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "loads the multi-gigabyte model from the attached SSD and allocates GPU layers"]
    fn loads_ssd_model_and_reports_adaptive_gpu_layers() {
        let runtime = InferenceRuntime::start().expect("configured default GGUF model should load");
        assert!(runtime.gpu_status.loaded_layers <= runtime.gpu_status.target_layers);
        assert!(runtime.gpu_status.device_name.is_some(), "expected CUDA device to be detected");
        let output = complete("Hi", 1).expect("single-token completion should finish");
        assert!(!output.is_empty(), "inference returned no token");
    }
}

pub fn current_embedding_model() -> Option<String> {
    let guard = ACTIVE_ENGINE.lock().ok()?;
    Some(guard.as_ref()?.embedding_model.clone())
}

pub fn embed_text(text: &str) -> Option<(String, Vec<f32>)> {
    let guard = ACTIVE_ENGINE.lock().ok()?;
    let engine = guard.as_ref()?;
    let tokens = engine.tokenizer.encode_retrieval(text);
    let vector = engine.model.mean_token_embedding(&tokens).ok()?;
    Some((engine.embedding_model.clone(), vector))
}

pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}
