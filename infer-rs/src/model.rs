// model.rs - LLaMA / Qwen / Mistral decoder with optional native CUDA layer offload
use crate::cuda::{CudaBackend, CudaLayerMatrices, DevicePtr};
use crate::gguf::{GGMLType, GGUFFile, TensorInfo};
use crate::kv_cache::PagedKVCache;
use crate::ops::*;
use half::f16;

pub struct ModelConfig {
    pub dim: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub hidden_dim: usize,
    pub vocab_size: usize,
    pub head_dim: usize,
    pub norm_eps: f32,
    pub rope_freq_base: f32,
}

pub struct LayerWeights<'a> {
    pub attn_norm: &'a [u8],
    pub wq: &'a [u8],
    pub wq_type: GGMLType,
    pub wk: &'a [u8],
    pub wk_type: GGMLType,
    pub wv: &'a [u8],
    pub wv_type: GGMLType,
    pub wo: &'a [u8],
    pub wo_type: GGMLType,
    pub ffn_norm: &'a [u8],
    pub w_gate: &'a [u8],
    pub w_gate_type: GGMLType,
    pub w_up: &'a [u8],
    pub w_up_type: GGMLType,
    pub w_down: &'a [u8],
    pub w_down_type: GGMLType,
    pub cuda: CudaLayerMatrices,
    pub cuda_weight_bytes: usize,
}

pub struct TransformerModel<'a> {
    pub config: ModelConfig,
    pub token_embd: &'a [u8],
    pub embd_qtype: GGMLType,
    pub layers: Vec<LayerWeights<'a>>,
    pub output_norm: &'a [u8],
    pub output_head: &'a [u8],
    pub head_qtype: GGMLType,
    pub output_head_rows: usize,
}

impl<'a> TransformerModel<'a> {
    pub fn from_gguf(gguf: &'a GGUFFile) -> Result<Self, String> {
        let arch = gguf
            .get_str_metadata("general.architecture")
            .unwrap_or("llama");
        if !matches!(arch, "llama" | "qwen2" | "mistral") {
            return Err(format!(
                "Unsupported GGUF architecture '{arch}'; supported: llama, qwen2, mistral"
            ));
        }
        let meta = |key: &str| format!("{arch}.{key}");
        let u32_meta = |suffix: &str| gguf.get_u32_metadata(&meta(suffix));

        let dim = u32_meta("embedding_length")
            .ok_or_else(|| format!("Missing {arch}.embedding_length"))? as usize;
        let n_layers =
            u32_meta("block_count").ok_or_else(|| format!("Missing {arch}.block_count"))? as usize;
        let n_heads = u32_meta("attention.head_count")
            .ok_or_else(|| format!("Missing {arch}.attention.head_count"))?
            as usize;
        let n_kv_heads = u32_meta("attention.head_count_kv").unwrap_or(n_heads as u32) as usize;
        let hidden_dim = u32_meta("feed_forward_length")
            .ok_or_else(|| format!("Missing {arch}.feed_forward_length"))?
            as usize;
        let vocab_size = gguf
            .get_u32_metadata(&meta("vocab_size"))
            .map(|v| v as usize)
            .or_else(|| match gguf.metadata.get("tokenizer.ggml.tokens") {
                Some(crate::gguf::MetadataValue::Array(v)) => Some(v.len()),
                _ => None,
            })
            .ok_or_else(|| "Cannot determine vocabulary size".to_string())?;
        if dim == 0
            || n_layers == 0
            || n_heads == 0
            || n_kv_heads == 0
            || hidden_dim == 0
            || vocab_size == 0
        {
            return Err("GGUF model metadata contains a zero-sized architecture dimension".into());
        }
        if dim % n_heads != 0 || n_heads % n_kv_heads != 0 {
            return Err(format!(
                "Invalid attention shape: dim={dim}, heads={n_heads}, kv_heads={n_kv_heads}"
            ));
        }
        let head_dim = u32_meta("attention.key_length")
            .map(|v| v as usize)
            .unwrap_or(dim / n_heads);
        let norm_eps = gguf
            .get_f32_metadata(&meta("attention.layer_norm_rms_epsilon"))
            .unwrap_or(1e-5);
        let rope_freq_base = gguf
            .get_f32_metadata(&meta("rope.freq_base"))
            .unwrap_or(10000.0);
        let config = ModelConfig {
            dim,
            n_layers,
            n_heads,
            n_kv_heads,
            hidden_dim,
            vocab_size,
            head_dim,
            norm_eps,
            rope_freq_base,
        };

        let token_embd_info =
            find_tensor(gguf, &["token_embd.weight", "model.embed_tokens.weight"])?;
        let output_norm_info = find_tensor(gguf, &["output_norm.weight", "model.norm.weight"])?;
        let output_head_info =
            find_tensor(gguf, &["output.weight", "lm_head.weight"]).unwrap_or(token_embd_info);
        let output_head_rows = tensor_rows(output_head_info, dim, vocab_size)?;

        let mut layers = Vec::with_capacity(n_layers);
        for layer in 0..n_layers {
            let attn_norm_info = find_layer_tensor(
                gguf,
                layer,
                &["attn_norm.weight", "blk.{}.attn_norm.weight"],
            )?;
            let wq = find_layer_tensor(gguf, layer, &["attn_q.weight", "blk.{}.attn_q.weight"])?;
            let wk = find_layer_tensor(gguf, layer, &["attn_k.weight", "blk.{}.attn_k.weight"])?;
            let wv = find_layer_tensor(gguf, layer, &["attn_v.weight", "blk.{}.attn_v.weight"])?;
            let wo = find_layer_tensor(
                gguf,
                layer,
                &["attn_output.weight", "blk.{}.attn_output.weight"],
            )?;
            let ffn_norm_info =
                find_layer_tensor(gguf, layer, &["ffn_norm.weight", "blk.{}.ffn_norm.weight"])?;
            let w_gate =
                find_layer_tensor(gguf, layer, &["ffn_gate.weight", "blk.{}.ffn_gate.weight"])?;
            let w_up = find_layer_tensor(gguf, layer, &["ffn_up.weight", "blk.{}.ffn_up.weight"])?;
            let w_down =
                find_layer_tensor(gguf, layer, &["ffn_down.weight", "blk.{}.ffn_down.weight"])?;

            let layer_bytes = [
                tensor_gpu_bytes(wq, dim, dim)?,
                tensor_gpu_bytes(wk, n_kv_heads * head_dim, dim)?,
                tensor_gpu_bytes(wv, n_kv_heads * head_dim, dim)?,
                tensor_gpu_bytes(wo, dim, dim)?,
                tensor_gpu_bytes(w_gate, hidden_dim, dim)?,
                tensor_gpu_bytes(w_up, hidden_dim, dim)?,
                tensor_gpu_bytes(w_down, dim, hidden_dim)?,
            ]
            .iter()
            .sum();

            layers.push(LayerWeights {
                attn_norm: gguf.get_tensor_data(attn_norm_info),
                wq: gguf.get_tensor_data(wq),
                wq_type: wq.qtype,
                wk: gguf.get_tensor_data(wk),
                wk_type: wk.qtype,
                wv: gguf.get_tensor_data(wv),
                wv_type: wv.qtype,
                wo: gguf.get_tensor_data(wo),
                wo_type: wo.qtype,
                ffn_norm: gguf.get_tensor_data(ffn_norm_info),
                w_gate: gguf.get_tensor_data(w_gate),
                w_gate_type: w_gate.qtype,
                w_up: gguf.get_tensor_data(w_up),
                w_up_type: w_up.qtype,
                w_down: gguf.get_tensor_data(w_down),
                w_down_type: w_down.qtype,
                cuda: CudaLayerMatrices::default(),
                cuda_weight_bytes: layer_bytes,
            });
        }

        Ok(Self {
            config,
            token_embd: gguf.get_tensor_data(token_embd_info),
            embd_qtype: token_embd_info.qtype,
            layers,
            output_norm: gguf.get_tensor_data(output_norm_info),
            output_head: gguf.get_tensor_data(output_head_info),
            head_qtype: output_head_info.qtype,
            output_head_rows,
        })
    }

    pub fn estimated_gpu_layer_bytes(&self, layer: usize) -> usize {
        self.layers
            .get(layer)
            .map(|l| l.cuda_weight_bytes)
            .unwrap_or(usize::MAX)
    }

    /// Mean-pool the model's learned token-embedding rows into one normalized
    /// vector. This is a lightweight local retrieval embedding, not a separate
    /// BERT/SentenceTransformer pass.
    pub fn mean_token_embedding(&self, token_ids: &[u32]) -> Result<Vec<f32>, String> {
        let mut pooled = vec![0.0f32; self.config.dim];
        let mut count = 0usize;
        let row_bytes = tensor_row_bytes(self.embd_qtype, self.config.dim)?;
        for token in token_ids {
            if *token as usize >= self.config.vocab_size {
                continue;
            }
            let start = (*token as usize)
                .checked_mul(row_bytes)
                .ok_or_else(|| "embedding offset overflow".to_string())?;
            let end = start
                .checked_add(row_bytes)
                .ok_or_else(|| "embedding range overflow".to_string())?;
            if end > self.token_embd.len() {
                return Err("token embedding row is outside tensor bounds".into());
            }
            let mut row = vec![0.0f32; self.config.dim];
            dequantize_row(&self.token_embd[start..end], self.embd_qtype, &mut row)?;
            for (sum, value) in pooled.iter_mut().zip(row) {
                *sum += value;
            }
            count += 1;
        }
        if count == 0 {
            return Err("no usable token embeddings".into());
        }
        for value in &mut pooled {
            *value /= count as f32;
        }
        let norm = pooled.iter().map(|value| value * value).sum::<f32>().sqrt();
        if norm > 0.0 {
            for value in &mut pooled {
                *value /= norm;
            }
        }
        Ok(pooled)
    }

    /// Dequantize and upload one transformer layer as FP16, returning resident bytes.
    /// Caller checkpoints/rolls back allocations if an upload fails part-way through.
    pub fn upload_layer_to_cuda(
        &mut self,
        layer_idx: usize,
        cuda: &mut CudaBackend,
    ) -> Result<usize, String> {
        let layer = self
            .layers
            .get(layer_idx)
            .ok_or_else(|| format!("invalid layer index {layer_idx}"))?;
        let d = &self.config;
        let dimensions = [
            (layer.wq, layer.wq_type, d.dim, d.dim),
            (layer.wk, layer.wk_type, d.n_kv_heads * d.head_dim, d.dim),
            (layer.wv, layer.wv_type, d.n_kv_heads * d.head_dim, d.dim),
            (layer.wo, layer.wo_type, d.dim, d.dim),
            (layer.w_gate, layer.w_gate_type, d.hidden_dim, d.dim),
            (layer.w_up, layer.w_up_type, d.hidden_dim, d.dim),
            (layer.w_down, layer.w_down_type, d.dim, d.hidden_dim),
        ];
        let checkpoint = cuda.allocation_checkpoint();
        let mut pointers = [0 as DevicePtr; 7];
        let uploaded = (|| {
            for (i, (bytes, qtype, rows, cols)) in dimensions.iter().enumerate() {
                let data = matrix_to_f16(bytes, *qtype, *rows, *cols)?;
                pointers[i] = cuda.upload_f16(&data)?;
            }
            Ok::<(), String>(())
        })();
        if let Err(error) = uploaded {
            cuda.rollback_allocations(checkpoint);
            return Err(error);
        }
        let layer = &mut self.layers[layer_idx];
        layer.cuda = CudaLayerMatrices { weights: pointers };
        Ok(layer.cuda_weight_bytes)
    }

    /// Single token forward pass using optional partially offloaded CUDA layers.
    pub fn forward(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut PagedKVCache,
        logits: &mut [f32],
    ) -> Result<(), String> {
        self.forward_with_cuda(token, pos, kv_cache, logits, None)
    }

    pub fn forward_with_cuda(
        &self,
        token: u32,
        pos: usize,
        kv_cache: &mut PagedKVCache,
        logits: &mut [f32],
        mut cuda: Option<&mut CudaBackend>,
    ) -> Result<(), String> {
        let dim = self.config.dim;
        if token as usize >= self.config.vocab_size {
            return Err(format!(
                "token id {token} exceeds model vocabulary {}",
                self.config.vocab_size
            ));
        }
        let mut x = vec![0.0f32; dim];
        let row_bytes = tensor_row_bytes(self.embd_qtype, dim)?;
        let start = (token as usize)
            .checked_mul(row_bytes)
            .ok_or_else(|| "embedding offset overflow".to_string())?;
        let end = start
            .checked_add(row_bytes)
            .ok_or_else(|| "embedding range overflow".to_string())?;
        if end > self.token_embd.len() {
            return Err(format!("token embedding {token} is outside tensor bounds"));
        }
        dequantize_row(&self.token_embd[start..end], self.embd_qtype, &mut x)?;

        let mut x_norm = vec![0.0f32; dim];
        let mut q = vec![0.0f32; self.config.n_heads * self.config.head_dim];
        let mut k = vec![0.0f32; self.config.n_kv_heads * self.config.head_dim];
        let mut v = vec![0.0f32; self.config.n_kv_heads * self.config.head_dim];
        let mut attn_out = vec![0.0f32; dim];
        let mut gate = vec![0.0f32; self.config.hidden_dim];
        let mut up = vec![0.0f32; self.config.hidden_dim];
        let mut ffn_out = vec![0.0f32; dim];

        for (layer_index, layer) in self.layers.iter().enumerate() {
            x_norm.copy_from_slice(&x);
            let attn_norm = tensor_as_f32(layer.attn_norm, dim)?;
            rms_norm(&mut x_norm, &attn_norm, self.config.norm_eps);

            linear(
                &mut cuda,
                layer.cuda,
                0,
                layer.wq,
                layer.wq_type,
                &x_norm,
                &mut q,
                dim,
            )?;
            linear(
                &mut cuda,
                layer.cuda,
                1,
                layer.wk,
                layer.wk_type,
                &x_norm,
                &mut k,
                dim,
            )?;
            linear(
                &mut cuda,
                layer.cuda,
                2,
                layer.wv,
                layer.wv_type,
                &x_norm,
                &mut v,
                dim,
            )?;

            apply_rope(
                &mut q,
                pos,
                self.config.head_dim,
                self.config.rope_freq_base,
            );
            apply_rope(
                &mut k,
                pos,
                self.config.head_dim,
                self.config.rope_freq_base,
            );
            kv_cache.append(layer_index, &k, &v);

            let head_dim = self.config.head_dim;
            let scale = 1.0 / (head_dim as f32).sqrt();
            for h in 0..self.config.n_heads {
                let kv_h = h / (self.config.n_heads / self.config.n_kv_heads);
                let q_h = &q[h * head_dim..(h + 1) * head_dim];
                let mut scores = vec![0.0f32; pos + 1];
                kv_cache.compute_attention_scores(layer_index, kv_h, q_h, &mut scores);
                for score in &mut scores {
                    *score *= scale;
                }
                softmax(&mut scores);
                let head_out = &mut attn_out[h * head_dim..(h + 1) * head_dim];
                kv_cache.aggregate_values(layer_index, kv_h, &scores, head_out);
            }

            let mut proj_out = vec![0.0f32; dim];
            linear(
                &mut cuda,
                layer.cuda,
                3,
                layer.wo,
                layer.wo_type,
                &attn_out,
                &mut proj_out,
                dim,
            )?;
            for (value, proj) in x.iter_mut().zip(proj_out) {
                *value += proj;
            }

            x_norm.copy_from_slice(&x);
            let ffn_norm = tensor_as_f32(layer.ffn_norm, dim)?;
            rms_norm(&mut x_norm, &ffn_norm, self.config.norm_eps);
            linear(
                &mut cuda,
                layer.cuda,
                4,
                layer.w_gate,
                layer.w_gate_type,
                &x_norm,
                &mut gate,
                dim,
            )?;
            linear(
                &mut cuda,
                layer.cuda,
                5,
                layer.w_up,
                layer.w_up_type,
                &x_norm,
                &mut up,
                dim,
            )?;
            swiglu(&mut gate, &up);
            linear(
                &mut cuda,
                layer.cuda,
                6,
                layer.w_down,
                layer.w_down_type,
                &gate,
                &mut ffn_out,
                self.config.hidden_dim,
            )?;
            for (value, ffn) in x.iter_mut().zip(&ffn_out) {
                *value += *ffn;
            }
        }

        kv_cache.advance_token();
        let output_norm = tensor_as_f32(self.output_norm, dim)?;
        rms_norm(&mut x, &output_norm, self.config.norm_eps);
        matvec_cpu(self.output_head, self.head_qtype, &x, logits, dim)?;
        Ok(())
    }
}

fn linear(
    cuda: &mut Option<&mut CudaBackend>,
    matrices: CudaLayerMatrices,
    matrix_index: usize,
    weights: &[u8],
    qtype: GGMLType,
    input: &[f32],
    output: &mut [f32],
    cols: usize,
) -> Result<(), String> {
    let pointer = matrices.weights[matrix_index];
    if pointer != 0 {
        if let Some(backend) = cuda.as_deref_mut() {
            if backend.disabled_reason().is_none() {
                match backend.matvec(pointer, input, output, output.len(), cols) {
                    Ok(()) => return Ok(()),
                    Err(error) => {
                        eprintln!("CUDA inference failed; disabling GPU backend and retrying this layer on CPU: {error}");
                        backend.disable(error);
                    }
                }
            }
        }
    }
    matvec_cpu(weights, qtype, input, output, cols)
}

fn matvec_cpu(
    weights: &[u8],
    qtype: GGMLType,
    input: &[f32],
    output: &mut [f32],
    cols: usize,
) -> Result<(), String> {
    if input.len() != cols {
        return Err(format!(
            "matvec input {} does not match matrix width {cols}",
            input.len()
        ));
    }
    match qtype {
        GGMLType::Q4_K => matmul_q4_k(weights, input, output, cols, output.len()),
        GGMLType::Q4_0 => matmul_q4_0(weights, input, output, cols, output.len()),
        GGMLType::Q6_K => matmul_q6_k(weights, input, output, cols, output.len()),
        GGMLType::Q8_0 => matmul_q8_0(weights, input, output, cols)?,
        GGMLType::F32 => {
            let values: &[f32] = bytemuck::try_cast_slice(weights)
                .map_err(|e| format!("invalid F32 matrix: {e}"))?;
            if values.len() < output.len() * cols {
                return Err("F32 matrix tensor is too small".into());
            }
            for (row, target) in output.iter_mut().enumerate() {
                *target = crate::quant::dot_f32_f32(&values[row * cols..(row + 1) * cols], input);
            }
        }
        GGMLType::F16 => {
            let expected = output
                .len()
                .checked_mul(cols)
                .ok_or_else(|| "F16 matrix size overflow".to_string())?;
            if weights.len() < expected * 2 {
                return Err("F16 matrix tensor is too small".into());
            }
            for (row, target) in output.iter_mut().enumerate() {
                let mut sum = 0.0f32;
                for col in 0..cols {
                    let offset = (row * cols + col) * 2;
                    let w = f16::from_le_bytes([weights[offset], weights[offset + 1]]).to_f32();
                    sum += w * input[col];
                }
                *target = sum;
            }
        }
        other => {
            return Err(format!(
                "CPU matvec does not support tensor quantization {other:?}"
            ))
        }
    }
    Ok(())
}

fn matmul_q8_0(
    weights: &[u8],
    input: &[f32],
    output: &mut [f32],
    cols: usize,
) -> Result<(), String> {
    if cols % 32 != 0 {
        return Err(format!("Q8_0 width {cols} must be divisible by 32"));
    }
    let row_bytes = cols / 32 * 34;
    if weights.len() < output.len() * row_bytes {
        return Err("Q8_0 tensor is too small for requested matrix shape".into());
    }
    for (row, target) in output.iter_mut().enumerate() {
        let mut sum = 0.0f32;
        let row = &weights[row * row_bytes..(row + 1) * row_bytes];
        for block in 0..cols / 32 {
            let offset = block * 34;
            let scale = f16::from_le_bytes([row[offset], row[offset + 1]]).to_f32();
            for j in 0..32 {
                sum += (row[offset + 2 + j] as i8 as f32) * scale * input[block * 32 + j];
            }
        }
        *target = sum;
    }
    Ok(())
}

fn matrix_to_f16(
    bytes: &[u8],
    qtype: GGMLType,
    rows: usize,
    cols: usize,
) -> Result<Vec<f16>, String> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    let mut values = vec![0.0f32; elements];
    match qtype {
        GGMLType::Q4_K if elements % 256 == 0 => crate::quant::dequantize_q4_k(bytes, &mut values),
        GGMLType::Q4_0 if elements % 32 == 0 => crate::quant::dequantize_q4_0(bytes, &mut values),
        GGMLType::Q6_K if elements % 256 == 0 => crate::quant::dequantize_q6_k(bytes, &mut values),
        GGMLType::Q8_0 => {
            if elements % 32 != 0 || bytes.len() < elements / 32 * 34 {
                return Err("invalid Q8_0 matrix layout".into());
            }
            for block in 0..elements / 32 {
                let offset = block * 34;
                let scale = f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
                for j in 0..32 {
                    values[block * 32 + j] = bytes[offset + 2 + j] as i8 as f32 * scale;
                }
            }
        }
        GGMLType::F32 => {
            let source: &[f32] =
                bytemuck::try_cast_slice(bytes).map_err(|e| format!("invalid F32 matrix: {e}"))?;
            if source.len() < elements {
                return Err("F32 matrix tensor is too small".into());
            }
            values.copy_from_slice(&source[..elements]);
        }
        GGMLType::F16 => {
            if bytes.len() < elements * 2 {
                return Err("F16 matrix tensor is too small".into());
            }
            return Ok(bytes[..elements * 2]
                .chunks_exact(2)
                .map(|b| f16::from_le_bytes([b[0], b[1]]))
                .collect());
        }
        other => {
            return Err(format!(
                "CUDA FP16 layer upload does not support tensor format {other:?}"
            ))
        }
    }
    Ok(values.into_iter().map(f16::from_f32).collect())
}

fn dequantize_row(bytes: &[u8], qtype: GGMLType, output: &mut [f32]) -> Result<(), String> {
    match qtype {
        GGMLType::Q4_K if output.len() % 256 == 0 => crate::quant::dequantize_q4_k(bytes, output),
        GGMLType::Q4_0 if output.len() % 32 == 0 => crate::quant::dequantize_q4_0(bytes, output),
        GGMLType::Q6_K if output.len() % 256 == 0 => crate::quant::dequantize_q6_k(bytes, output),
        GGMLType::Q8_0 => {
            if output.len() % 32 != 0 || bytes.len() < output.len() / 32 * 34 {
                return Err("invalid Q8_0 embedding row".into());
            }
            for block in 0..output.len() / 32 {
                let offset = block * 34;
                let scale = f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
                for j in 0..32 {
                    output[block * 32 + j] = bytes[offset + 2 + j] as i8 as f32 * scale;
                }
            }
        }
        GGMLType::F16 => {
            if bytes.len() < output.len() * 2 {
                return Err("F16 embedding row is too small".into());
            }
            for (index, value) in output.iter_mut().enumerate() {
                let offset = index * 2;
                *value = f16::from_le_bytes([bytes[offset], bytes[offset + 1]]).to_f32();
            }
        }
        GGMLType::F32 => {
            let source: &[f32] = bytemuck::try_cast_slice(bytes)
                .map_err(|e| format!("invalid F32 embedding row: {e}"))?;
            if source.len() < output.len() {
                return Err("F32 embedding row is too small".into());
            }
            output.copy_from_slice(&source[..output.len()]);
        }
        other => {
            return Err(format!(
                "embedding lookup does not support tensor format {other:?}"
            ))
        }
    }
    Ok(())
}

fn tensor_as_f32(bytes: &[u8], width: usize) -> Result<Vec<f32>, String> {
    if bytes.len() < width * 4 {
        return Err("normalization tensor is smaller than the model width".into());
    }
    let values: &[f32] = bytemuck::try_cast_slice(&bytes[..width * 4])
        .map_err(|e| format!("invalid normalization tensor: {e}"))?;
    Ok(values.to_vec())
}

fn tensor_row_bytes(qtype: GGMLType, elements: usize) -> Result<usize, String> {
    match qtype {
        GGMLType::Q4_K if elements % 256 == 0 => Ok(elements / 256 * 144),
        GGMLType::Q4_0 if elements % 32 == 0 => Ok(elements / 32 * 18),
        GGMLType::Q6_K if elements % 256 == 0 => Ok(elements / 256 * 210),
        GGMLType::Q8_0 if elements % 32 == 0 => Ok(elements / 32 * 34),
        GGMLType::F32 => Ok(elements * 4),
        GGMLType::F16 => Ok(elements * 2),
        other => Err(format!(
            "Unsupported embedding tensor format {other:?} or invalid row size {elements}"
        )),
    }
}

fn tensor_gpu_bytes(info: &TensorInfo, rows: usize, cols: usize) -> Result<usize, String> {
    let elements = rows
        .checked_mul(cols)
        .ok_or_else(|| "matrix element count overflow".to_string())?;
    if info.dims.iter().map(|d| *d as usize).product::<usize>() < elements {
        return Err(format!(
            "tensor '{}' dimensions do not match expected matrix {rows}x{cols}",
            info.name
        ));
    }
    // FP16 working weights; allocator reserve and workspace are budgeted separately.
    elements
        .checked_mul(2)
        .ok_or_else(|| "FP16 GPU matrix size overflow".to_string())
}

fn tensor_rows(info: &TensorInfo, cols: usize, default_rows: usize) -> Result<usize, String> {
    let elements = info.dims.iter().map(|d| *d as usize).product::<usize>();
    if cols == 0 || elements % cols != 0 {
        return Ok(default_rows);
    }
    Ok(elements / cols)
}

fn find_tensor<'a>(gguf: &'a GGUFFile, candidates: &[&str]) -> Result<&'a TensorInfo, String> {
    candidates
        .iter()
        .find_map(|name| gguf.tensors.get(*name))
        .ok_or_else(|| format!("Missing required tensor (tried {})", candidates.join(", ")))
}

fn find_layer_tensor<'a>(
    gguf: &'a GGUFFile,
    layer: usize,
    candidates: &[&str],
) -> Result<&'a TensorInfo, String> {
    for pattern in candidates {
        let name = pattern.replace("{}", &layer.to_string());
        if let Some(info) = gguf.tensors.get(&name) {
            return Ok(info);
        }
    }
    Err(format!(
        "Could not find tensor for layer {layer} (tried {})",
        candidates.join(", ")
    ))
}
