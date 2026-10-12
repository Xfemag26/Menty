// kv_cache.rs - Asymmetric Paged KV Cache (q8_0 Keys / q4_0 Values)
// Block-based memory manager: 16 tokens per page block
use crate::quant::{dequantize_q4_0, dot_f32_f32, quantize_q4_0, quantize_q8_0};

pub const KV_BLOCK_SIZE: usize = 16;

pub struct PagedKVCache {
    pub n_layers: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub block_size: usize,
    pub max_seq_len: usize,
    // Page blocks: [layer][block_idx] -> K bytes (q8_0) & V bytes (q4_0)
    // For each block of 16 tokens:
    // K size: 16 * n_kv_heads * ((head_dim / 32) * 34 bytes)
    // V size: 16 * n_kv_heads * ((head_dim / 32) * 18 bytes)
    pub k_blocks: Vec<Vec<Vec<u8>>>,
    pub v_blocks: Vec<Vec<Vec<u8>>>,
    pub seq_len: usize,
    k_block_stride: usize,
    v_block_stride: usize,
}

impl PagedKVCache {
    pub fn new(n_layers: usize, n_kv_heads: usize, head_dim: usize) -> Self {
        Self::with_limit(n_layers, n_kv_heads, head_dim, usize::MAX)
    }

    pub fn with_limit(
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        assert_eq!(head_dim % 32, 0, "head_dim must be multiple of 32");
        assert!(max_seq_len > 0, "max_seq_len must be positive");
        let k_token_bytes = n_kv_heads * (head_dim / 32) * 34;
        let v_token_bytes = n_kv_heads * (head_dim / 32) * 18;

        let k_block_stride = KV_BLOCK_SIZE * k_token_bytes;
        let v_block_stride = KV_BLOCK_SIZE * v_token_bytes;

        let mut k_blocks = Vec::with_capacity(n_layers);
        let mut v_blocks = Vec::with_capacity(n_layers);

        for _ in 0..n_layers {
            k_blocks.push(Vec::new());
            v_blocks.push(Vec::new());
        }

        PagedKVCache {
            n_layers,
            n_kv_heads,
            head_dim,
            block_size: KV_BLOCK_SIZE,
            max_seq_len,
            k_blocks,
            v_blocks,
            seq_len: 0,
            k_block_stride,
            v_block_stride,
        }
    }

    /// Appends key and value vectors for a single token at a specific layer
    pub fn append(&mut self, layer: usize, k: &[f32], v: &[f32]) {
        assert!(
            self.seq_len < self.max_seq_len,
            "KV cache context limit exceeded"
        );
        let block_idx = self.seq_len / self.block_size;
        let token_in_block = self.seq_len % self.block_size;

        if block_idx >= self.k_blocks[layer].len() {
            self.k_blocks[layer].push(vec![0u8; self.k_block_stride]);
            self.v_blocks[layer].push(vec![0u8; self.v_block_stride]);
        }

        let k_token_bytes = self.n_kv_heads * (self.head_dim / 32) * 34;
        let v_token_bytes = self.n_kv_heads * (self.head_dim / 32) * 18;

        let k_dst = &mut self.k_blocks[layer][block_idx]
            [token_in_block * k_token_bytes..(token_in_block + 1) * k_token_bytes];
        let v_dst = &mut self.v_blocks[layer][block_idx]
            [token_in_block * v_token_bytes..(token_in_block + 1) * v_token_bytes];

        // Quantize K to q8_0 and V to q4_0
        quantize_q8_0(k, k_dst);
        quantize_q4_0(v, v_dst);
    }

    pub fn advance_token(&mut self) {
        self.seq_len += 1;
    }

    /// Read historical K and V for attention computation
    pub fn compute_attention_scores(
        &self,
        layer: usize,
        head: usize,
        q_head: &[f32],
        out_scores: &mut [f32],
    ) {
        let k_token_bytes = self.n_kv_heads * (self.head_dim / 32) * 34;
        let head_k_bytes = (self.head_dim / 32) * 34;

        // append() has already stored the current token at seq_len; include it in attention.
        for pos in 0..out_scores.len().min(self.seq_len + 1) {
            let b_idx = pos / self.block_size;
            let t_idx = pos % self.block_size;

            let block = &self.k_blocks[layer][b_idx];
            let offset = t_idx * k_token_bytes + head * head_k_bytes;
            let k_bytes = &block[offset..offset + head_k_bytes];

            // Dequantize and dot product with Q
            let mut k_f32 = [0.0f32; 128]; // max head dim support in stack
            let k_slice = &mut k_f32[..self.head_dim];
            dequantize_q8_0(k_bytes, k_slice);

            out_scores[pos] = dot_f32_f32(q_head, k_slice);
        }
    }

    /// Accumulate attention weighted values into output
    pub fn aggregate_values(
        &self,
        layer: usize,
        head: usize,
        attn_probs: &[f32],
        out_v: &mut [f32],
    ) {
        out_v.fill(0.0);
        let v_token_bytes = self.n_kv_heads * (self.head_dim / 32) * 18;
        let head_v_bytes = (self.head_dim / 32) * 18;

        let mut v_f32 = [0.0f32; 128];
        let v_slice = &mut v_f32[..self.head_dim];

        // Include the token appended by the current forward() call.
        for pos in 0..attn_probs.len().min(self.seq_len + 1) {
            let prob = attn_probs[pos];
            if prob < 1e-6 {
                continue;
            }

            let b_idx = pos / self.block_size;
            let t_idx = pos % self.block_size;

            let block = &self.v_blocks[layer][b_idx];
            let offset = t_idx * v_token_bytes + head * head_v_bytes;
            let v_bytes = &block[offset..offset + head_v_bytes];

            dequantize_q4_0(v_bytes, v_slice);

            for d in 0..self.head_dim {
                out_v[d] += prob * v_slice[d];
            }
        }
    }
}

#[inline(always)]
fn dequantize_q8_0(src: &[u8], dst: &mut [f32]) {
    let num_blocks = dst.len() / 32;
    for i in 0..num_blocks {
        let block_bytes = &src[i * 34..(i + 1) * 34];
        let scale = half::f16::from_le_bytes([block_bytes[0], block_bytes[1]]).to_f32();
        let quants = &block_bytes[2..34];
        let out_slice = &mut dst[i * 32..(i + 1) * 32];

        for j in 0..32 {
            out_slice[j] = (quants[j] as i8 as f32) * scale;
        }
    }
}
