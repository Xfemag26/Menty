// ops.rs - High performance neural network operations (RMSNorm, RoPE, Softmax, SiLU/SwiGLU, MatMul)
use crate::quant::{dot_f32_f32, dot_q4_0_q8_0, dot_q4_k_f32, quantize_q8_0};
use rayon::prelude::*;

/// RMSNorm: in-place root-mean-square layer normalization with scaling weight
#[inline(always)]
pub fn rms_norm(x: &mut [f32], weight: &[f32], eps: f32) {
    let mut ss = 0.0f32;
    for &v in x.iter() {
        ss += v * v;
    }
    let mean_sq = ss / (x.len() as f32);
    let scale = 1.0 / (mean_sq + eps).sqrt();

    for i in 0..x.len() {
        x[i] = x[i] * scale * weight[i];
    }
}

/// Rotary Positional Embeddings (RoPE)
#[inline(always)]
pub fn apply_rope(vec: &mut [f32], pos: usize, head_dim: usize, freq_base: f32) {
    let n_heads = vec.len() / head_dim;
    let half_dim = head_dim / 2;

    for h in 0..n_heads {
        let head_slice = &mut vec[h * head_dim..(h + 1) * head_dim];
        for i in 0..half_dim {
            let freq = 1.0 / freq_base.powf((2 * i) as f32 / head_dim as f32);
            let theta = (pos as f32) * freq;
            let (sin, cos) = theta.sin_cos();

            let x0 = head_slice[i];
            let x1 = head_slice[i + half_dim];

            head_slice[i] = x0 * cos - x1 * sin;
            head_slice[i + half_dim] = x0 * sin + x1 * cos;
        }
    }
}

/// In-place Softmax over float slice
#[inline(always)]
pub fn softmax(slice: &mut [f32]) {
    let mut max_v = f32::NEG_INFINITY;
    for &v in slice.iter() {
        if v > max_v {
            max_v = v;
        }
    }

    let mut sum = 0.0f32;
    for v in slice.iter_mut() {
        *v = (*v - max_v).exp();
        sum += *v;
    }

    let inv_sum = 1.0 / sum;
    for v in slice.iter_mut() {
        *v *= inv_sum;
    }
}

/// SwiGLU element-wise activation: gate * silu(up) -> gate * (up / (1 + exp(-up)))
#[inline(always)]
pub fn swiglu(gate: &mut [f32], up: &[f32]) {
    for i in 0..gate.len() {
        let u = up[i];
        let silu = u / (1.0 + (-u).exp());
        gate[i] *= silu;
    }
}

/// Quantized Parallel Matrix-Vector Multiplication: Y = W (Q4_0) * X (F32)
/// Automatically quantizes X to Q8_0 and uses multi-threaded Rayon dot-products.
pub fn matmul_q4_0(
    weight_bytes: &[u8],
    x_act: &[f32],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
) {
    // Quantize activation vector x to Q8_0 buffer
    let mut q8_act = vec![0u8; (in_dim / 32) * 34];
    quantize_q8_0(x_act, &mut q8_act);

    let row_bytes = (in_dim / 32) * 18;

    out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_start = r * row_bytes;
        let row_slice = &weight_bytes[row_start..row_start + row_bytes];
        *out_val = dot_q4_0_q8_0(row_slice, &q8_act, in_dim);
    });
}

/// Float32 Parallel Matrix-Vector Multiplication: Y = W (F32) * X (F32)
pub fn matmul_f32(weight: &[f32], x_act: &[f32], out: &mut [f32], in_dim: usize, _out_dim: usize) {
    out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_slice = &weight[r * in_dim..(r + 1) * in_dim];
        *out_val = dot_f32_f32(row_slice, x_act);
    });
}

/// Quantized Parallel Matrix-Vector Multiplication for Q4_K weights
pub fn matmul_q4_k(
    weight_bytes: &[u8],
    x_act: &[f32],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
) {
    let row_bytes = (in_dim / 256) * 144;

    out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_start = r * row_bytes;
        let row_slice = &weight_bytes[row_start..row_start + row_bytes];
        *out_val = dot_q4_k_f32(row_slice, x_act);
    });
}

/// Quantized Parallel Matrix-Vector Multiplication for Q6_K weights
pub fn matmul_q6_k(
    weight_bytes: &[u8],
    x_act: &[f32],
    out: &mut [f32],
    in_dim: usize,
    _out_dim: usize,
) {
    let row_bytes = (in_dim / 256) * 210;

    out.par_iter_mut().enumerate().for_each(|(r, out_val)| {
        let row_start = r * row_bytes;
        let row_slice = &weight_bytes[row_start..row_start + row_bytes];
        let mut row_f32 = vec![0.0f32; in_dim];
        crate::quant::dequantize_q6_k(row_slice, &mut row_f32);
        *out_val = dot_f32_f32(&row_f32, x_act);
    });
}
