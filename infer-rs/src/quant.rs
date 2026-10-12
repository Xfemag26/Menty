// quant.rs - Fast vector quantization & dot-products for Q4_0, Q8_0, Q4_K, F16, F32
use half::f16;

#[inline(always)]
pub fn dequantize_q4_0(src: &[u8], dst: &mut [f32]) {
    let num_blocks = dst.len() / 32;
    for i in 0..num_blocks {
        let block_bytes = &src[i * 18..(i + 1) * 18];
        let scale = f16::from_le_bytes([block_bytes[0], block_bytes[1]]).to_f32();
        let quants = &block_bytes[2..18];
        let out_slice = &mut dst[i * 32..(i + 1) * 32];

        for j in 0..16 {
            let byte = quants[j];
            let v0 = ((byte & 0x0F) as i8 - 8) as f32 * scale;
            let v1 = (((byte >> 4) & 0x0F) as i8 - 8) as f32 * scale;
            out_slice[j] = v0;
            out_slice[j + 16] = v1;
        }
    }
}

#[inline(always)]
pub fn quantize_q8_0(src: &[f32], dst: &mut [u8]) {
    let num_blocks = src.len() / 32;
    for i in 0..num_blocks {
        let in_slice = &src[i * 32..(i + 1) * 32];
        let mut amax = 0.0f32;
        for &v in in_slice {
            let abs_v = v.abs();
            if abs_v > amax {
                amax = abs_v;
            }
        }
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f16::from_f32(d).to_le_bytes();

        let block_out = &mut dst[i * 34..(i + 1) * 34];
        block_out[0] = d_f16[0];
        block_out[1] = d_f16[1];

        for j in 0..32 {
            let val = (in_slice[j] * id).round().clamp(-128.0, 127.0) as i8;
            block_out[2 + j] = val as u8;
        }
    }
}

#[inline(always)]
pub fn quantize_q4_0(src: &[f32], dst: &mut [u8]) {
    let num_blocks = src.len() / 32;
    for i in 0..num_blocks {
        let in_slice = &src[i * 32..(i + 1) * 32];
        let mut amax = 0.0f32;
        for &v in in_slice {
            let abs_v = v.abs();
            if abs_v > amax {
                amax = abs_v;
            }
        }
        let d = amax / -8.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        let d_f16 = f16::from_f32(-d).to_le_bytes();

        let block_out = &mut dst[i * 18..(i + 1) * 18];
        block_out[0] = d_f16[0];
        block_out[1] = d_f16[1];

        for j in 0..16 {
            let v0 = (in_slice[j] * id + 8.5).clamp(0.0, 15.0) as u8;
            let v1 = (in_slice[j + 16] * id + 8.5).clamp(0.0, 15.0) as u8;
            block_out[2 + j] = (v0 & 0x0F) | ((v1 & 0x0F) << 4);
        }
    }
}

/// Fast dot product between Q4_0 quantized row and Q8_0 quantized activation vector
#[inline(always)]
pub fn dot_q4_0_q8_0(q4_row: &[u8], q8_act: &[u8], n_elements: usize) -> f32 {
    let num_blocks = n_elements / 32;
    let mut sum = 0.0f32;

    for i in 0..num_blocks {
        let q4_b = &q4_row[i * 18..(i + 1) * 18];
        let q8_b = &q8_act[i * 34..(i + 1) * 34];

        let d4 = f16::from_le_bytes([q4_b[0], q4_b[1]]).to_f32();
        let d8 = f16::from_le_bytes([q8_b[0], q8_b[1]]).to_f32();

        let mut block_sum = 0i32;
        for j in 0..16 {
            let byte4 = q4_b[2 + j];
            let q4_v0 = (byte4 & 0x0F) as i32 - 8;
            let q4_v1 = ((byte4 >> 4) & 0x0F) as i32 - 8;

            let q8_v0 = q8_b[2 + j] as i8 as i32;
            let q8_v1 = q8_b[2 + j + 16] as i8 as i32;

            block_sum += q4_v0 * q8_v0 + q4_v1 * q8_v1;
        }

        sum += (block_sum as f32) * (d4 * d8);
    }

    sum
}

/// Fast dot product between F32 row and F32 vector
#[inline(always)]
pub fn dot_f32_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0f32;
    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let rem_a = chunks_a.remainder();
    let rem_b = chunks_b.remainder();

    for (ca, cb) in chunks_a.zip(chunks_b) {
        sum += ca[0] * cb[0]
            + ca[1] * cb[1]
            + ca[2] * cb[2]
            + ca[3] * cb[3]
            + ca[4] * cb[4]
            + ca[5] * cb[5]
            + ca[6] * cb[6]
            + ca[7] * cb[7];
    }
    for (va, vb) in rem_a.iter().zip(rem_b.iter()) {
        sum += va * vb;
    }
    sum
}

/// Dequantizes Q4_K super-block (256 elements per 144 bytes)
#[inline(always)]
pub fn dequantize_q4_k(src: &[u8], dst: &mut [f32]) {
    let num_super_blocks = dst.len() / 256;
    for sb in 0..num_super_blocks {
        let sb_bytes = &src[sb * 144..(sb + 1) * 144];
        let d = f16::from_le_bytes([sb_bytes[0], sb_bytes[1]]).to_f32();
        let dmin = f16::from_le_bytes([sb_bytes[2], sb_bytes[3]]).to_f32();

        // 12 bytes scale & min
        let scales_buf = &sb_bytes[4..16];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];

        // ggml get_scale_min_k4: scales[0..4] = 6-bit scales, scales[4..8] = 6-bit mins.
        // High 2 bits of sub-blocks 4..7 live in the top bits of those bytes; low 4 bits in scales[8..12].
        for j in 0..4 {
            let sc = scales_buf[j] & 0x3F;
            let m = scales_buf[j + 4] & 0x3F;
            scales[j] = d * (sc as f32);
            mins[j] = dmin * (m as f32);

            let sc2 = (scales_buf[j + 8] & 0x0F) | ((scales_buf[j + 4] >> 6) << 4);
            let m2 = (scales_buf[j + 8] >> 4) | ((scales_buf[j] >> 6) << 4);
            scales[j + 4] = d * (sc2 as f32);
            mins[j + 4] = dmin * (m2 as f32);
        }

        let quants = &sb_bytes[16..144];
        let out_sb = &mut dst[sb * 256..(sb + 1) * 256];

        for chunk in 0..4 {
            let q_chunk = &quants[chunk * 32..(chunk + 1) * 32];
            let sc0 = scales[chunk * 2];
            let min0 = mins[chunk * 2];
            let sc1 = scales[chunk * 2 + 1];
            let min1 = mins[chunk * 2 + 1];

            for j in 0..32 {
                let byte = q_chunk[j];
                let v0 = (byte & 0x0F) as f32;
                let v1 = ((byte >> 4) & 0x0F) as f32;

                out_sb[chunk * 64 + j] = v0 * sc0 - min0;
                out_sb[chunk * 64 + 32 + j] = v1 * sc1 - min1;
            }
        }
    }
}

/// Dequantizes Q6_K super-block (256 elements per 210 bytes)
#[inline(always)]
pub fn dequantize_q6_k(src: &[u8], dst: &mut [f32]) {
    let num_super_blocks = dst.len() / 256;
    for sb in 0..num_super_blocks {
        let sb_bytes = &src[sb * 210..(sb + 1) * 210];
        let ql = &sb_bytes[0..128];
        let qh = &sb_bytes[128..192];
        let sc = &sb_bytes[192..208];
        let d = f16::from_le_bytes([sb_bytes[208], sb_bytes[209]]).to_f32();

        let out_sb = &mut dst[sb * 256..(sb + 1) * 256];

        for i in 0..16 {
            let scale = d * (sc[i] as i8 as f32);
            for j in 0..16 {
                let idx = i * 16 + j;
                let l_byte = ql[idx / 2];
                let l_val = if idx % 2 == 0 {
                    l_byte & 0x0F
                } else {
                    l_byte >> 4
                };

                let h_byte = qh[idx / 4];
                let shift = (idx % 4) * 2;
                let h_val = (h_byte >> shift) & 0x03;

                let q_val = ((h_val << 4) | l_val) as i8 - 32;
                out_sb[idx] = (q_val as f32) * scale;
            }
        }
    }
}

/// Dot Q4_K row with f32 activation. Same weights as `dequantize_q4_k`
/// then `dot_f32_f32`, but the float row is never stored.
/// Q4_K super-block = 256 values / 144 bytes. `n` must be a multiple of 256.
#[inline(always)]
pub fn dot_q4_k_f32_scalar(q4_row: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let num_super_blocks = n / 256;
    let mut sum = 0.0f32;

    for sb in 0..num_super_blocks {
        let sb_bytes = &q4_row[sb * 144..(sb + 1) * 144];
        let d = f16::from_le_bytes([sb_bytes[0], sb_bytes[1]]).to_f32();
        let dmin = f16::from_le_bytes([sb_bytes[2], sb_bytes[3]]).to_f32();

        let scales_buf = &sb_bytes[4..16];
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];

        for j in 0..4 {
            let sc = scales_buf[j] & 0x3F;
            let m = scales_buf[j + 4] & 0x3F;
            scales[j] = d * (sc as f32);
            mins[j] = dmin * (m as f32);

            let sc2 = (scales_buf[j + 8] & 0x0F) | ((scales_buf[j + 4] >> 6) << 4);
            let m2 = (scales_buf[j + 8] >> 4) | ((scales_buf[j] >> 6) << 4);
            scales[j + 4] = d * (sc2 as f32);
            mins[j + 4] = dmin * (m2 as f32);
        }

        let quants = &sb_bytes[16..144];
        let x_sb = &x[sb * 256..(sb + 1) * 256];

        // Layout matches dequantize_q4_k: 4 chunks × 64 values.
        // Low nibbles use scales[chunk*2], high nibbles use scales[chunk*2+1].
        for chunk in 0..4 {
            let q_chunk = &quants[chunk * 32..(chunk + 1) * 32];
            let sc0 = scales[chunk * 2];
            let min0 = mins[chunk * 2];
            let sc1 = scales[chunk * 2 + 1];
            let min1 = mins[chunk * 2 + 1];
            let x0 = &x_sb[chunk * 64..chunk * 64 + 32];
            let x1 = &x_sb[chunk * 64 + 32..chunk * 64 + 64];

            for j in 0..32 {
                let byte = q_chunk[j];
                let v0 = (byte & 0x0F) as f32 * sc0 - min0;
                let v1 = (byte >> 4) as f32 * sc1 - min1;
                // SAFETY: j is within 0..32, x0 and x1 are of length 32
                unsafe {
                    sum += v0 * *x0.get_unchecked(j) + v1 * *x1.get_unchecked(j);
                }
            }
        }
    }

    sum
}

#[inline(always)]
pub fn dot_q4_k_f32(q4_row: &[u8], x: &[f32]) -> f32 {
    if is_x86_feature_detected!("avx2") {
        unsafe { crate::avx::dot_q4_k_f32_avx(q4_row, x) }
    } else {
        dot_q4_k_f32_scalar(q4_row, x)
    }
}
