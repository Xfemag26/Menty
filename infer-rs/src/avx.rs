use half::f16;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

#[inline(always)]
#[cfg(target_arch = "x86_64")]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(vlow, vhigh);
    let sum64 = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let sum32 = _mm_add_ss(sum64, _mm_shuffle_ps(sum64, sum64, 1));
    _mm_cvtss_f32(sum32)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn dot_q4_k_f32_avx(q4_row: &[u8], x: &[f32]) -> f32 {
    let n = x.len();
    let num_super_blocks = n / 256;
    if num_super_blocks == 0 || q4_row.len() < num_super_blocks * 144 {
        return 0.0;
    }

    let low_mask = _mm_set1_epi8(0x0F);
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    let q_ptr = q4_row.as_ptr();
    let x_ptr = x.as_ptr();

    for sb in 0..num_super_blocks {
        let sb_offset = sb * 144;
        let x_sb_offset = sb * 256;

        let d_bytes = [*q_ptr.add(sb_offset), *q_ptr.add(sb_offset + 1)];
        let dmin_bytes = [*q_ptr.add(sb_offset + 2), *q_ptr.add(sb_offset + 3)];

        let d = f16::from_le_bytes(d_bytes).to_f32();
        let dmin = f16::from_le_bytes(dmin_bytes).to_f32();

        let scales_ptr = q_ptr.add(sb_offset + 4);
        let mut scales = [0.0f32; 8];
        let mut mins = [0.0f32; 8];

        for j in 0..4 {
            let sc_byte = *scales_ptr.add(j);
            let m_byte = *scales_ptr.add(j + 4);
            let sc2_m2_byte = *scales_ptr.add(j + 8);

            let sc = sc_byte & 0x3F;
            let m = m_byte & 0x3F;
            scales[j] = d * (sc as f32);
            mins[j] = dmin * (m as f32);

            // ggml get_scale_min_k4: high bits of later scales sit in the min bytes, and vice versa
            let sc2 = (sc2_m2_byte & 0x0F) | ((m_byte >> 6) << 4);
            let m2 = (sc2_m2_byte >> 4) | ((sc_byte >> 6) << 4);
            scales[j + 4] = d * (sc2 as f32);
            mins[j + 4] = dmin * (m2 as f32);
        }

        let quants_ptr = q_ptr.add(sb_offset + 16);

        // 4 chunks × 64 values per super block
        for chunk in 0..4 {
            let sc0_vec = _mm256_set1_ps(scales[chunk * 2]);
            let min0_vec = _mm256_set1_ps(mins[chunk * 2]);
            let sc1_vec = _mm256_set1_ps(scales[chunk * 2 + 1]);
            let min1_vec = _mm256_set1_ps(mins[chunk * 2 + 1]);

            let q_chunk_ptr = quants_ptr.add(chunk * 32);
            let x0_chunk_ptr = x_ptr.add(x_sb_offset + chunk * 64);
            let x1_chunk_ptr = x_ptr.add(x_sb_offset + chunk * 64 + 32);

            // 32 bytes = 4 iterations of 8 bytes
            for i in 0..4 {
                let off = i * 8;
                let raw_u64 = (q_chunk_ptr.add(off) as *const u64).read_unaligned();
                let bytes_vec = _mm_cvtsi64_si128(raw_u64 as i64);

                // Low nibbles -> x0
                let q_low = _mm_and_si128(bytes_vec, low_mask);
                let q0_i32 = _mm256_cvtepu8_epi32(q_low);
                let q0_ps = _mm256_cvtepi32_ps(q0_i32);
                let x0_vec = _mm256_loadu_ps(x0_chunk_ptr.add(off));
                let w0 = _mm256_fmsub_ps(q0_ps, sc0_vec, min0_vec);
                acc0 = _mm256_fmadd_ps(w0, x0_vec, acc0);

                // High nibbles -> x1
                let q_high = _mm_and_si128(_mm_srli_epi16(bytes_vec, 4), low_mask);
                let q1_i32 = _mm256_cvtepu8_epi32(q_high);
                let q1_ps = _mm256_cvtepi32_ps(q1_i32);
                let x1_vec = _mm256_loadu_ps(x1_chunk_ptr.add(off));
                let w1 = _mm256_fmsub_ps(q1_ps, sc1_vec, min1_vec);
                acc1 = _mm256_fmadd_ps(w1, x1_vec, acc1);
            }
        }
    }

    hsum256_ps(_mm256_add_ps(acc0, acc1))
}

#[cfg(not(target_arch = "x86_64"))]
pub unsafe fn dot_q4_k_f32_avx(q4_row: &[u8], x: &[f32]) -> f32 {
    crate::quant::dot_q4_k_f32_scalar(q4_row, x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dot_q4_k_f32_avx_equivalence() {
        let mut q4_row = vec![0u8; 144 * 2];
        let mut x = vec![0.0f32; 512];

        for i in 0..512 {
            x[i] = ((i as f32 * 0.1337).sin()) * 2.0;
        }

        for sb in 0..2 {
            let sb_offset = sb * 144;
            let d_bytes = f16::from_f32(0.25).to_le_bytes();
            q4_row[sb_offset] = d_bytes[0];
            q4_row[sb_offset + 1] = d_bytes[1];
            let dmin_bytes = f16::from_f32(0.05).to_le_bytes();
            q4_row[sb_offset + 2] = dmin_bytes[0];
            q4_row[sb_offset + 3] = dmin_bytes[1];

            for j in 0..12 {
                q4_row[sb_offset + 4 + j] = ((j * 17 + sb * 31) % 64) as u8;
            }

            for j in 0..128 {
                q4_row[sb_offset + 16 + j] = ((j * 37 + sb * 53) % 256) as u8;
            }
        }

        let scalar_result = crate::quant::dot_q4_k_f32_scalar(&q4_row, &x);
        let top_level_result = crate::quant::dot_q4_k_f32(&q4_row, &x);

        #[cfg(target_arch = "x86_64")]
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            let avx_result = unsafe { dot_q4_k_f32_avx(&q4_row, &x) };
            let diff = (scalar_result - avx_result).abs();
            let rel_diff = diff / scalar_result.abs().max(1.0);
            assert!(
                rel_diff < 1e-4,
                "Scalar ({scalar_result}) and AVX ({avx_result}) differ (rel_diff={rel_diff})"
            );
        }

        let diff2 = (scalar_result - top_level_result).abs();
        let rel_diff2 = diff2 / scalar_result.abs().max(1.0);
        assert!(rel_diff2 < 1e-4, "Scalar and top-level differ");
    }
}
