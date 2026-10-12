// partition.rs - HELM Dynamic Layer Partitioning for Heterogeneous Memory (GDDR5 + DDR4)

#[derive(Debug, Clone)]
pub struct PartitionPlan {
    pub total_layers: usize,
    pub gpu_layers: usize,
    pub cpu_layers: usize,
    pub gpu_memory_mb: usize,
    pub cpu_memory_mb: usize,
    pub kv_cache_mb: usize,
}

pub struct HelmPartitioner {
    pub vram_total_mb: usize,
    pub vram_reserved_mb: usize,
    pub ctx_size: usize,
}

impl HelmPartitioner {
    pub fn new_mx150(ctx_size: usize) -> Self {
        // MX150 has 2048MB physical VRAM:
        // ~200MB X11/display server
        // ~150MB CUDA driver context + scratchpad
        // Reserved = 350MB + KV cache allowance
        HelmPartitioner {
            vram_total_mb: 2048,
            vram_reserved_mb: 350,
            ctx_size,
        }
    }

    pub fn plan(
        &self,
        total_layers: usize,
        total_weight_bytes: usize,
        n_kv_heads: usize,
        head_dim: usize,
    ) -> PartitionPlan {
        let total_weight_mb = total_weight_bytes / (1024 * 1024);
        let per_layer_weight_mb = if total_layers > 0 {
            total_weight_mb / total_layers
        } else {
            1
        };

        // Calculate asymmetric KV cache requirement per layer:
        // q8_0 K: (head_dim / 32) * 34 bytes
        // q4_0 V: (head_dim / 32) * 18 bytes
        let kv_token_bytes = n_kv_heads * (head_dim / 32) * (34 + 18);
        let kv_total_mb = (self.ctx_size * total_layers * kv_token_bytes) / (1024 * 1024);

        let usable_vram_mb = self
            .vram_total_mb
            .saturating_sub(self.vram_reserved_mb + kv_total_mb);

        // Compute optimal GPU layers within usable envelope
        let possible_gpu_layers = if per_layer_weight_mb > 0 {
            usable_vram_mb / per_layer_weight_mb
        } else {
            0
        };

        let gpu_layers = possible_gpu_layers.min(total_layers);
        let cpu_layers = total_layers.saturating_sub(gpu_layers);

        let gpu_memory_mb = gpu_layers * per_layer_weight_mb;
        let cpu_memory_mb = cpu_layers * per_layer_weight_mb;

        PartitionPlan {
            total_layers,
            gpu_layers,
            cpu_layers,
            gpu_memory_mb,
            cpu_memory_mb,
            kv_cache_mb: kv_total_mb,
        }
    }
}
