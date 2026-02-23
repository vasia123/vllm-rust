//! OpenPangu-VL vision-language model.
//!
//! Architecture:
//! - Vision: Qwen2.5-VL-adapted ViT with window attention + multiple intermediate
//!   mergers (one per selected layer), outputs summed then projected.
//! - Language: PanguEmbeddedForCausalLM (dense transformer).
//!
//! # Key difference vs Qwen2.5-VL vision encoder
//!
//! Instead of a single merger applied to the final layer's output, OpenPangu-VL
//! captures `N` intermediate hidden states (at `mm_unit_vision_select_layer` indices),
//! applies a shared `final_layernorm` to each, passes each through a separate
//! `OpenPanguVisionPatchMerger`, then **sums** the results.  A final linear
//! projection (`vision_projection`) maps from `out_hidden_size → text_hidden_size`.
//!
//! # Weight paths
//!
//! After the vLLM `WeightsMapper` (`model.language_model.` → `language_model.model.`,
//! `model.visual.` → `visual.`, `lm_head.` → `language_model.lm_head.`,
//! `model.` → `language_model.model.`):
//!
//! ```text
//! visual.patch_embed.proj.*
//! visual.blocks.{i}.{norm1,norm2,attn.qkv,attn.proj,mlp.gate_up_proj,mlp.down_proj}.*
//! visual.final_layernorm.*
//! visual.merger.{i}.ln_q.*
//! visual.merger.{i}.mlp.{0,2}.*
//! visual.vision_projection.linear.*
//! language_model.{model.*,lm_head.*}
//! ```
//!
//! # Reference
//!
//! `reference/vllm/vllm/model_executor/models/openpangu_vl.py`

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{linear, linear_no_bias, Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::{rms_norm, RmsNorm, RotaryEmbedding};
use crate::multimodal::MultimodalInputs;

use super::pangu::PanguEmbeddedForCausalLM;

// ─── Config ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct OpVLVisionConfig {
    depth: usize,
    embed_dim: usize,
    num_heads: usize,
    intermediate_size: usize,
    in_channels: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    spatial_merge_size: usize,
    window_size: usize,
    fullatt_block_indexes: Vec<usize>,
    out_hidden_size: usize,
    /// Sorted absolute layer indices at which to capture intermediate states.
    take_indices: Vec<usize>,
    /// Number of mergers (= len(select_layer)).
    num_mergers: usize,
}

impl OpVLVisionConfig {
    fn head_dim(&self) -> usize {
        self.embed_dim / self.num_heads
    }

    fn patch_input_dim(&self) -> usize {
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    }

    fn merger_dim(&self) -> usize {
        self.embed_dim * self.spatial_merge_size * self.spatial_merge_size
    }

    fn spatial_merge_unit(&self) -> usize {
        self.spatial_merge_size * self.spatial_merge_size
    }

    fn from_json(json: &serde_json::Value, depth: usize) -> Self {
        let embed_dim = json
            .get("hidden_size")
            .or_else(|| json.get("embed_dim"))
            .and_then(|v| v.as_u64())
            .unwrap_or(1152) as usize;

        let fullatt_block_indexes = json
            .get("fullatt_block_indexes")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|x| x as usize))
                    .collect()
            })
            .unwrap_or_else(|| vec![7, 15, 23, 31]);

        // Replicate Python logic for take_indices:
        //   select_layer = mm_unit_vision_select_layer  (default [-1, -3])
        //   select_index = [depth + sl for sl in select_layer]
        //   select_index = select_index[::-1]           (reversed)
        //   take_indices = select_index
        let select_layer: Vec<i64> = json
            .get("mm_unit_vision_select_layer")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|v| v.as_i64()).collect())
            .unwrap_or_else(|| vec![-1, -3]);

        let mut select_index: Vec<usize> = select_layer
            .iter()
            .map(|&sl| (depth as i64 + sl) as usize)
            .collect();
        select_index.reverse();
        let num_mergers = select_index.len();
        let mut take_indices = select_index;
        take_indices.sort_unstable();

        Self {
            depth,
            embed_dim,
            num_heads: json.get("num_heads").and_then(|v| v.as_u64()).unwrap_or(16) as usize,
            intermediate_size: json
                .get("intermediate_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(4304) as usize,
            in_channels: json
                .get("in_channels")
                .and_then(|v| v.as_u64())
                .unwrap_or(3) as usize,
            patch_size: json
                .get("patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(14) as usize,
            temporal_patch_size: json
                .get("temporal_patch_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            spatial_merge_size: json
                .get("spatial_merge_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize,
            window_size: json
                .get("window_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(112) as usize,
            fullatt_block_indexes,
            out_hidden_size: json
                .get("out_hidden_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize,
            take_indices,
            num_mergers,
        }
    }
}

// ─── Window Attention Helpers ────────────────────────────────────────────────
//
// Copied from qwen2_5_vl.rs: computes a permutation that groups tokens into
// spatial windows for block-diagonal attention, and the inverse permutation.

/// Compute window indices for grouping LLM-level (merged) tokens into windows.
fn get_window_index(
    grid_t: usize,
    grid_h: usize,
    grid_w: usize,
    spatial_merge_size: usize,
    window_size: usize,
    patch_size: usize,
) -> (Vec<usize>, Vec<usize>) {
    let llm_grid_h = grid_h / spatial_merge_size;
    let llm_grid_w = grid_w / spatial_merge_size;
    let vit_merger_window_size = window_size / spatial_merge_size / patch_size;
    let total_llm_tokens = grid_t * llm_grid_h * llm_grid_w;

    if vit_merger_window_size == 0 || llm_grid_h == 0 || llm_grid_w == 0 {
        let seqlens = vec![total_llm_tokens];
        return ((0..total_llm_tokens).collect(), seqlens);
    }

    let pad_h = if llm_grid_h.is_multiple_of(vit_merger_window_size) {
        0
    } else {
        vit_merger_window_size - llm_grid_h % vit_merger_window_size
    };
    let pad_w = if llm_grid_w.is_multiple_of(vit_merger_window_size) {
        0
    } else {
        vit_merger_window_size - llm_grid_w % vit_merger_window_size
    };
    let num_windows_h = (llm_grid_h + pad_h) / vit_merger_window_size;
    let num_windows_w = (llm_grid_w + pad_w) / vit_merger_window_size;

    let mut window_index = Vec::with_capacity(total_llm_tokens);
    let mut window_seqlens = Vec::new();

    for t in 0..grid_t {
        // Build index for this temporal frame [llm_grid_h, llm_grid_w]
        let mut index = vec![vec![0i64; llm_grid_w]; llm_grid_h];
        for (h, row) in index.iter_mut().enumerate() {
            for (w, cell) in row.iter_mut().enumerate() {
                *cell = (t * llm_grid_h * llm_grid_w + h * llm_grid_w + w) as i64;
            }
        }

        // Pad to multiple of window size and reshape into windows
        let padded_h = num_windows_h * vit_merger_window_size;
        let padded_w = num_windows_w * vit_merger_window_size;
        let mut padded = vec![vec![-1i64; padded_w]; padded_h];
        for h in 0..llm_grid_h {
            for w in 0..llm_grid_w {
                padded[h][w] = index[h][w];
            }
        }

        // Extract windows in row-major order
        for wh in 0..num_windows_h {
            for ww in 0..num_windows_w {
                let mut valid_count = 0usize;
                for i in 0..vit_merger_window_size {
                    for j in 0..vit_merger_window_size {
                        let val = padded[wh * vit_merger_window_size + i]
                            [ww * vit_merger_window_size + j];
                        if val >= 0 {
                            window_index.push(val as usize);
                            valid_count += 1;
                        }
                    }
                }
                if valid_count > 0 {
                    window_seqlens.push(valid_count);
                }
            }
        }
    }

    (window_index, window_seqlens)
}

/// Build a block-diagonal attention mask from window boundaries.
fn build_window_mask(
    window_seqlens: &[usize],
    total_tokens: usize,
    smu: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let raw = total_tokens * smu;
    let mut mask = vec![f32::NEG_INFINITY; raw * raw];
    let mut start = 0usize;
    for &seqlen in window_seqlens {
        let scaled = seqlen * smu;
        for i in start..start + scaled {
            for j in start..start + scaled {
                mask[i * raw + j] = 0.0;
            }
        }
        start += scaled;
    }
    Tensor::from_vec(mask, (1, 1, raw, raw), device)?.to_dtype(dtype)
}

/// Invert a permutation: given perm[i] = j, returns inv[j] = i.
fn invert_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    inv
}

// ─── Vision Components ───────────────────────────────────────────────────────

struct OpVisionPatchEmbed {
    proj: Linear,
}

impl OpVisionPatchEmbed {
    fn new(cfg: &OpVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let proj = linear_no_bias(cfg.patch_input_dim(), cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self { proj })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.proj.forward(x)
    }
}

struct OpVisionMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
}

impl OpVisionMLP {
    fn new(embed_dim: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        let gate_up_proj = linear(embed_dim, 2 * intermediate_size, vb.pp("gate_up_proj"))?;
        let down_proj = linear(intermediate_size, embed_dim, vb.pp("down_proj"))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate_up = self.gate_up_proj.forward(x)?;
        let chunks = gate_up.chunk(2, candle_core::D::Minus1)?;
        let silu_gate = candle_nn::ops::silu(&chunks[0])?;
        let activated = silu_gate.mul(&chunks[1])?;
        self.down_proj.forward(&activated)
    }
}

struct OpVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl OpVisionAttention {
    fn new(cfg: &OpVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        let qkv = linear_no_bias(cfg.embed_dim, 3 * cfg.embed_dim, vb.pp("qkv"))?;
        let proj = linear_no_bias(cfg.embed_dim, cfg.embed_dim, vb.pp("proj"))?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim: cfg.head_dim(),
            scale: (cfg.head_dim() as f64).powf(-0.5),
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary_emb: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (num_tokens, _) = x.dims2()?;
        let qkv = self.qkv.forward(x)?;
        let q_size = self.num_heads * self.head_dim;
        let q = qkv.narrow(1, 0, q_size)?;
        let k = qkv.narrow(1, q_size, q_size)?;
        let v = qkv.narrow(1, 2 * q_size, q_size)?;

        let q = q
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let k = k
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;
        let v = v
            .reshape((num_tokens, self.num_heads, self.head_dim))?
            .transpose(0, 1)?
            .contiguous()?;

        let (q, k) = if let Some((cos, sin)) = rotary_emb {
            let rotary_dim = cos.dim(1)? * 2;
            let q_rot = q.narrow(2, 0, rotary_dim)?.contiguous()?;
            let q_pass = q.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;
            let k_rot = k.narrow(2, 0, rotary_dim)?.contiguous()?;
            let k_pass = k.narrow(2, rotary_dim, self.head_dim - rotary_dim)?;
            let q_rot = q_rot.unsqueeze(0)?;
            let k_rot = k_rot.unsqueeze(0)?;
            let q_rot = candle_nn::rotary_emb::rope(&q_rot.contiguous()?, cos, sin)?;
            let k_rot = candle_nn::rotary_emb::rope(&k_rot.contiguous()?, cos, sin)?;
            let q_rot = q_rot.squeeze(0)?;
            let k_rot = k_rot.squeeze(0)?;
            let q = Tensor::cat(&[q_rot, q_pass.contiguous()?], 2)?;
            let k = Tensor::cat(&[k_rot, k_pass.contiguous()?], 2)?;
            (q, k)
        } else {
            (q, k)
        };

        let q = q.unsqueeze(0)?;
        let k = k.unsqueeze(0)?;
        let v = v.unsqueeze(0)?;

        let attn = (q.matmul(&k.transpose(2, 3)?)? * self.scale)?;
        let attn = if let Some(mask) = attention_mask {
            attn.broadcast_add(mask)?
        } else {
            attn
        };
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v.contiguous()?)?;

        let out = out
            .squeeze(0)?
            .transpose(0, 1)?
            .contiguous()?
            .reshape((num_tokens, self.num_heads * self.head_dim))?;
        self.proj.forward(&out)
    }
}

struct OpVisionBlock {
    norm1: RmsNorm,
    attn: OpVisionAttention,
    norm2: RmsNorm,
    mlp: OpVisionMLP,
}

impl OpVisionBlock {
    fn new(cfg: &OpVLVisionConfig, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: rms_norm(cfg.embed_dim, 1e-6, vb.pp("norm1"))?,
            attn: OpVisionAttention::new(cfg, vb.pp("attn"))?,
            norm2: rms_norm(cfg.embed_dim, 1e-6, vb.pp("norm2"))?,
            mlp: OpVisionMLP::new(cfg.embed_dim, cfg.intermediate_size, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        x: &Tensor,
        rotary_emb: Option<(&Tensor, &Tensor)>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = x;
        let x = self
            .attn
            .forward(&self.norm1.forward(x)?, rotary_emb, attention_mask)?;
        let x = (residual + &x)?;
        let residual = &x;
        let x = self.mlp.forward(&self.norm2.forward(&x)?)?;
        residual + x
    }
}

// ─── Patch Merger ────────────────────────────────────────────────────────────

/// Spatial merger applied to each selected intermediate layer.
///
/// Unlike the Qwen2.5-VL merger which takes explicit grid dimensions, this
/// merger relies on the token sequence being pre-arranged (by the window
/// permutation) so that consecutive `spatial_merge_size²` tokens form a
/// spatial group.  It applies a flat `view(-1, merge_dim)` before the MLP.
struct OpPatchMerger {
    ln_q: RmsNorm,
    fc1: Linear,
    fc2: Linear,
    spatial_merge_unit: usize,
}

impl OpPatchMerger {
    fn new(cfg: &OpVLVisionConfig, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let merge_dim = cfg.merger_dim();
        let ln_q = rms_norm(cfg.embed_dim, 1e-6, vb.pp("ln_q"))?;
        let fc1 = linear(merge_dim, merge_dim, vb.pp("mlp").pp("0"))?;
        let fc2 = linear(merge_dim, out_dim, vb.pp("mlp").pp("2"))?;
        Ok(Self {
            ln_q,
            fc1,
            fc2,
            spatial_merge_unit: cfg.spatial_merge_unit(),
        })
    }

    /// `x`: [num_raw_tokens, embed_dim] — already window-permuted.
    /// Returns [num_raw_tokens / smu, out_dim].
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (num_raw, _embed_dim) = x.dims2()?;
        let smu = self.spatial_merge_unit;
        let merge_dim = smu * _embed_dim;
        let x = self.ln_q.forward(x)?;
        let x = x.reshape((num_raw / smu, merge_dim))?;
        let x = self.fc1.forward(&x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── Vision Transformer ──────────────────────────────────────────────────────

struct OpVisionTransformer {
    patch_embed: OpVisionPatchEmbed,
    rotary_emb: RotaryEmbedding,
    blocks: Vec<OpVisionBlock>,
    final_layernorm: RmsNorm,
    mergers: Vec<OpPatchMerger>,
    vision_projection: Linear,
    spatial_merge_size: usize,
    spatial_merge_unit: usize,
    window_size: usize,
    patch_size: usize,
    fullatt_block_indexes: Vec<usize>,
    take_indices: Vec<usize>,
}

impl OpVisionTransformer {
    fn new(cfg: &OpVLVisionConfig, text_hidden: usize, vb: VarBuilder) -> Result<Self> {
        let patch_embed = OpVisionPatchEmbed::new(cfg, vb.pp("patch_embed"))?;

        let rotary_emb = RotaryEmbedding::new_partial(
            cfg.head_dim(),
            8192,
            10000.0,
            0.5,
            true,
            DType::F32,
            vb.device(),
        )?;

        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(OpVisionBlock::new(cfg, vb.pp("blocks").pp(i))?);
        }

        let final_layernorm = rms_norm(cfg.embed_dim, 1e-6, vb.pp("final_layernorm"))?;

        let mut mergers = Vec::with_capacity(cfg.num_mergers);
        for i in 0..cfg.num_mergers {
            mergers.push(OpPatchMerger::new(
                cfg,
                cfg.out_hidden_size,
                vb.pp("merger").pp(i),
            )?);
        }

        let vision_projection = linear_no_bias(
            cfg.out_hidden_size,
            text_hidden,
            vb.pp("vision_projection").pp("linear"),
        )?;

        Ok(Self {
            patch_embed,
            rotary_emb,
            blocks,
            final_layernorm,
            mergers,
            vision_projection,
            spatial_merge_size: cfg.spatial_merge_size,
            spatial_merge_unit: cfg.spatial_merge_unit(),
            window_size: cfg.window_size,
            patch_size: cfg.patch_size,
            fullatt_block_indexes: cfg.fullatt_block_indexes.clone(),
            take_indices: cfg.take_indices.clone(),
        })
    }

    /// Encode patches for a single image/video.
    ///
    /// `patches`: [t*h*w, patch_input_dim] pre-extracted flat patches.
    /// `grid`: (t, h_raw, w_raw) — temporal and raw-patch-space grid dimensions.
    /// Returns [t*lh*lw, text_hidden] where lh = h_raw/merge, lw = w_raw/merge.
    fn forward(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        let (grid_t, grid_h, grid_w) = grid;
        let smu = self.spatial_merge_unit;

        // 1. Patch embedding
        let mut x = self.patch_embed.forward(patches)?; // [num_raw, D]
        let num_raw = x.dim(0)?;

        // 2. Compute 2D rotary embeddings [num_raw, rotary_dim]
        let rotary_emb = self.compute_2d_rotary(grid_h, grid_w, x.device())?;

        // 3. Window permutation (at merged-token granularity)
        let (window_index, window_seqlens) = get_window_index(
            grid_t,
            grid_h,
            grid_w,
            self.spatial_merge_size,
            self.window_size,
            self.patch_size,
        );

        // Expand window_index to raw-token level (each merged token = smu raw tokens)
        let mut raw_perm: Vec<u32> = Vec::with_capacity(num_raw);
        for &wi in &window_index {
            for s in 0..smu {
                raw_perm.push((wi * smu + s) as u32);
            }
        }
        let perm_t = Tensor::from_vec(raw_perm, (num_raw,), x.device())?;
        x = x.index_select(&perm_t, 0)?;

        // Also permute rotary embeddings
        let rotary_emb = rotary_emb.map(|(cos, sin)| {
            let cos = cos.index_select(&perm_t, 0).unwrap_or(cos);
            let sin = sin.index_select(&perm_t, 0).unwrap_or(sin);
            (cos, sin)
        });
        let rot_ref = rotary_emb
            .as_ref()
            .map(|(c, s)| (c as &Tensor, s as &Tensor));

        // 4. Build window attention mask
        let window_mask = build_window_mask(
            &window_seqlens,
            window_index.len(),
            smu,
            x.dtype(),
            x.device(),
        )?;

        // 5. Process through blocks, collecting intermediates at take_indices
        let mut intermediates: Vec<Tensor> = Vec::new();
        for (layer_idx, block) in self.blocks.iter().enumerate() {
            let mask = if self.fullatt_block_indexes.contains(&layer_idx) {
                None
            } else {
                Some(&window_mask)
            };
            x = block.forward(&x, rot_ref, mask)?;
            if self.take_indices.contains(&layer_idx) {
                intermediates.push(self.final_layernorm.forward(&x)?);
            }
        }

        // 6. Apply mergers (merger[0] → last intermediate, merger[1] → second-to-last, …)
        let n_inter = intermediates.len();
        let mut sum: Option<Tensor> = None;
        for (idx, merger) in self.mergers.iter().enumerate() {
            let intermed = &intermediates[n_inter - 1 - idx];
            let merged = merger.forward(intermed)?; // [num_merged, out_hidden]
            sum = Some(match sum {
                None => merged,
                Some(prev) => (prev + merged)?,
            });
        }
        let x = sum.ok_or_else(|| candle_core::Error::Msg("no mergers".to_string()))?;

        // 7. Reverse window permutation at merged-token level
        let reverse_index = invert_permutation(&window_index);
        let rev_t = Tensor::from_vec(
            reverse_index.iter().map(|&i| i as u32).collect(),
            (reverse_index.len(),),
            x.device(),
        )?;
        let x = x.index_select(&rev_t, 0)?;

        // 8. Final linear projection
        self.vision_projection.forward(&x)
    }

    fn compute_2d_rotary(
        &self,
        grid_h: usize,
        grid_w: usize,
        device: &Device,
    ) -> Result<Option<(Tensor, Tensor)>> {
        let half_rotary = self.rotary_emb.rotary_dim() / 2;
        let mut h_pos = Vec::with_capacity(grid_h * grid_w);
        let mut w_pos = Vec::with_capacity(grid_h * grid_w);
        for h in 0..grid_h {
            for w in 0..grid_w {
                h_pos.push(h as u32);
                w_pos.push(w as u32);
            }
        }
        let h_t = Tensor::from_vec(h_pos, (grid_h * grid_w,), device)?;
        let w_t = Tensor::from_vec(w_pos, (grid_h * grid_w,), device)?;

        let cos_h = self
            .rotary_emb
            .cos()
            .index_select(&h_t, 0)?
            .narrow(1, 0, half_rotary)?;
        let sin_h = self
            .rotary_emb
            .sin()
            .index_select(&h_t, 0)?
            .narrow(1, 0, half_rotary)?;
        let cos_w = self
            .rotary_emb
            .cos()
            .index_select(&w_t, 0)?
            .narrow(1, 0, half_rotary)?;
        let sin_w = self
            .rotary_emb
            .sin()
            .index_select(&w_t, 0)?
            .narrow(1, 0, half_rotary)?;

        let cos = Tensor::cat(&[cos_h, cos_w], 1)?;
        let sin = Tensor::cat(&[sin_h, sin_w], 1)?;
        Ok(Some((cos, sin)))
    }
}

// ─── merge_multimodal ────────────────────────────────────────────────────────

fn merge_multimodal(
    text_embeds: &Tensor,
    mm_inputs: &MultimodalInputs,
    device: &Device,
) -> Result<Tensor> {
    if !mm_inputs.has_images() {
        return Ok(text_embeds.clone());
    }
    let (_b, seq_len, _d) = text_embeds.dims3()?;
    let mut merged = text_embeds.to_vec3::<f32>()?;
    for (position, processed) in &mm_inputs.image_embeddings {
        let emb_vec: Vec<Vec<f32>> = processed.embedding.to_dtype(DType::F32)?.to_vec2()?;
        let batch_idx = position / seq_len;
        let start_pos = position % seq_len;
        for (i, emb) in emb_vec.iter().enumerate() {
            let target = start_pos + i;
            if target < seq_len && batch_idx < merged.len() {
                merged[batch_idx][target] = emb.clone();
            }
        }
    }
    Tensor::new(merged, device)?.to_dtype(text_embeds.dtype())
}

// ─── Main Model ──────────────────────────────────────────────────────────────

/// OpenPangu-VL: Qwen2.5-VL-adapted ViT with multi-merger + PanguEmbeddedForCausalLM.
pub struct OpenPanguVLForConditionalGeneration {
    visual: OpVisionTransformer,
    language_model: PanguEmbeddedForCausalLM,
    device: Device,
}

impl OpenPanguVLForConditionalGeneration {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        let vis_json = cfg
            .extra
            .get("vision_config")
            .cloned()
            .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
        let depth = vis_json.get("depth").and_then(|v| v.as_u64()).unwrap_or(28) as usize;
        let vis_cfg = OpVLVisionConfig::from_json(&vis_json, depth);

        let visual = OpVisionTransformer::new(&vis_cfg, cfg.hidden_size, vb.pp("visual"))?;
        let language_model = PanguEmbeddedForCausalLM::new(cfg, vb.pp("language_model"))?;

        Ok(Self {
            visual,
            language_model,
            device: vb.device().clone(),
        })
    }

    /// Encode pixel patches: `patches` [T*H*W, patch_input_dim] → [lh*lw, text_hidden].
    pub fn encode_images(&self, patches: &Tensor, grid: (usize, usize, usize)) -> Result<Tensor> {
        self.visual.forward(patches, grid)
    }
}

impl crate::engine::ModelForward for OpenPanguVLForConditionalGeneration {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn forward_decode_batch(
        &self,
        input_ids: &Tensor,
        sequences: &[DecodeSequenceMetadata],
        kv_cache_mgr: &mut KVCacheManager,
    ) -> Result<Tensor> {
        let embeddings = self.language_model.embed_text(input_ids)?;
        self.language_model.forward_decode_batch_with_embeddings(
            &embeddings,
            sequences,
            kv_cache_mgr,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }

    fn supports_multimodal(&self) -> bool {
        true
    }

    fn forward_multimodal(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let text_embeddings = self.language_model.embed_text(input_ids)?;
        let embeddings = if let Some(mm) = multimodal_inputs {
            merge_multimodal(&text_embeddings, mm, &self.device)?
        } else {
            text_embeddings
        };
        self.language_model.forward_with_embeddings(
            &embeddings,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device, Tensor};
    use candle_nn::VarBuilder;
    use serde_json::json;

    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, BlockTable, KVCacheDtype, KVCacheManager};
    use crate::multimodal::{MultimodalInputs, ProcessedImage};

    // Test config: tiny vision (4×4 raw grid, 1ch, patch=2, temporal=2)
    // + tiny text (hidden=32, 2 layers).
    fn test_cfg() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "vision_config".to_string(),
            json!({
                "depth": 2,
                "hidden_size": 16,
                "num_heads": 2,
                "intermediate_size": 8,
                "in_channels": 1,
                "patch_size": 2,
                "temporal_patch_size": 2,
                "spatial_merge_size": 2,
                "window_size": 4,
                "fullatt_block_indexes": [1],
                "out_hidden_size": 16,
                "mm_unit_vision_select_layer": [-1, -2]
            }),
        );
        ModelConfig {
            architectures: vec!["OpenPanguVLForConditionalGeneration".to_string()],
            hidden_size: 32,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 16,
            vocab_size: 64,
            max_position_embeddings: 512,
            head_dim: 16,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            bos_token_id: 1,
            eos_token_id: 2,
            sliding_window: None,
            attention_bias: Some(false),
            extra,
        }
    }

    fn make_cache(cfg: &ModelConfig, device: &Device) -> KVCacheManager {
        KVCacheManager::new(&CacheConfig {
            block_size: 4,
            num_blocks: 32,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        })
        .unwrap()
    }

    #[test]
    fn test_opangu_vl_new() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = OpenPanguVLForConditionalGeneration::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_opangu_vl_vision_encode() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = OpenPanguVLForConditionalGeneration::new(&cfg, vb).unwrap();

        // grid=(1, 4, 4): t=1, h_raw=4, w_raw=4 → 16 raw patches
        // patch_input_dim = 1*2*2*2 = 8
        // spatial_merge_size=2 → smu=4 → 16/4 = 4 merged tokens
        // vision_projection: [4, 16] → [4, 32]
        let patches = Tensor::zeros((16usize, 8), DType::F32, &device).unwrap();
        let result = model.encode_images(&patches, (1, 4, 4));
        assert!(result.is_ok(), "vision encode failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[4, 32]);
    }

    #[test]
    fn test_opangu_vl_text_only() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = OpenPanguVLForConditionalGeneration::new(&cfg, vb).unwrap();

        let seq_len = 4usize;
        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::zeros((1usize, seq_len), DType::U32, &device).unwrap();
        let result = model.forward(&input_ids, 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "text-only forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, seq_len, 64]);
    }

    #[test]
    fn test_opangu_vl_with_image() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = OpenPanguVLForConditionalGeneration::new(&cfg, vb).unwrap();

        // Encode 1 image: 16 raw patches → 4 merged tokens → vision_proj → [4, 32]
        let patches = Tensor::zeros((16usize, 8), DType::F32, &device).unwrap();
        let img_feats = model
            .encode_images(&patches, (1, 4, 4))
            .expect("encode failed");
        let processed = ProcessedImage::new(img_feats, 4);

        // Sequence: 6 tokens, image at positions 1..4 (4 tokens), surrounded by text
        let input_ids = Tensor::from_slice(&[0u32, 9, 9, 9, 9, 0], (1usize, 6), &device).unwrap();
        let mm = MultimodalInputs::with_images(vec![0u32, 9, 9, 9, 9, 0], vec![(1, processed)]);

        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 6).unwrap();
        let slot_mapping = bt.slot_mapping(0, 6);

        let result =
            model.forward_multimodal(&input_ids, Some(&mm), 0, &mut kv, &bt, &slot_mapping);
        assert!(
            result.is_ok(),
            "multimodal forward failed: {:?}",
            result.err()
        );
        assert_eq!(result.unwrap().dims(), &[1, 6, 64]);
    }

    #[test]
    fn test_opangu_vl_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_cfg();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = OpenPanguVLForConditionalGeneration::new(&cfg, vb).unwrap();

        let mut kv = make_cache(&cfg, &device);
        let mut bt = BlockTable::new(4);
        kv.allocate_for_request(&mut bt, 4).unwrap();
        let slot = bt.slot_mapping(3, 1);
        let seq = crate::engine::DecodeSequenceMetadata {
            request_id: 1,
            seqlen_offset: 3,
            slot_mapping: slot.clone(),
            block_ids: bt.block_ids().to_vec(),
        };

        let input_ids = Tensor::from_slice(&[5u32], (1usize, 1), &device).unwrap();
        let result = model.forward_decode_batch(&input_ids, &[seq], &mut kv);
        assert!(result.is_ok(), "decode_batch failed: {:?}", result.err());
        assert_eq!(result.unwrap().dims(), &[1, 1, 64]);
    }
}
