//! Phi-4 Multimodal (Phi4MMForCausalLM) vision-language model implementation.
//!
//! Architecture:
//! - Vision encoder: SigLIP (SigLIP-so400m-patch14-448)
//! - HD transform: Multi-crop image processing with learnable separators
//! - Projector: 2-layer MLP with AvgPool2d compression
//! - Language model: LLaMA-compatible decoder layers
//!
//! Audio support: `AudioEmbedding` (`embed_tokens_extend`) is defined in
//! `phi4mm_audio.rs` and wired in here. The conformer encoder runs in Python
//! preprocessing; at inference time we scatter pre-encoded audio embeddings.
//!
//! Reference: reference/vllm/vllm/model_executor/models/phi4mm.py

use crate::layers::{rms_norm, RmsNorm};
use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Linear, VarBuilder};

use crate::config::ModelConfig;
use crate::distributed::{LocalProcessGroup, ProcessGroup};
use crate::engine::DecodeSequenceMetadata;
use crate::kv_cache::{BlockTable, KVCacheManager};
use crate::layers::causal_mask;
use crate::multimodal::{MultimodalInputs, VisionEncoder, VisionEncoderConfig, VisionEncoderType};

use super::llama::{LlamaDecoderLayer, TpContext};
use super::phi4mm_audio::Phi4MMAudioEmbedding;
use super::tp_layers::{TpEmbedding, TpLinear};

// ─── Config ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
#[allow(dead_code)]
struct Phi4MMConfig {
    model_config: ModelConfig,
    vision_config: VisionEncoderConfig,
    image_dim_out: usize,
    image_token_id: u32,
    use_hd_transform: bool,
    hd_transform_order: String,
}

impl Phi4MMConfig {
    fn from_model_config(cfg: &ModelConfig) -> Self {
        let defaults = VisionEncoderConfig {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 448,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        };

        let vision_config = if let Some(vc) = cfg.extra.get("img_processor") {
            VisionEncoderConfig {
                encoder_type: VisionEncoderType::SigLip,
                hidden_size: vc
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.hidden_size as u64) as usize,
                intermediate_size: vc
                    .get("intermediate_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.intermediate_size as u64)
                    as usize,
                num_attention_heads: vc
                    .get("num_attention_heads")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_attention_heads as u64)
                    as usize,
                num_hidden_layers: vc
                    .get("num_hidden_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.num_hidden_layers as u64)
                    as usize,
                image_size: vc
                    .get("image_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.image_size as u64) as usize,
                patch_size: vc
                    .get("patch_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(defaults.patch_size as u64) as usize,
                num_channels: defaults.num_channels,
                layer_norm_eps: defaults.layer_norm_eps,
            }
        } else {
            defaults
        };

        let image_dim_out = vision_config.hidden_size;

        let image_token_id = cfg
            .extra
            .get("image_token_id")
            .and_then(|v| v.as_u64())
            .unwrap_or(200010) as u32;

        let use_hd_transform = cfg
            .extra
            .get("use_hd_transform")
            .and_then(|v| v.as_bool())
            .unwrap_or(true);

        let hd_transform_order = cfg
            .extra
            .get("hd_transform_order")
            .and_then(|v| v.as_str())
            .unwrap_or("sub_glb")
            .to_string();

        Self {
            model_config: cfg.clone(),
            vision_config,
            image_dim_out,
            image_token_id,
            use_hd_transform,
            hd_transform_order,
        }
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

/// 2×2 average pooling for a batched NHWC tensor.
///
/// Reduces spatial H and W by a factor of 2, equivalent to PyTorch's
/// `nn.AvgPool2d(kernel_size=2, stride=2)` applied to an NCHW tensor that
/// has been converted to NHWC before and after.
fn avg_pool2x2_nhwc(x: &Tensor) -> Result<Tensor> {
    let (n, h, w, c) = x.dims4()?;
    // Reshape each 2×2 spatial block so we can average over it:
    //   [N, H, W, C] → [N, H/2, 2, W/2, 2, C]
    // Element (n, a, b, c_, d, e) maps to original (n, 2a+b, 2c_+d, e). ✓
    x.contiguous()?
        .reshape((n, h / 2, 2, w / 2, 2, c))?
        .mean(2)? // [N, H/2, W/2, 2, C]
        .mean(3) // [N, H/2, W/2, C]
}

// ─── Image Projector ────────────────────────────────────────────────────────

/// Phi4MM image projector: 2-layer MLP.
///
/// Projects from compressed image features to LLM hidden size.
/// In the reference, input is first compressed via 2x2 AvgPool (4x channels).
struct Phi4MMProjector {
    fc1: Linear,
    fc2: Linear,
}

impl Phi4MMProjector {
    fn new(input_dim: usize, output_dim: usize, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(input_dim, output_dim, vb.pp("0"))?;
        let fc2 = candle_nn::linear(output_dim, output_dim, vb.pp("2"))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu_erf()?;
        self.fc2.forward(&x)
    }
}

// ─── Full Model ─────────────────────────────────────────────────────────────

/// Phi-4 Multimodal model for conditional generation.
///
/// SigLIP vision encoder + HD transform + MLP projector + LLaMA backbone.
/// Audio: `Phi4MMAudioEmbedding` (conformer stub + projection layers) at
/// `embed_tokens_extend.*`; pre-encoded audio embeddings scattered at
/// `_AUDIO_PLACEHOLDER_TOKEN_ID = 200011` positions.
pub struct Phi4MMForCausalLM {
    #[allow(dead_code)]
    vision_encoder: VisionEncoder,
    projector: Phi4MMProjector,
    audio_embed: Option<Phi4MMAudioEmbedding>,
    /// Global separator token inserted between sub-images and the global crop.
    /// Shape: [1, image_dim_out].
    glb_gn: Tensor,
    /// Row separator appended at the end of each spatial row in a crop.
    /// Shape: [1, image_dim_out].
    sub_gn: Tensor,
    // LLM
    embed_tokens: TpEmbedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RmsNorm,
    lm_head: TpLinear,
    tp_ctx: TpContext,
    #[allow(dead_code)]
    config: Phi4MMConfig,
    device: Device,
    dtype: DType,
}

impl Phi4MMForCausalLM {
    pub fn new(cfg: &ModelConfig, vb: VarBuilder) -> Result<Self> {
        Self::new_with_tp(cfg, vb, &LocalProcessGroup::new(), TpContext::single_gpu())
    }

    pub fn new_with_tp(
        cfg: &ModelConfig,
        vb: VarBuilder,
        pg: &dyn ProcessGroup,
        tp_ctx: TpContext,
    ) -> Result<Self> {
        let config = Phi4MMConfig::from_model_config(cfg);
        let world_size = pg.world_size();

        // Vision encoder (SigLIP)
        let vision_encoder = VisionEncoder::new(
            &config.vision_config,
            vb.pp("model")
                .pp("vision_embed_tokens")
                .pp("img_processor")
                .pp("vision_model"),
        )?;

        // Projector: input_dim == image_dim_out because base_feat_height_reduction=1.
        // The 2x2 spatial compression (AvgPool2d) is applied in hd_transform()
        // before projection; it halves H and W but does NOT merge channels.
        let vb_ve = vb.pp("model").pp("vision_embed_tokens");
        let image_dim_out = config.image_dim_out;
        let projector =
            Phi4MMProjector::new(image_dim_out, cfg.hidden_size, vb_ve.pp("img_projection"))?;

        // Learnable separators for the HD transform (Python: glb_GN [1,1,C], sub_GN [1,1,1,C]).
        let glb_gn = vb_ve
            .get((1, 1, image_dim_out), "glb_GN")?
            .reshape((1, image_dim_out))?;
        let sub_gn = vb_ve
            .get((1, 1, 1, image_dim_out), "sub_GN")?
            .reshape((1, image_dim_out))?;

        // Audio embedding (conformer stub + projection; optional — only present
        // if audio_processor config is in the model config).
        let audio_embed = Phi4MMAudioEmbedding::new(cfg, vb.pp("embed_tokens_extend"))?;

        // LLM backbone (Llama-like)
        let vb_m = vb.pp("model");

        let embed_tokens =
            TpEmbedding::new(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"), pg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_with_tp(cfg, vb_l.pp(i), pg)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;

        let lm_head = if cfg.tie_word_embeddings && world_size == 1 {
            let emb_weights = embed_tokens
                .embeddings()
                .expect("single GPU should have accessible embeddings")
                .clone();
            TpLinear::from_linear(candle_nn::Linear::new(emb_weights, None))
        } else if cfg.tie_word_embeddings {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb_m.pp("embed_tokens"),
                pg,
            )?
        } else {
            TpLinear::column_parallel(
                cfg.hidden_size,
                cfg.vocab_size,
                false,
                true,
                vb.pp("lm_head"),
                pg,
            )?
        };

        Ok(Self {
            vision_encoder,
            projector,
            audio_embed,
            glb_gn,
            sub_gn,
            embed_tokens,
            layers,
            norm,
            lm_head,
            tp_ctx,
            config,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Apply HD transform + 2×2 compression + MLP projection to one image.
    ///
    /// `embedding`: `[num_crops * H_raw², C]` — all crop patch features concatenated.
    /// `grid_size`: `(h_crops, w_crops)` sub-image grid, or `None` / `(0,0)` for
    ///   a single global crop only.
    ///
    /// Returns `[T, hidden_size]` where T is the token count after the HD transform.
    fn hd_transform(
        &self,
        embedding: &Tensor,
        grid_size: Option<(usize, usize)>,
    ) -> Result<Tensor> {
        let (total_patches, c) = embedding.dims2()?;
        let (h_crops, w_crops) = grid_size.unwrap_or((0, 0));

        if h_crops == 0 || w_crops == 0 {
            // Global-only path: compress and add row separators.
            let h_raw = (total_patches as f64).sqrt() as usize;
            let x = embedding.reshape((1, h_raw, h_raw, c))?;
            let x = avg_pool2x2_nhwc(&x)?; // [1, H, H, C]
            let h = h_raw / 2;
            let global_img = x.squeeze(0)?.contiguous()?; // [H, H, C]

            // Append sub_GN separator at the end of each row: [H, H, C]+[H,1,C] → [H,H+1,C]
            let sep = self.sub_gn.unsqueeze(0)?.expand((h, 1, c))?;
            let glb_with_sep = Tensor::cat(&[&global_img, &sep], 1)?;
            let flat = glb_with_sep.reshape((h * (h + 1), c))?;
            return self.projector.forward(&flat);
        }

        let b_ = h_crops * w_crops;
        let num_crops = 1 + b_;
        let patches_per_crop = total_patches / num_crops;
        let h_raw = (patches_per_crop as f64).sqrt() as usize;
        let h = h_raw / 2; // after 2×2 avg-pool

        // Compress all crops.
        let x = embedding.reshape((num_crops, h_raw, h_raw, c))?;
        let x = avg_pool2x2_nhwc(&x)?; // [num_crops, H, H, C]

        // Split global (index 0) from sub crops (indices 1..).
        let global_img = x.narrow(0, 0, 1)?.squeeze(0)?.contiguous()?; // [H, H, C]
        let sub_imgs = x.narrow(0, 1, b_)?.contiguous()?; // [B_, H, H, C]

        // ── Build global arrangement ───────────────────────────────────────
        // Append sub_GN at the end of each row → [H, H+1, C] → [H*(H+1), C]
        let sep_glb = self.sub_gn.unsqueeze(0)?.expand((h, 1, c))?;
        let glb_with_sep = Tensor::cat(&[&global_img, &sep_glb], 1)?;
        let glb_flat = glb_with_sep.reshape((h * (h + 1), c))?;

        // ── Build sub arrangement ──────────────────────────────────────────
        // Arrange h_crops×w_crops patches into a spatial grid:
        //   [B_, H, H, C] → [1, h_c, w_c, H, H, C]
        //   → permute(0,1,3,2,4,5) → [1, h_c*H, w_c*H, C]
        let sub_grid = sub_imgs
            .reshape((1, h_crops, w_crops, h, h, c))?
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((1, h_crops * h, w_crops * h, c))?
            .squeeze(0)?
            .contiguous()?; // [h*H, w*H, C]

        // Append sub_GN at the end of each row → [h*H, w*H+1, C] → flat
        let sep_sub = self.sub_gn.unsqueeze(0)?.expand((h_crops * h, 1, c))?;
        let sub_with_sep = Tensor::cat(&[&sub_grid, &sep_sub], 1)?;
        let sub_flat = sub_with_sep.reshape((h_crops * h * (w_crops * h + 1), c))?;

        // ── Concatenate: sub + glb_GN + global (sub_glb order) ────────────
        let combined = Tensor::cat(&[&sub_flat, &self.glb_gn, &glb_flat], 0)?;
        self.projector.forward(&combined)
    }

    fn merge_multimodal_embeddings(
        &self,
        text_embeddings: &Tensor,
        mm_inputs: &MultimodalInputs,
    ) -> Result<Tensor> {
        if !mm_inputs.has_images() {
            return Ok(text_embeddings.clone());
        }

        let (_batch_size, seq_len, _) = text_embeddings.dims3()?;
        let mut merged = text_embeddings.to_vec3::<f32>()?;

        for (position, processed_image) in &mm_inputs.image_embeddings {
            let image_features =
                self.hd_transform(&processed_image.embedding, processed_image.grid_size)?;
            let img_emb: Vec<Vec<f32>> = image_features.to_dtype(DType::F32)?.to_vec2()?;

            let batch_idx = *position / seq_len;
            let start_pos = *position % seq_len;

            for (i, emb) in img_emb.iter().enumerate() {
                let target_pos = start_pos + i;
                if target_pos < seq_len && batch_idx < merged.len() {
                    merged[batch_idx][target_pos] = emb.clone();
                }
            }
        }

        Tensor::new(merged, &self.device)?.to_dtype(self.dtype)
    }

    fn forward_inner(
        &self,
        input_ids: &Tensor,
        multimodal_inputs: Option<&MultimodalInputs>,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        let (_b_size, seq_len) = input_ids.dims2()?;
        let attention_mask = if seq_len <= 1 {
            None
        } else {
            Some(causal_mask(
                seq_len,
                seqlen_offset,
                self.dtype,
                &self.device,
            )?)
        };

        // Audio placeholder token IDs (e.g., 200011) are out of vocabulary range.
        // Replace them with 0 before embedding; the scatter step overrides those
        // positions with pre-encoded audio embeddings anyway.
        let safe_input_ids = if let Some(mm) = multimodal_inputs {
            if mm.has_audio() {
                if let Some(audio) = &self.audio_embed {
                    let tok = audio.audio_token_id;
                    let ids = input_ids.to_vec2::<u32>()?;
                    let masked: Vec<Vec<u32>> = ids
                        .into_iter()
                        .map(|row| {
                            row.into_iter()
                                .map(|t| if t == tok { 0 } else { t })
                                .collect()
                        })
                        .collect();
                    Tensor::new(masked, input_ids.device())?
                } else {
                    input_ids.clone()
                }
            } else {
                input_ids.clone()
            }
        } else {
            input_ids.clone()
        };
        let text_embeddings = self.embed_tokens.forward(&safe_input_ids, &self.tp_ctx)?;

        let mut xs = if let Some(mm_inputs) = multimodal_inputs {
            let xs = if mm_inputs.has_images() {
                self.merge_multimodal_embeddings(&text_embeddings, mm_inputs)?
            } else {
                text_embeddings
            };
            if mm_inputs.has_audio() {
                if let Some(audio) = &self.audio_embed {
                    scatter_audio_into_text(&xs, mm_inputs, audio.audio_token_id)?
                } else {
                    xs
                }
            } else {
                xs
            }
        } else {
            text_embeddings
        };

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward(
                &xs,
                attention_mask.as_ref(),
                seqlen_offset,
                kv_cache_mgr,
                layer_idx,
                block_table,
                slot_mapping,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
    }
}

// ─── Audio scatter ────────────────────────────────────────────────────────────

/// Scatter pre-encoded audio embeddings into text embedding positions.
///
/// Works identically to the audio scatter in other audio-language models:
/// iterates token_ids and replaces `audio_token_id` positions with
/// successive rows from the sorted audio clip embeddings.
fn scatter_audio_into_text(
    text_embeds: &Tensor,
    mm: &MultimodalInputs,
    audio_token_id: u32,
) -> Result<Tensor> {
    if mm.audio_embeddings.is_empty() {
        return Ok(text_embeds.clone());
    }

    let (b, s, d) = text_embeds.dims3()?;
    let mut audio_clips: Vec<(usize, Tensor)> = mm
        .audio_embeddings
        .iter()
        .map(|(pos, pa)| (*pos, pa.embedding.clone()))
        .collect();
    audio_clips.sort_by_key(|(pos, _)| *pos);

    let flat_embeds = text_embeds.reshape((b * s, d))?;
    let token_ids = &mm.token_ids;

    let mut rows: Vec<Tensor> = Vec::with_capacity(b * s);
    let mut clip_idx = 0usize;
    let mut clip_offset = 0usize;

    for (seq_idx, &tok) in token_ids.iter().enumerate() {
        if tok == audio_token_id && clip_idx < audio_clips.len() {
            let clip = &audio_clips[clip_idx].1;
            let clip_len = clip.dim(0)?;
            rows.push(clip.narrow(0, clip_offset, 1)?.squeeze(0)?);
            clip_offset += 1;
            if clip_offset >= clip_len {
                clip_idx += 1;
                clip_offset = 0;
            }
        } else {
            rows.push(flat_embeds.narrow(0, seq_idx, 1)?.squeeze(0)?);
        }
    }

    Tensor::stack(&rows, 0)?.reshape((b, s, d))
}

impl crate::engine::ModelForward for Phi4MMForCausalLM {
    fn forward(
        &self,
        input_ids: &Tensor,
        seqlen_offset: usize,
        kv_cache_mgr: &mut KVCacheManager,
        block_table: &BlockTable,
        slot_mapping: &[usize],
    ) -> Result<Tensor> {
        self.forward_inner(
            input_ids,
            None,
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
        let mut xs = self.embed_tokens.forward(input_ids, &self.tp_ctx)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            xs = layer.forward_decode_batch(
                &xs,
                sequences,
                kv_cache_mgr,
                layer_idx,
                &self.tp_ctx,
            )?;
        }

        let xs = self.norm.forward(&xs)?;
        self.lm_head.forward(&xs, &self.tp_ctx)
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
        self.forward_inner(
            input_ids,
            multimodal_inputs,
            seqlen_offset,
            kv_cache_mgr,
            block_table,
            slot_mapping,
        )
    }

    fn device(&self) -> &Device {
        &self.device
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::ModelForward;
    use crate::kv_cache::{config::CacheConfig, KVCacheDtype};
    use crate::multimodal::ProcessedImage;

    fn test_model_config() -> ModelConfig {
        let mut extra = serde_json::Map::new();
        extra.insert(
            "img_processor".to_string(),
            serde_json::json!({
                "hidden_size": 64,
                "intermediate_size": 128,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "image_size": 56,
                "patch_size": 14,
                "model_type": "siglip"
            }),
        );
        extra.insert("use_hd_transform".to_string(), serde_json::json!(true));
        extra.insert("image_token_id".to_string(), serde_json::json!(200010));

        ModelConfig {
            architectures: vec!["Phi4MMForCausalLM".to_string()],
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 128,
            vocab_size: 256,
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

    fn create_cache_config(cfg: &ModelConfig, device: &Device) -> CacheConfig {
        CacheConfig {
            block_size: 16,
            num_blocks: 4,
            num_layers: cfg.num_hidden_layers,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            dtype: DType::F32,
            device: device.clone(),
            kv_cache_dtype: KVCacheDtype::Auto,
            cpu_offload: None,
        }
    }

    #[test]
    fn test_model_construction() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb);
        assert!(model.is_ok(), "construction failed: {:?}", model.err());
        assert!(model.unwrap().supports_multimodal());
    }

    #[test]
    fn test_text_only_forward() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let block_table = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..4).collect();

        let input_ids = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let logits = model
            .forward(&input_ids, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, 4, cfg.vocab_size]);
    }

    #[test]
    fn test_decode_batch() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let sequences = vec![
            DecodeSequenceMetadata {
                request_id: 1,
                seqlen_offset: 5,
                block_ids: vec![0],
                slot_mapping: vec![5],
            },
            DecodeSequenceMetadata {
                request_id: 2,
                seqlen_offset: 3,
                block_ids: vec![1],
                slot_mapping: vec![3],
            },
        ];

        let input_ids = Tensor::from_vec(vec![10u32, 20], (2, 1), &device).unwrap();
        let logits = model
            .forward_decode_batch(&input_ids, &sequences, &mut kv_cache)
            .unwrap();

        assert_eq!(logits.dim(0).unwrap(), 2);
    }

    #[test]
    fn test_prefill_then_decode() {
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb).unwrap();

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let mut block_table = BlockTable::new(cache_cfg.block_size);

        let prompt = Tensor::zeros((1, 3), DType::U32, &device).unwrap();
        kv_cache.allocate_for_request(&mut block_table, 3).unwrap();
        let slot_mapping = block_table.slot_mapping(0, 3);

        let logits = model
            .forward(&prompt, 0, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 3, cfg.vocab_size]);
        block_table.advance(3);

        kv_cache.allocate_for_request(&mut block_table, 1).unwrap();
        let slot_mapping = block_table.slot_mapping(3, 1);
        let next_token = Tensor::zeros((1, 1), DType::U32, &device).unwrap();

        let logits = model
            .forward(&next_token, 3, &mut kv_cache, &block_table, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, 1, cfg.vocab_size]);
    }

    /// Helper: test config uses image_size=56, patch_size=14 → 4 patches per side per crop.
    /// After 2×2 AvgPool: H=2, 4 patches per crop.
    fn make_processed_image(
        num_crops: usize,
        c: usize,
        h_raw: usize,
        grid_size: Option<(usize, usize)>,
        device: &Device,
    ) -> ProcessedImage {
        let total = num_crops * h_raw * h_raw;
        let emb = Tensor::zeros((total, c), DType::F32, device).unwrap();
        let mut pi = ProcessedImage::new(emb, /* num_tokens placeholder */ 0);
        pi.grid_size = grid_size;
        pi
    }

    #[test]
    fn test_forward_multimodal_single_image() {
        // Single global crop (no sub-images). H_raw=4, H=2 after compression.
        // HD-transform output tokens = H*(H+1) = 2*3 = 6.
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb).unwrap();

        let image_dim_out = 64; // matches test config hidden_size
        let h_raw = 4; // image_size/patch_size = 56/14 = 4 patches per side
        let h = h_raw / 2; // 2 after AvgPool
        let num_tokens = h * (h + 1); // 6

        // Sequence: 3 text tokens + 6 image tokens (positions 0..5 are image)
        let seq_len = 3 + num_tokens;
        let pi = make_processed_image(1, image_dim_out, h_raw, None, &device);
        let mm = MultimodalInputs::with_images(vec![0u32; seq_len], vec![(0, pi)]);

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let bt = BlockTable::from_block_ids(vec![0, 1], 0);
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        let logits = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut kv_cache, &bt, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_forward_multimodal_hd_transform() {
        // 2×2 sub crops + 1 global = 5 crops total. H_raw=4, H=2.
        // sub tokens = h_c * H * (w_c * H + 1) = 2*2*(2*2+1) = 20
        // glb tokens = H*(H+1) = 6
        // separator = 1
        // Total image tokens = 20 + 1 + 6 = 27.
        let device = Device::Cpu;
        let cfg = test_model_config();
        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb).unwrap();

        let image_dim_out = 64;
        let h_raw = 4;
        let h = h_raw / 2;
        let h_crops = 2usize;
        let w_crops = 2usize;
        let num_tokens = h_crops * h * (w_crops * h + 1) // sub + sub_GN separators
            + 1                              // glb_GN
            + h * (h + 1); // global + sub_GN per row

        let seq_len = num_tokens + 2; // 2 extra text tokens
        let num_crops = 1 + h_crops * w_crops; // 5
        let pi = make_processed_image(
            num_crops,
            image_dim_out,
            h_raw,
            Some((h_crops, w_crops)),
            &device,
        );
        let mm = MultimodalInputs::with_images(vec![0u32; seq_len], vec![(0, pi)]);

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();
        let bt = BlockTable::from_block_ids(vec![0, 1, 2], 0);
        let slot_mapping: Vec<usize> = (0..seq_len).collect();

        let input_ids = Tensor::zeros((1, seq_len), DType::U32, &device).unwrap();
        let logits = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut kv_cache, &bt, &slot_mapping)
            .unwrap();

        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }

    #[test]
    fn test_forward_multimodal_audio() {
        use crate::kv_cache::KVCacheManager;
        use crate::multimodal::ProcessedAudio;

        let device = Device::Cpu;
        let mut extra = serde_json::Map::new();
        extra.insert("audio_token_id".into(), serde_json::json!(200011u32));
        extra.insert(
            "audio_processor".into(),
            serde_json::json!({
                "name": "cascades",
                "config": { "input_size": 8, "attention_dim": 8,
                            "attention_heads": 2, "linear_units": 16, "num_block": 1 }
            }),
        );
        extra.insert(
            "embd_layer".into(),
            serde_json::json!({
                "audio_embd_layer": {
                    "embedding_cls": "cascades",
                    "projection_cls": "mlp",
                    "downsample_rate": 1,
                    "use_qformer": false,
                    "use_conv_downsample": false
                }
            }),
        );
        extra.insert(
            "img_processor".to_string(),
            serde_json::json!({
                "hidden_size": 8,
                "intermediate_size": 16,
                "num_hidden_layers": 1,
                "num_attention_heads": 2,
                "image_size": 28,
                "patch_size": 14,
                "model_type": "siglip"
            }),
        );
        extra.insert("use_hd_transform".to_string(), serde_json::json!(false));
        extra.insert("image_token_id".to_string(), serde_json::json!(200010u32));

        let cfg = ModelConfig {
            architectures: vec!["Phi4MMForCausalLM".to_string()],
            hidden_size: 8,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            num_hidden_layers: 1,
            intermediate_size: 16,
            vocab_size: 32,
            max_position_embeddings: 64,
            head_dim: 4,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            extra,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Phi4MMForCausalLM::new(&cfg, vb).unwrap();
        assert!(model.audio_embed.is_some());

        let cache_cfg = create_cache_config(&cfg, &device);
        let mut kv_cache = KVCacheManager::new(&cache_cfg).unwrap();

        let audio_token_id: u32 = 200011;
        let token_ids = vec![1u32, audio_token_id, audio_token_id, 2u32];
        let seq_len = token_ids.len();
        let audio_emb = Tensor::zeros((2usize, 8usize), DType::F32, &device).unwrap();
        let processed = ProcessedAudio::new(audio_emb, 2);
        let mm = MultimodalInputs::with_audio(token_ids.clone(), vec![(1, processed)]);

        let mut bt = BlockTable::new(cache_cfg.block_size);
        kv_cache.allocate_for_request(&mut bt, seq_len).unwrap();
        let slot_mapping = bt.slot_mapping(0, seq_len);

        let input_ids = Tensor::from_vec(token_ids, (1usize, seq_len), &device)
            .unwrap()
            .to_dtype(DType::U32)
            .unwrap();
        let logits = model
            .forward_multimodal(&input_ids, Some(&mm), 0, &mut kv_cache, &bt, &slot_mapping)
            .unwrap();
        assert_eq!(logits.dims(), &[1, seq_len, cfg.vocab_size]);
    }
}
