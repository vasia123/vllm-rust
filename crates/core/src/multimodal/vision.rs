//! Vision encoder implementations for multimodal models.
//!
//! Provides CLIP and SigLIP vision encoders that convert images
//! to embedding tensors compatible with language model hidden states.

use candle_core::{DType, Device, Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Embedding, LayerNorm, Linear, VarBuilder};

use super::inputs::ProcessedImage;

/// Vision encoder type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VisionEncoderType {
    /// CLIP ViT encoder (OpenAI).
    Clip,
    /// SigLIP encoder (Google).
    SigLip,
}

/// Configuration for vision encoders.
#[derive(Debug, Clone)]
pub struct VisionEncoderConfig {
    /// Type of vision encoder.
    pub encoder_type: VisionEncoderType,
    /// Hidden size of the vision encoder.
    pub hidden_size: usize,
    /// Intermediate size in MLP blocks.
    pub intermediate_size: usize,
    /// Number of attention heads.
    pub num_attention_heads: usize,
    /// Number of transformer layers.
    pub num_hidden_layers: usize,
    /// Image size (square).
    pub image_size: usize,
    /// Patch size (square).
    pub patch_size: usize,
    /// Number of channels (3 for RGB).
    pub num_channels: usize,
    /// Layer norm epsilon.
    pub layer_norm_eps: f64,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        // CLIP ViT-L/14 @ 336px defaults
        Self {
            encoder_type: VisionEncoderType::Clip,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_attention_heads: 16,
            num_hidden_layers: 24,
            image_size: 336,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-5,
        }
    }
}

impl VisionEncoderConfig {
    /// CLIP ViT-L/14 @ 336px configuration.
    pub fn clip_vit_l_14_336() -> Self {
        Self::default()
    }

    /// CLIP ViT-L/14 @ 224px configuration.
    pub fn clip_vit_l_14_224() -> Self {
        Self {
            image_size: 224,
            ..Self::default()
        }
    }

    /// SigLIP ViT-SO400M/14 @ 384px configuration.
    pub fn siglip_so400m_384() -> Self {
        Self {
            encoder_type: VisionEncoderType::SigLip,
            hidden_size: 1152,
            intermediate_size: 4304,
            num_attention_heads: 16,
            num_hidden_layers: 27,
            image_size: 384,
            patch_size: 14,
            num_channels: 3,
            layer_norm_eps: 1e-6,
        }
    }

    /// Number of patches per image dimension.
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches (excluding CLS token if any).
    pub fn num_patches(&self) -> usize {
        let n = self.num_patches_per_side();
        n * n
    }

    /// Sequence length including CLS token (CLIP) or not (SigLIP).
    pub fn seq_len(&self) -> usize {
        match self.encoder_type {
            VisionEncoderType::Clip => self.num_patches() + 1, // +1 for CLS token
            VisionEncoderType::SigLip => self.num_patches(),   // No CLS token
        }
    }
}

// ─── Patch Embedding ─────────────────────────────────────────────────────────

/// Converts images into patch embeddings using a convolutional layer.
struct PatchEmbedding {
    proj: Conv2d,
    #[allow(dead_code)] // For future dynamic patch handling
    num_patches: usize,
    #[allow(dead_code)] // For future dynamic patch handling
    hidden_size: usize,
}

impl PatchEmbedding {
    fn new(cfg: &VisionEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let proj = candle_nn::conv2d(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            Conv2dConfig {
                stride: cfg.patch_size,
                padding: 0,
                dilation: 1,
                groups: 1,
                cudnn_fwd_algo: None,
            },
            vb.pp("patch_embedding"),
        )?;

        Ok(Self {
            proj,
            num_patches: cfg.num_patches(),
            hidden_size: cfg.hidden_size,
        })
    }

    fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // pixel_values: [batch, channels, height, width]
        let _batch_size = pixel_values.dim(0)?;

        // Apply convolution: [batch, hidden_size, num_patches_h, num_patches_w]
        let embeddings = self.proj.forward(pixel_values)?;

        // Flatten spatial dimensions: [batch, hidden_size, num_patches]
        let embeddings = embeddings.flatten(2, 3)?;

        // Transpose to [batch, num_patches, hidden_size]
        embeddings.transpose(1, 2)
    }
}

// ─── Vision Attention ────────────────────────────────────────────────────────

struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn new(cfg: &VisionEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;

        let q_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Project Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Attention scores
        let attn_weights = q.matmul(&k.transpose(2, 3)?)?;
        let attn_weights = (attn_weights * self.scale)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        // Apply attention to values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back: [batch, seq_len, hidden_size]
        let attn_output = attn_output.transpose(1, 2)?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }
}

// ─── Vision MLP ──────────────────────────────────────────────────────────────

struct VisionMLP {
    fc1: Linear,
    fc2: Linear,
    activation: Activation,
}

#[derive(Debug, Clone, Copy)]
enum Activation {
    QuickGelu,
    Gelu,
}

impl VisionMLP {
    fn new(cfg: &VisionEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))?;

        let activation = match cfg.encoder_type {
            VisionEncoderType::Clip => Activation::QuickGelu,
            VisionEncoderType::SigLip => Activation::Gelu,
        };

        Ok(Self {
            fc1,
            fc2,
            activation,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let hidden = self.fc1.forward(hidden_states)?;
        let hidden = match self.activation {
            Activation::QuickGelu => quick_gelu(&hidden)?,
            Activation::Gelu => hidden.gelu_erf()?,
        };
        self.fc2.forward(&hidden)
    }
}

/// QuickGELU activation: x * sigmoid(1.702 * x)
fn quick_gelu(x: &Tensor) -> Result<Tensor> {
    let sigmoid_input = (x * 1.702)?;
    x.mul(&candle_nn::ops::sigmoid(&sigmoid_input)?)
}

// ─── Vision Encoder Layer ────────────────────────────────────────────────────

struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMLP,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
}

impl VisionEncoderLayer {
    fn new(cfg: &VisionEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = VisionAttention::new(cfg, vb.pp("self_attn"))?;
        let mlp = VisionMLP::new(cfg, vb.pp("mlp"))?;

        let layer_norm1 =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm1"))?;
        let layer_norm2 =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("layer_norm2"))?;

        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        // Pre-norm attention
        let residual = hidden_states;
        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(&hidden_states)?;
        let hidden_states = (residual + hidden_states)?;

        // Pre-norm MLP
        let residual = &hidden_states;
        let hidden_states = self.layer_norm2.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        residual + hidden_states
    }
}

// ─── Vision Encoder ──────────────────────────────────────────────────────────

/// Vision encoder that processes images into embeddings.
pub struct VisionEncoder {
    patch_embedding: PatchEmbedding,
    class_embedding: Option<Tensor>,
    position_embedding: Embedding,
    layers: Vec<VisionEncoderLayer>,
    pre_layernorm: Option<LayerNorm>,
    post_layernorm: LayerNorm,
    config: VisionEncoderConfig,
    device: Device,
    dtype: DType,
}

impl VisionEncoder {
    /// Create a new vision encoder.
    pub fn new(cfg: &VisionEncoderConfig, vb: VarBuilder) -> Result<Self> {
        let patch_embedding = PatchEmbedding::new(cfg, vb.clone())?;

        // CLS token embedding (CLIP only)
        let class_embedding = if cfg.encoder_type == VisionEncoderType::Clip {
            Some(vb.get((1, 1, cfg.hidden_size), "class_embedding")?)
        } else {
            None
        };

        // Position embeddings
        let position_embedding =
            candle_nn::embedding(cfg.seq_len(), cfg.hidden_size, vb.pp("position_embedding"))?;

        // Transformer layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("encoder.layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::new(cfg, vb_layers.pp(i))?);
        }

        // Layer norms
        let pre_layernorm = if cfg.encoder_type == VisionEncoderType::Clip {
            Some(candle_nn::layer_norm(
                cfg.hidden_size,
                cfg.layer_norm_eps,
                vb.pp("pre_layernorm"),
            )?)
        } else {
            None
        };

        let post_layernorm =
            candle_nn::layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("post_layernorm"))?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            layers,
            pre_layernorm,
            post_layernorm,
            config: cfg.clone(),
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Encode a batch of images into embeddings.
    ///
    /// # Arguments
    /// * `pixel_values` - Image tensor [batch, channels, height, width], normalized to [-1, 1] or [0, 1]
    ///
    /// # Returns
    /// Embeddings tensor [batch, num_tokens, hidden_size]
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let batch_size = pixel_values.dim(0)?;

        // Get patch embeddings
        let mut embeddings = self.patch_embedding.forward(pixel_values)?;

        // Add CLS token (CLIP)
        if let Some(cls) = &self.class_embedding {
            let cls = cls.broadcast_as((batch_size, 1, self.config.hidden_size))?;
            embeddings = Tensor::cat(&[cls, embeddings], 1)?;
        }

        // Add position embeddings
        let seq_len = embeddings.dim(1)?;
        let position_ids = Tensor::arange(0u32, seq_len as u32, &self.device)?;
        let position_embeddings = self.position_embedding.forward(&position_ids)?;
        embeddings = embeddings.broadcast_add(&position_embeddings)?;

        // Pre layer norm (CLIP)
        if let Some(ln) = &self.pre_layernorm {
            embeddings = ln.forward(&embeddings)?;
        }

        // Apply transformer layers
        for layer in &self.layers {
            embeddings = layer.forward(&embeddings)?;
        }

        // Post layer norm
        self.post_layernorm.forward(&embeddings)
    }

    /// Encode a single image and return a ProcessedImage.
    pub fn encode_image(&self, pixel_values: &Tensor) -> Result<ProcessedImage> {
        let embeddings = self.forward(pixel_values)?;
        let num_tokens = embeddings.dim(1)?;

        // For CLIP, we can optionally return only patch embeddings (exclude CLS)
        // For now, return all tokens
        let embedding = embeddings.squeeze(0)?; // Remove batch dim

        Ok(ProcessedImage::new(embedding, num_tokens))
    }

    /// Get the hidden size of the encoder.
    pub fn hidden_size(&self) -> usize {
        self.config.hidden_size
    }

    /// Get the number of output tokens per image.
    pub fn num_image_tokens(&self) -> usize {
        self.config.seq_len()
    }

    /// Get the expected image size.
    pub fn image_size(&self) -> usize {
        self.config.image_size
    }

    /// Get the device.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the dtype.
    pub fn dtype(&self) -> DType {
        self.dtype
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_clip_vit_l_14_336() {
        let cfg = VisionEncoderConfig::clip_vit_l_14_336();
        assert_eq!(cfg.image_size, 336);
        assert_eq!(cfg.patch_size, 14);
        assert_eq!(cfg.num_patches_per_side(), 24);
        assert_eq!(cfg.num_patches(), 576);
        assert_eq!(cfg.seq_len(), 577); // +1 for CLS
    }

    #[test]
    fn test_config_siglip() {
        let cfg = VisionEncoderConfig::siglip_so400m_384();
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.num_patches_per_side(), 27); // 384/14 = 27.4 -> 27
        assert_eq!(cfg.num_patches(), 729);
        assert_eq!(cfg.seq_len(), 729); // No CLS token
    }

    #[test]
    fn test_vision_encoder_creation() {
        let device = Device::Cpu;
        let cfg = VisionEncoderConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 56,
            patch_size: 14,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let encoder = VisionEncoder::new(&cfg, vb);
        assert!(encoder.is_ok());

        let encoder = encoder.unwrap();
        assert_eq!(encoder.hidden_size(), 64);
        assert_eq!(encoder.image_size(), 56);
    }

    #[test]
    fn test_vision_encoder_forward() {
        let device = Device::Cpu;
        let cfg = VisionEncoderConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_attention_heads: 4,
            num_hidden_layers: 2,
            image_size: 28,
            patch_size: 14,
            ..Default::default()
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let encoder = VisionEncoder::new(&cfg, vb).unwrap();

        // Create dummy image: [batch=1, channels=3, height=28, width=28]
        let pixel_values = Tensor::randn(0f32, 1.0, (1, 3, 28, 28), &device).unwrap();
        let embeddings = encoder.forward(&pixel_values).unwrap();

        // Expected: [1, num_patches + 1 (CLS), hidden_size] = [1, 5, 64]
        // 28/14 = 2 patches per side, 2*2 = 4 patches + 1 CLS = 5
        assert_eq!(embeddings.dims(), &[1, 5, 64]);
    }

    #[test]
    fn test_quick_gelu() {
        let device = Device::Cpu;
        let x = Tensor::new(&[0.0f32, 1.0, -1.0], &device).unwrap();
        let result = quick_gelu(&x).unwrap();

        // QuickGELU(0) = 0
        // QuickGELU(1) ≈ 0.8458
        // QuickGELU(-1) ≈ -0.1542
        let values: Vec<f32> = result.to_vec1().unwrap();
        assert!((values[0] - 0.0).abs() < 1e-5);
        assert!((values[1] - 0.8458).abs() < 0.01);
        assert!((values[2] - (-0.1542)).abs() < 0.01);
    }
}
