//! Gemma 4 image preprocessing.
//!
//! Mirrors `transformers.models.gemma4.image_processing_gemma4.Gemma4ImageProcessor`
//! closely enough for `Gemma4VisionTower` to consume the output verbatim.
//!
//! Pipeline per image:
//!   1. aspect-ratio-preserving resize so (H·W)/patch² ≤ `max_patches` and
//!      both sides divisible by `patch_size · pooling_kernel_size`;
//!   2. rescale to `[0, 1]` by dividing by 255 (Gemma 4 does NOT apply
//!      ImageNet normalization — the model's `patch_embedder` rescales
//!      internally via `2·(x-0.5)`);
//!   3. patchify into `[N, 3·ps·ps]`;
//!   4. generate `[N, 2]` `(x, y)` position ids (row-major meshgrid);
//!   5. pad both tensors along the N axis up to `max_patches` with
//!      zero pixels / `(-1, -1)` positions.
//!
//! The pure aspect-ratio helper is always compiled; the full `preprocess`
//! pipeline only when the `image-loading` feature is enabled.

use candle_core::{DType, Device, Result, Tensor};

/// Sizes supported by HuggingFace Gemma 4: {70, 140, 280, 560, 1120}.
pub const SUPPORTED_SOFT_TOKENS: &[usize] = &[70, 140, 280, 560, 1120];

#[derive(Debug, Clone)]
pub struct Gemma4ImageProcessorConfig {
    pub patch_size: usize,
    pub max_soft_tokens: usize,
    pub pooling_kernel_size: usize,
    /// Multiplier applied to uint8 pixels before passing to the model.
    /// HF default is `1/255`.
    pub rescale_factor: f32,
}

impl Default for Gemma4ImageProcessorConfig {
    fn default() -> Self {
        Self {
            patch_size: 16,
            max_soft_tokens: 280,
            pooling_kernel_size: 3,
            rescale_factor: 1.0 / 255.0,
        }
    }
}

/// Compute the target (height, width) that preserves aspect ratio and
/// fits within the patch budget.
///
/// Matches HF's `get_aspect_ratio_preserving_size`. Both target sides are
/// multiples of `patch_size · pooling_kernel_size`.
///
/// Returns `Err(&'static str)` if the inputs would round both sides to
/// zero (image too small for any valid patchification).
pub fn aspect_ratio_preserving_size(
    height: usize,
    width: usize,
    patch_size: usize,
    max_patches: usize,
    pooling_kernel_size: usize,
) -> std::result::Result<(usize, usize), &'static str> {
    let total_px = (height * width) as f64;
    let target_px = (max_patches * patch_size * patch_size) as f64;
    let factor = (target_px / total_px).sqrt();
    let side_mult = pooling_kernel_size * patch_size;

    let ideal_h = factor * height as f64;
    let ideal_w = factor * width as f64;
    let mut target_h = (ideal_h / side_mult as f64).floor() as usize * side_mult;
    let mut target_w = (ideal_w / side_mult as f64).floor() as usize * side_mult;

    if target_h == 0 && target_w == 0 {
        return Err("input resolution rounds to 0x0 after aspect-ratio resize");
    }

    let max_side_length = (max_patches / (pooling_kernel_size * pooling_kernel_size)) * side_mult;
    if target_h == 0 {
        target_h = side_mult;
        let approx = (width as f64 / height as f64).floor() as usize * side_mult;
        target_w = approx.min(max_side_length);
    } else if target_w == 0 {
        target_w = side_mult;
        let approx = (height as f64 / width as f64).floor() as usize * side_mult;
        target_h = approx.min(max_side_length);
    }

    if target_h * target_w > (max_patches * patch_size * patch_size) {
        return Err("aspect-ratio resize exceeded the patch budget");
    }

    Ok((target_h, target_w))
}

/// Convert a row-major `(H, W, 3)` u8 buffer into the `[N, 3·ps·ps]`
/// Gemma 4 patch layout.
///
/// The layout exactly mirrors HF: `image.reshape(C, nh, ps, nw, ps)
/// .permute(1, 3, 2, 4, 0).reshape(nh·nw, -1)` — i.e. for each patch the
/// 2D (ps × ps × C) block is flattened in (patch-row, patch-col,
/// channel) order.
pub(crate) fn patchify_u8(
    rgb: &[u8],
    height: usize,
    width: usize,
    patch_size: usize,
    rescale_factor: f32,
) -> Vec<f32> {
    assert_eq!(
        rgb.len(),
        height * width * 3,
        "patchify_u8: buffer size mismatch"
    );
    assert!(
        height.is_multiple_of(patch_size) && width.is_multiple_of(patch_size),
        "patchify_u8: dimensions must be multiples of patch_size"
    );
    let nh = height / patch_size;
    let nw = width / patch_size;
    let ps = patch_size;
    let patch_px = ps * ps * 3;
    let mut out = vec![0f32; nh * nw * patch_px];

    for py in 0..nh {
        for px in 0..nw {
            let patch_idx = py * nw + px;
            let out_base = patch_idx * patch_px;
            // Flatten in (row, col, channel) order — matching HF's
            // permute(1, 3, 2, 4, 0) reshape.
            for dy in 0..ps {
                for dx in 0..ps {
                    let src_y = py * ps + dy;
                    let src_x = px * ps + dx;
                    let src = (src_y * width + src_x) * 3;
                    let dst = out_base + (dy * ps + dx) * 3;
                    out[dst] = rgb[src] as f32 * rescale_factor;
                    out[dst + 1] = rgb[src + 1] as f32 * rescale_factor;
                    out[dst + 2] = rgb[src + 2] as f32 * rescale_factor;
                }
            }
        }
    }
    out
}

/// Per-image preprocessing result.
#[derive(Debug, Clone)]
pub struct Gemma4PreprocessedImage {
    /// `[max_patches, 3·ps·ps]` float pixel patches (row-major).
    pub pixel_values: Tensor,
    /// `[max_patches, 2]` i64 `(x, y)` patch coordinates. Padding rows
    /// are `(-1, -1)`.
    pub position_ids: Tensor,
    /// How many soft tokens this image produces after pooling.
    pub num_soft_tokens: usize,
}

#[cfg(feature = "image-loading")]
pub struct Gemma4ImageProcessor {
    cfg: Gemma4ImageProcessorConfig,
    device: Device,
}

#[cfg(feature = "image-loading")]
impl Gemma4ImageProcessor {
    pub fn new(cfg: Gemma4ImageProcessorConfig, device: Device) -> Self {
        Self { cfg, device }
    }

    pub fn config(&self) -> &Gemma4ImageProcessorConfig {
        &self.cfg
    }

    /// Preprocess a single image given its raw encoded bytes (PNG / JPEG /
    /// WebP — whatever the `image` crate can decode). Returns padded
    /// pixel patches + position ids + number of soft tokens.
    pub fn preprocess_bytes(
        &self,
        bytes: &[u8],
    ) -> std::result::Result<Gemma4PreprocessedImage, String> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| format!("image decode failed: {e}"))?
            .to_rgb8();
        self.preprocess_rgb(img.width() as usize, img.height() as usize, img.as_raw())
    }

    /// Preprocess a decoded RGB image (row-major `height * width * 3`).
    pub fn preprocess_rgb(
        &self,
        width: usize,
        height: usize,
        rgb_row_major: &[u8],
    ) -> std::result::Result<Gemma4PreprocessedImage, String> {
        assert_eq!(
            rgb_row_major.len(),
            width * height * 3,
            "preprocess_rgb: buffer size mismatch"
        );

        let max_patches =
            self.cfg.max_soft_tokens * self.cfg.pooling_kernel_size * self.cfg.pooling_kernel_size;

        let (th, tw) = aspect_ratio_preserving_size(
            height,
            width,
            self.cfg.patch_size,
            max_patches,
            self.cfg.pooling_kernel_size,
        )
        .map_err(|e| e.to_string())?;

        // Resize via `image` crate (bicubic approximation via CatmullRom).
        let src = image::ImageBuffer::<image::Rgb<u8>, _>::from_raw(
            width as u32,
            height as u32,
            rgb_row_major.to_vec(),
        )
        .ok_or_else(|| "could not wrap raw RGB buffer".to_string())?;
        let resized = if th == height && tw == width {
            src
        } else {
            image::imageops::resize(
                &src,
                tw as u32,
                th as u32,
                image::imageops::FilterType::CatmullRom,
            )
        };

        let ps = self.cfg.patch_size;
        let flat_patches = patchify_u8(resized.as_raw(), th, tw, ps, self.cfg.rescale_factor);

        let n_patches = (th / ps) * (tw / ps);
        let patch_pixels = 3 * ps * ps;
        let pool_k2 = self.cfg.pooling_kernel_size * self.cfg.pooling_kernel_size;
        assert!(
            n_patches.is_multiple_of(pool_k2),
            "aspect-ratio resize produced {n_patches} patches which is not a multiple of pool_k²={pool_k2}"
        );
        let num_soft_tokens = n_patches / pool_k2;

        // Positions: row-major meshgrid over (y, x). HF uses
        // `meshgrid(arange(W), arange(H), indexing='xy')` then
        // `stack(dim=-1)` — which yields a `(H, W, 2)` tensor with entry
        // `(x, y)` at position `[y, x]`. We emit them row-by-row.
        let mut positions = Vec::with_capacity(n_patches * 2);
        for y in 0..(th / ps) as i64 {
            for x in 0..(tw / ps) as i64 {
                positions.push(x);
                positions.push(y);
            }
        }

        // Pad up to max_patches.
        let mut padded_pixels = flat_patches;
        padded_pixels.resize(max_patches * patch_pixels, 0.0);
        let mut padded_positions = positions;
        padded_positions.resize(max_patches * 2, -1);

        let pixel_values =
            Tensor::from_vec(padded_pixels, (max_patches, patch_pixels), &self.device)
                .map_err(|e| format!("pixel tensor: {e}"))?;
        let position_ids = Tensor::from_vec(padded_positions, (max_patches, 2), &self.device)
            .map_err(|e| format!("position tensor: {e}"))?;

        Ok(Gemma4PreprocessedImage {
            pixel_values,
            position_ids,
            num_soft_tokens,
        })
    }

    /// Stack a batch of preprocessed images into `[B, max_patches, …]`.
    pub fn batch(
        &self,
        per_image: &[Gemma4PreprocessedImage],
    ) -> std::result::Result<(Tensor, Tensor, Vec<usize>), String> {
        assert!(!per_image.is_empty(), "batch: no images");
        let px_refs: Vec<&Tensor> = per_image.iter().map(|p| &p.pixel_values).collect();
        let pos_refs: Vec<&Tensor> = per_image.iter().map(|p| &p.position_ids).collect();

        let pixel_values = Tensor::stack(&px_refs, 0).map_err(|e| format!("stack pixels: {e}"))?;
        let position_ids =
            Tensor::stack(&pos_refs, 0).map_err(|e| format!("stack positions: {e}"))?;
        let tokens = per_image.iter().map(|p| p.num_soft_tokens).collect();
        Ok((pixel_values, position_ids, tokens))
    }
}

/// Convenience helper that wires `patchify_u8` directly into Candle tensors
/// without invoking `image`. Useful for tests/benchmarks with synthetic
/// data when the `image-loading` feature is off.
pub fn patches_to_tensor(
    rgb_row_major: &[u8],
    height: usize,
    width: usize,
    patch_size: usize,
    rescale_factor: f32,
    device: &Device,
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    let flat = patchify_u8(rgb_row_major, height, width, patch_size, rescale_factor);
    let nh = height / patch_size;
    let nw = width / patch_size;
    let n = nh * nw;
    let ps2 = patch_size * patch_size * 3;
    let pixels = Tensor::from_vec(flat, (n, ps2), device)?.to_dtype(dtype)?;

    let mut positions = Vec::with_capacity(n * 2);
    for y in 0..nh as i64 {
        for x in 0..nw as i64 {
            positions.push(x);
            positions.push(y);
        }
    }
    let pos = Tensor::from_vec(positions, (n, 2), device)?;

    Ok((pixels, pos))
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aspect_ratio_preserving_exact_square() {
        // 480×480 with patch=16, max_patches=280·9=2520, pool_k=3.
        // side_mult=48, target_px=2520·256=645120. sqrt(645120/230400)≈1.672.
        // ideal=802.6 → target=768 (floor(802.6/48)·48=768). 768·768=589824 ≤ 645120.
        let (h, w) = aspect_ratio_preserving_size(480, 480, 16, 2520, 3).unwrap();
        assert_eq!((h, w), (768, 768));
        assert_eq!(h % (16 * 3), 0);
    }

    #[test]
    fn aspect_ratio_landscape_preserves_ratio() {
        // 200×400 image, 2:1 aspect ratio.
        let (h, w) = aspect_ratio_preserving_size(200, 400, 16, 2520, 3).unwrap();
        assert_eq!(h % (16 * 3), 0);
        assert_eq!(w % (16 * 3), 0);
        // Width should remain ≈ 2× height.
        assert!((w as f64 / h as f64 - 2.0).abs() < 0.5);
        // Total patches should not exceed budget.
        assert!((h / 16) * (w / 16) <= 2520);
    }

    #[test]
    fn patchify_round_trip_identity() {
        // 4×4 image, 2×2 patches → 4 patches.
        #[rustfmt::skip]
        let rgb: Vec<u8> = vec![
            // row 0
            1, 2, 3,   4, 5, 6,   7, 8, 9,   10, 11, 12,
            // row 1
            13,14,15,  16,17,18,  19,20,21,  22,23,24,
            // row 2
            25,26,27,  28,29,30,  31,32,33,  34,35,36,
            // row 3
            37,38,39,  40,41,42,  43,44,45,  46,47,48,
        ];
        let out = patchify_u8(&rgb, 4, 4, 2, 1.0);
        // 4 patches × (2·2·3 = 12) = 48 elements.
        assert_eq!(out.len(), 48);

        // Patch (0, 0) = top-left 2×2 — rows 0-1, cols 0-1.
        //   (y=0,x=0): (1,2,3), (y=0,x=1): (4,5,6),
        //   (y=1,x=0): (13,14,15), (y=1,x=1): (16,17,18)
        assert_eq!(
            &out[0..12],
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0,]
        );

        // Patch (0, 1) = top-right: cols 2-3, rows 0-1.
        assert_eq!(
            &out[12..24],
            &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,]
        );
    }

    #[test]
    fn patchify_rescale_applied() {
        let rgb = vec![255u8; 4 * 4 * 3];
        let out = patchify_u8(&rgb, 4, 4, 2, 1.0 / 255.0);
        for v in &out {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn patches_to_tensor_shapes() {
        let rgb = vec![128u8; 32 * 32 * 3];
        let (pixels, pos) =
            patches_to_tensor(&rgb, 32, 32, 16, 1.0 / 255.0, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(pixels.dims(), &[4, 3 * 16 * 16]); // 2·2 patches
        assert_eq!(pos.dims(), &[4, 2]);

        // Positions: (0,0), (1,0), (0,1), (1,1).
        let pos_vec: Vec<i64> = pos.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(pos_vec, vec![0, 0, 1, 0, 0, 1, 1, 1]);
    }

    #[cfg(feature = "image-loading")]
    #[test]
    fn processor_pads_to_max_patches() {
        let dev = Device::Cpu;
        let cfg = Gemma4ImageProcessorConfig {
            patch_size: 16,
            max_soft_tokens: 70,
            pooling_kernel_size: 3,
            rescale_factor: 1.0 / 255.0,
        };
        let proc = Gemma4ImageProcessor::new(cfg, dev);

        // Small 96×96 image → after resize & patch will give a handful of
        // patches, but we expect padding up to `max_soft_tokens · pool_k² = 630`.
        let rgb = vec![100u8; 96 * 96 * 3];
        let out = proc.preprocess_rgb(96, 96, &rgb).unwrap();
        assert_eq!(out.pixel_values.dims(), &[630, 3 * 16 * 16]);
        assert_eq!(out.position_ids.dims(), &[630, 2]);

        // Padded rows are (-1, -1).
        let pos: Vec<i64> = out.position_ids.flatten_all().unwrap().to_vec1().unwrap();
        let last_two = &pos[pos.len() - 2..];
        assert_eq!(last_two, &[-1, -1]);
    }
}
