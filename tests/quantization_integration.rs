//! Integration tests for quantization infrastructure.
//!
//! These tests verify quantization detection, config creation, weight loader
//! instantiation, and basic quantized linear layer operations. All CPU-only.

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use vllm_core::{
    config::ModelConfig,
    models::from_config_with_quant,
    quantization::{
        create_config, create_weight_loader_with_params, detect_from_directory,
        DetectedQuantConfig, NoQuantizationConfig, QuantizationConfig, QuantizationMethod,
    },
};

// ─── Detection tests ─────────────────────────────────────────────────────────

#[test]
fn test_detect_from_nonexistent_directory_returns_none() {
    let detected = detect_from_directory(std::path::Path::new("/nonexistent/path"));
    assert_eq!(detected.method, QuantizationMethod::None);
}

#[test]
fn test_detect_from_directory_with_no_quant_config() {
    // Create a temp dir with a minimal config.json that has no quantization
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("config.json");
    std::fs::write(
        &config_path,
        r#"{"architectures": ["LlamaForCausalLM"], "model_type": "llama"}"#,
    )
    .unwrap();

    let detected = detect_from_directory(dir.path());
    assert_eq!(detected.method, QuantizationMethod::None);
}

// ─── Config creation tests ───────────────────────────────────────────────────

#[test]
fn test_unquantized_detected_config_creates_no_quant_config() {
    let detected = DetectedQuantConfig::default();
    let config = create_config(&detected);
    assert_eq!(config.method(), QuantizationMethod::None);
}

#[test]
fn test_create_config_for_fp8() {
    let detected = DetectedQuantConfig {
        method: QuantizationMethod::Fp8,
        bits: Some(8),
        group_size: None,
        desc_act: None,
        activation_scheme: Some("dynamic".to_string()),
        raw_config: Default::default(),
    };
    let config = create_config(&detected);
    assert_eq!(config.method(), QuantizationMethod::Fp8);
}

#[test]
fn test_create_config_for_gptq() {
    let detected = DetectedQuantConfig {
        method: QuantizationMethod::Gptq,
        bits: Some(4),
        group_size: Some(128),
        desc_act: Some(true),
        activation_scheme: None,
        raw_config: Default::default(),
    };
    let config = create_config(&detected);
    assert_eq!(config.method(), QuantizationMethod::Gptq);
}

#[test]
fn test_create_config_for_awq() {
    let detected = DetectedQuantConfig {
        method: QuantizationMethod::Awq,
        bits: Some(4),
        group_size: Some(128),
        desc_act: None,
        activation_scheme: None,
        raw_config: Default::default(),
    };
    let config = create_config(&detected);
    assert_eq!(config.method(), QuantizationMethod::Awq);
}

// ─── Weight loader creation tests ────────────────────────────────────────────

#[test]
fn test_create_weight_loader_for_none() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let detected = DetectedQuantConfig::default();
    let loader = create_weight_loader_with_params(vb, &detected);
    assert_eq!(loader.method(), QuantizationMethod::None);
}

#[test]
fn test_create_weight_loader_for_fp8() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let detected = DetectedQuantConfig {
        method: QuantizationMethod::Fp8,
        bits: Some(8),
        group_size: None,
        desc_act: None,
        activation_scheme: Some("dynamic".to_string()),
        raw_config: Default::default(),
    };
    let loader = create_weight_loader_with_params(vb, &detected);
    assert_eq!(loader.method(), QuantizationMethod::Fp8);
}

#[test]
fn test_create_weight_loader_for_gptq() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let detected = DetectedQuantConfig {
        method: QuantizationMethod::Gptq,
        bits: Some(4),
        group_size: Some(128),
        desc_act: Some(false),
        activation_scheme: None,
        raw_config: Default::default(),
    };
    let loader = create_weight_loader_with_params(vb, &detected);
    assert_eq!(loader.method(), QuantizationMethod::Gptq);
}

#[test]
fn test_create_weight_loader_for_awq() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let detected = DetectedQuantConfig {
        method: QuantizationMethod::Awq,
        bits: Some(4),
        group_size: Some(128),
        desc_act: None,
        activation_scheme: None,
        raw_config: Default::default(),
    };
    let loader = create_weight_loader_with_params(vb, &detected);
    assert_eq!(loader.method(), QuantizationMethod::Awq);
}

// ─── from_config_with_quant fallthrough tests ────────────────────────────────

#[test]
fn test_unquantized_falls_through_to_regular_model() {
    let cfg = ModelConfig {
        architectures: vec!["LlamaForCausalLM".to_string()],
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
        attention_bias: None,
        extra: serde_json::Map::new(),
    };

    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let quant_config = DetectedQuantConfig::default();

    let model = from_config_with_quant(&cfg, vb, &quant_config);
    assert!(
        model.is_ok(),
        "unquantized fallthrough should produce a valid model: {:?}",
        model.err()
    );
}

// ─── QuantizedLinear trait tests ─────────────────────────────────────────────

#[test]
fn test_no_quantization_config_creates_linear() {
    let config: Box<dyn QuantizationConfig> = Box::new(NoQuantizationConfig::default());
    assert_eq!(config.method(), QuantizationMethod::None);

    let linear = config.create_linear(64, 128, true, &Device::Cpu).unwrap();
    assert_eq!(linear.in_features(), 64);
    assert_eq!(linear.out_features(), 128);
}

#[test]
fn test_no_quantization_linear_forward() {
    // Use F32 for CPU testing (BF16 matmul not supported on CPU)
    let config: Box<dyn QuantizationConfig> = Box::new(NoQuantizationConfig::new(DType::F32));
    let linear = config.create_linear(64, 128, false, &Device::Cpu).unwrap();

    let input = Tensor::zeros((4, 64), DType::F32, &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[4, 128]);
}

#[test]
fn test_unquantized_weight_loader_roundtrip() {
    let vb = VarBuilder::zeros(DType::F32, &Device::Cpu);
    let detected = DetectedQuantConfig::default();
    let loader = create_weight_loader_with_params(vb, &detected);

    let linear = loader.load_linear("test.layer", 32, 64, false).unwrap();
    assert_eq!(linear.in_features(), 32);
    assert_eq!(linear.out_features(), 64);

    let input = Tensor::zeros((8, 32), DType::F32, &Device::Cpu).unwrap();
    let output = linear.forward(&input).unwrap();
    assert_eq!(output.dims(), &[8, 64]);
}

// ─── Quantization method support checks ──────────────────────────────────────

#[test]
fn test_quantization_method_display() {
    assert_eq!(format!("{}", QuantizationMethod::None), "none");
    assert_eq!(format!("{}", QuantizationMethod::Fp8), "fp8");
    assert_eq!(format!("{}", QuantizationMethod::Gptq), "gptq");
    assert_eq!(format!("{}", QuantizationMethod::Awq), "awq");
    assert_eq!(format!("{}", QuantizationMethod::Gguf), "gguf");
}

#[test]
fn test_is_supported_gpu_capabilities() {
    use vllm_core::quantization::is_supported;

    // FP8 needs Hopper (90+)
    assert!(!is_supported(80, QuantizationMethod::Fp8));
    assert!(is_supported(90, QuantizationMethod::Fp8));

    // GPTQ works on Volta+
    assert!(is_supported(70, QuantizationMethod::Gptq));
    assert!(!is_supported(60, QuantizationMethod::Gptq));

    // Marlin needs Ampere+
    assert!(is_supported(80, QuantizationMethod::Marlin));
    assert!(!is_supported(70, QuantizationMethod::Marlin));

    // None works everywhere
    assert!(is_supported(0, QuantizationMethod::None));
}
