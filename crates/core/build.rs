use std::process::Command;

/// Minimum SM version required for each kernel category.
///
/// - sm_75 (Turing, T4): basic integer ops, no bf16
/// - sm_80 (Ampere, A100): native bf16, tf32
/// - sm_89 (Ada, L4/RTX 4090): native fp8 (e4m3/e5m2)
/// - sm_90 (Hopper, H100): fp8 tensor cores, TMA
struct KernelDef {
    source: &'static str,
    output: &'static str,
    min_sm: u32,
}

const KERNELS: &[KernelDef] = &[
    // BF16 attention — requires sm_80+ for __nv_bfloat16 arithmetic
    KernelDef {
        source: "kernels/paged_attention.cu",
        output: "kernels/paged_attention.ptx",
        min_sm: 80,
    },
    // FP8 kernels — require sm_89+ for native FP8 types
    KernelDef {
        source: "kernels/fp8_quant.cu",
        output: "kernels/fp8_quant.ptx",
        min_sm: 89,
    },
    KernelDef {
        source: "kernels/fp8_gemm.cu",
        output: "kernels/fp8_gemm.ptx",
        min_sm: 89,
    },
    // GPTQ uses integer arithmetic — works on sm_75+
    KernelDef {
        source: "kernels/gptq_dequant.cu",
        output: "kernels/gptq_dequant.ptx",
        min_sm: 75,
    },
    // MoE kernels use bf16
    KernelDef {
        source: "kernels/fused_moe_align.cu",
        output: "kernels/fused_moe_align.ptx",
        min_sm: 75,
    },
    KernelDef {
        source: "kernels/fused_moe_gemm.cu",
        output: "kernels/fused_moe_gemm.ptx",
        min_sm: 80,
    },
    // SwiGLU supports bf16/fp16/fp32
    KernelDef {
        source: "kernels/swiglu.cu",
        output: "kernels/swiglu.ptx",
        min_sm: 80,
    },
    // Top-k softmax — f32 arithmetic, but uses bf16 variants
    KernelDef {
        source: "kernels/topk_softmax.cu",
        output: "kernels/topk_softmax.ptx",
        min_sm: 80,
    },
    // BitsAndBytes — uses bf16
    KernelDef {
        source: "kernels/bnb_fused_matmul.cu",
        output: "kernels/bnb_fused_matmul.ptx",
        min_sm: 80,
    },
    // RMSNorm/LayerNorm — uses bf16/fp16/fp32
    KernelDef {
        source: "kernels/layernorm.cu",
        output: "kernels/layernorm.ptx",
        min_sm: 80,
    },
    // Fused RoPE — uses bf16/fp16/fp32
    KernelDef {
        source: "kernels/rope.cu",
        output: "kernels/rope.ptx",
        min_sm: 80,
    },
    // GELU/GeGLU/SiLU activations — uses bf16/fp16/fp32
    KernelDef {
        source: "kernels/activations.cu",
        output: "kernels/activations.ptx",
        min_sm: 80,
    },
    // Cache reshape/copy operations — uses bf16/fp16
    KernelDef {
        source: "kernels/cache_ops.cu",
        output: "kernels/cache_ops.ptx",
        min_sm: 80,
    },
    // GPU-side sampling (argmax, top-k/top-p, softmax) — uses bf16
    KernelDef {
        source: "kernels/sampling.cu",
        output: "kernels/sampling.ptx",
        min_sm: 80,
    },
    // Marlin fused dequant+GEMM for GPTQ/AWQ INT4/INT8 — uses bf16
    KernelDef {
        source: "kernels/marlin_gemm.cu",
        output: "kernels/marlin_gemm.ptx",
        min_sm: 80,
    },
    // Fused LayerNorm/RMSNorm + FP8 quantization — requires sm_89+ for FP8
    KernelDef {
        source: "kernels/layernorm_quant.cu",
        output: "kernels/layernorm_quant.ptx",
        min_sm: 89,
    },
    // Fused QK-RMSNorm + RoPE — uses bf16/fp16
    KernelDef {
        source: "kernels/qknorm_rope.cu",
        output: "kernels/qknorm_rope.ptx",
        min_sm: 80,
    },
    // Custom all-reduce for multi-GPU tensor parallelism — uses bf16/f32
    // P2P kernel that is 2-3x faster than NCCL for small tensors (< 8MB)
    KernelDef {
        source: "kernels/custom_allreduce.cu",
        output: "kernels/custom_allreduce.ptx",
        min_sm: 80,
    },
];

fn parse_sm_version(arch: &str) -> u32 {
    // Parse "sm_XX" or "compute_XX" format
    arch.trim_start_matches("sm_")
        .trim_start_matches("compute_")
        .parse::<u32>()
        .unwrap_or(89)
}

fn detect_gpu_arch() -> Option<String> {
    // Try nvidia-smi to detect the GPU compute capability
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader,nounits"])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let cap = String::from_utf8_lossy(&output.stdout)
        .lines()
        .next()?
        .trim()
        .to_string();

    // Convert "8.9" -> "sm_89", "9.0" -> "sm_90"
    let parts: Vec<&str> = cap.split('.').collect();
    if parts.len() == 2 {
        if let (Ok(major), Ok(minor)) = (parts[0].parse::<u32>(), parts[1].parse::<u32>()) {
            return Some(format!("sm_{}{}", major, minor));
        }
    }

    None
}

fn main() {
    // Register rerun triggers for all kernel sources.
    // Also track kernels that may be added later (build.rs itself triggers rerun).
    for kernel in KERNELS {
        println!("cargo:rerun-if-changed={}", kernel.source);
    }
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=CUDA_MULTI_ARCH");

    // Declare custom cfg names so rustc knows they are intentional.
    // These are emitted unconditionally; the values are set conditionally below.
    println!("cargo::rustc-check-cfg=cfg(cuda_hopper_fp8)");
    println!("cargo::rustc-check-cfg=cfg(cuda_ampere_fp8)");

    if std::env::var("CARGO_FEATURE_CUDA_KERNELS").is_err() {
        return;
    }

    // Determine target architecture:
    // 1. CUDA_ARCH env var (explicit override)
    // 2. Auto-detect from installed GPU via nvidia-smi
    // 3. Default: sm_80 (Ampere — broadest useful compatibility)
    let target_arch = std::env::var("CUDA_ARCH").ok().or_else(detect_gpu_arch);
    let target_sm = target_arch.as_deref().map(parse_sm_version).unwrap_or(80);

    let arch_str = format!("sm_{target_sm}");
    println!("cargo:warning=CUDA target architecture: {arch_str}");

    // Export the detected SM version so Rust code can query it at build time
    println!("cargo:rustc-env=CUDA_TARGET_SM={target_sm}");

    // Emit cfg flags for compile-time FP8 dispatch:
    //   cuda_hopper_fp8  — sm_89+ (Ada Lovelace / Hopper): use native hardware FP8 GEMM
    //   cuda_ampere_fp8  — sm_80..88 (Ampere): use software-decode FP8 kernel
    if target_sm >= 89 {
        println!("cargo:rustc-cfg=cuda_hopper_fp8");
    } else if target_sm >= 80 {
        println!("cargo:rustc-cfg=cuda_ampere_fp8");
    }

    let mut compiled = 0;
    let mut skipped = 0;

    for kernel in KERNELS {
        if target_sm < kernel.min_sm {
            println!(
                "cargo:warning=Skipping {} (requires sm_{}, target is sm_{})",
                kernel.source, kernel.min_sm, target_sm
            );
            skipped += 1;
            continue;
        }

        // Compile at the target arch (or kernel minimum, whichever is higher).
        // PTX is forward-compatible: sm_80 PTX will JIT to sm_89+ at load time.
        let compile_sm = std::cmp::max(target_sm, kernel.min_sm);
        let compile_arch = format!("sm_{compile_sm}");

        let args = vec![
            "--ptx".to_string(),
            format!("-arch={compile_arch}"),
            "-O3".to_string(),
            "--use_fast_math".to_string(),
            "-o".to_string(),
            kernel.output.to_string(),
            kernel.source.to_string(),
        ];

        let status = Command::new("nvcc").args(&args).status();

        match status {
            Ok(s) if s.success() => {
                println!(
                    "cargo:warning=Compiled {} -> {} ({})",
                    kernel.source, kernel.output, compile_arch
                );
                compiled += 1;
            }
            Ok(s) => {
                panic!(
                    "nvcc failed for {} with exit code: {}. \
                     Ensure CUDA toolkit is installed.",
                    kernel.source, s
                );
            }
            Err(e) => {
                panic!(
                    "Failed to run nvcc for {}: {}. \
                     Ensure CUDA toolkit is installed and nvcc is in PATH.",
                    kernel.source, e
                );
            }
        }
    }

    println!("cargo:warning=CUDA kernels: {compiled} compiled, {skipped} skipped (sm_{target_sm})");
}
