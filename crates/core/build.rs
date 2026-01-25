use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/paged_attention.cu");
    println!("cargo:rerun-if-changed=kernels/fp8_quant.cu");
    println!("cargo:rerun-if-changed=kernels/fp8_gemm.cu");
    println!("cargo:rerun-if-changed=kernels/gptq_dequant.cu");

    // Only compile CUDA kernels when the feature is enabled
    if std::env::var("CARGO_FEATURE_CUDA_KERNELS").is_err() {
        return;
    }

    // Detect GPU architecture (default to sm_89 for Ada Lovelace with FP8 support)
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_89".to_string());

    // Kernel definitions: (source, output, extra_flags)
    let kernels: Vec<(&str, &str, Vec<&str>)> = vec![
        (
            "kernels/paged_attention.cu",
            "kernels/paged_attention.ptx",
            vec![],
        ),
        (
            "kernels/fp8_quant.cu",
            "kernels/fp8_quant.ptx",
            vec![], // FP8 requires sm_89+ which we set as default
        ),
        (
            "kernels/fp8_gemm.cu",
            "kernels/fp8_gemm.ptx",
            vec![], // FP8 GEMM kernel
        ),
        (
            "kernels/gptq_dequant.cu",
            "kernels/gptq_dequant.ptx",
            vec![], // GPTQ dequantization kernel
        ),
    ];

    for (src_path, out_path, extra_flags) in kernels {
        let mut args = vec![
            "--ptx".to_string(),
            format!("-arch={arch}"),
            "-O3".to_string(),
            "--use_fast_math".to_string(),
            "-o".to_string(),
            out_path.to_string(),
            src_path.to_string(),
        ];

        for flag in extra_flags {
            args.push(flag.to_string());
        }

        let status = Command::new("nvcc").args(&args).status();

        match status {
            Ok(s) if s.success() => {
                println!("cargo:warning=Compiled {src_path} -> {out_path}");
            }
            Ok(s) => {
                panic!(
                    "nvcc failed for {src_path} with exit code: {s}. \
                     Ensure CUDA toolkit is installed."
                );
            }
            Err(e) => {
                panic!(
                    "Failed to run nvcc for {src_path}: {e}. \
                     Ensure CUDA toolkit is installed and nvcc is in PATH."
                );
            }
        }
    }
}
