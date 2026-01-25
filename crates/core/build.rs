use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels/paged_attention.cu");

    // Only compile CUDA kernels when the feature is enabled
    if std::env::var("CARGO_FEATURE_CUDA_KERNELS").is_err() {
        return;
    }

    let out_path = "kernels/paged_attention.ptx";
    let src_path = "kernels/paged_attention.cu";

    // Detect GPU architecture (default to sm_80 for broad Ampere+ support)
    let arch = std::env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_80".to_string());

    let status = Command::new("nvcc")
        .args([
            "--ptx",
            &format!("-arch={arch}"),
            "-O3",
            "--use_fast_math",
            "-o",
            out_path,
            src_path,
        ])
        .status();

    match status {
        Ok(s) if s.success() => {}
        Ok(s) => {
            panic!("nvcc failed with exit code: {s}. Ensure CUDA toolkit is installed.");
        }
        Err(e) => {
            panic!(
                "Failed to run nvcc: {e}. Ensure CUDA toolkit is installed and nvcc is in PATH."
            );
        }
    }
}
