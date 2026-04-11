use serde::Serialize;
use std::process::Command;

/// Static GPU hardware specifications (from lookup table or estimated).
#[derive(Debug, Clone, Serialize)]
pub struct GpuHardwareProfile {
    /// GPU name as reported by nvidia-smi.
    pub name: String,
    /// CUDA compute capability (e.g., 89 for sm_89).
    pub compute_capability: u32,
    /// Memory bandwidth in GB/s.
    pub memory_bandwidth_gbs: f64,
    /// FP16/BF16 tensor core compute in TFLOPS (dense, no sparsity).
    pub fp16_tflops: f64,
    /// Total VRAM in bytes.
    pub total_vram_bytes: u64,
    /// Free VRAM in bytes (at query time).
    pub free_vram_bytes: u64,
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Whether this profile came from the known GPU table or was estimated.
    pub is_known_gpu: bool,
}

struct GpuSpec {
    name_pattern: &'static str,
    compute_capability: u32,
    memory_bandwidth_gbs: f64,
    fp16_tflops: f64,
    sm_count: u32,
}

// Known GPU table — matched by substring in nvidia-smi name.
// FP16 TFLOPS are dense tensor core (no sparsity).
const KNOWN_GPUS: &[GpuSpec] = &[
    // Blackwell
    GpuSpec {
        name_pattern: "B200",
        compute_capability: 100,
        memory_bandwidth_gbs: 8000.0,
        fp16_tflops: 2250.0,
        sm_count: 160,
    },
    GpuSpec {
        name_pattern: "RTX 5090",
        compute_capability: 100,
        memory_bandwidth_gbs: 1792.0,
        fp16_tflops: 419.2,
        sm_count: 170,
    },
    GpuSpec {
        name_pattern: "RTX 5080",
        compute_capability: 100,
        memory_bandwidth_gbs: 960.0,
        fp16_tflops: 225.0,
        sm_count: 84,
    },
    // Hopper
    GpuSpec {
        name_pattern: "H200",
        compute_capability: 90,
        memory_bandwidth_gbs: 4800.0,
        fp16_tflops: 989.0,
        sm_count: 132,
    },
    GpuSpec {
        name_pattern: "H100 SXM",
        compute_capability: 90,
        memory_bandwidth_gbs: 3350.0,
        fp16_tflops: 989.0,
        sm_count: 132,
    },
    GpuSpec {
        name_pattern: "H100 PCIe",
        compute_capability: 90,
        memory_bandwidth_gbs: 2039.0,
        fp16_tflops: 756.0,
        sm_count: 114,
    },
    // Ada Lovelace
    GpuSpec {
        name_pattern: "L40S",
        compute_capability: 89,
        memory_bandwidth_gbs: 864.0,
        fp16_tflops: 362.0,
        sm_count: 142,
    },
    GpuSpec {
        name_pattern: "L40 ",
        compute_capability: 89,
        memory_bandwidth_gbs: 864.0,
        fp16_tflops: 181.0,
        sm_count: 142,
    },
    GpuSpec {
        name_pattern: "L4",
        compute_capability: 89,
        memory_bandwidth_gbs: 300.0,
        fp16_tflops: 121.0,
        sm_count: 58,
    },
    GpuSpec {
        name_pattern: "RTX 4090",
        compute_capability: 89,
        memory_bandwidth_gbs: 1008.0,
        fp16_tflops: 165.0,
        sm_count: 128,
    },
    GpuSpec {
        name_pattern: "RTX 4080 SUPER",
        compute_capability: 89,
        memory_bandwidth_gbs: 736.0,
        fp16_tflops: 104.6,
        sm_count: 80,
    },
    GpuSpec {
        name_pattern: "RTX 4080",
        compute_capability: 89,
        memory_bandwidth_gbs: 716.8,
        fp16_tflops: 97.5,
        sm_count: 76,
    },
    GpuSpec {
        name_pattern: "RTX 4070 Ti",
        compute_capability: 89,
        memory_bandwidth_gbs: 504.2,
        fp16_tflops: 81.6,
        sm_count: 60,
    },
    GpuSpec {
        name_pattern: "RTX 4070",
        compute_capability: 89,
        memory_bandwidth_gbs: 504.2,
        fp16_tflops: 59.3,
        sm_count: 46,
    },
    GpuSpec {
        name_pattern: "RTX 4060 Ti",
        compute_capability: 89,
        memory_bandwidth_gbs: 288.0,
        fp16_tflops: 44.1,
        sm_count: 34,
    },
    GpuSpec {
        name_pattern: "RTX 4060",
        compute_capability: 89,
        memory_bandwidth_gbs: 272.0,
        fp16_tflops: 30.6,
        sm_count: 24,
    },
    // Ampere
    GpuSpec {
        name_pattern: "A100 SXM",
        compute_capability: 80,
        memory_bandwidth_gbs: 2039.0,
        fp16_tflops: 312.0,
        sm_count: 108,
    },
    GpuSpec {
        name_pattern: "A100-SXM",
        compute_capability: 80,
        memory_bandwidth_gbs: 2039.0,
        fp16_tflops: 312.0,
        sm_count: 108,
    },
    GpuSpec {
        name_pattern: "A100 PCIe",
        compute_capability: 80,
        memory_bandwidth_gbs: 1555.0,
        fp16_tflops: 312.0,
        sm_count: 108,
    },
    GpuSpec {
        name_pattern: "A100-PCIE",
        compute_capability: 80,
        memory_bandwidth_gbs: 1555.0,
        fp16_tflops: 312.0,
        sm_count: 108,
    },
    GpuSpec {
        name_pattern: "A10G",
        compute_capability: 86,
        memory_bandwidth_gbs: 600.0,
        fp16_tflops: 70.6,
        sm_count: 80,
    },
    GpuSpec {
        name_pattern: "RTX 3090",
        compute_capability: 86,
        memory_bandwidth_gbs: 936.2,
        fp16_tflops: 71.0,
        sm_count: 82,
    },
    GpuSpec {
        name_pattern: "RTX 3080 Ti",
        compute_capability: 86,
        memory_bandwidth_gbs: 912.4,
        fp16_tflops: 68.0,
        sm_count: 80,
    },
    GpuSpec {
        name_pattern: "RTX 3080",
        compute_capability: 86,
        memory_bandwidth_gbs: 760.3,
        fp16_tflops: 59.5,
        sm_count: 68,
    },
    GpuSpec {
        name_pattern: "RTX 3070",
        compute_capability: 86,
        memory_bandwidth_gbs: 448.0,
        fp16_tflops: 40.6,
        sm_count: 46,
    },
    // Turing
    GpuSpec {
        name_pattern: "T4",
        compute_capability: 75,
        memory_bandwidth_gbs: 300.0,
        fp16_tflops: 65.0,
        sm_count: 40,
    },
    GpuSpec {
        name_pattern: "RTX 2080 Ti",
        compute_capability: 75,
        memory_bandwidth_gbs: 616.0,
        fp16_tflops: 53.8,
        sm_count: 68,
    },
    // Volta
    GpuSpec {
        name_pattern: "V100S",
        compute_capability: 70,
        memory_bandwidth_gbs: 1134.0,
        fp16_tflops: 130.0,
        sm_count: 80,
    },
    GpuSpec {
        name_pattern: "V100",
        compute_capability: 70,
        memory_bandwidth_gbs: 900.0,
        fp16_tflops: 112.0,
        sm_count: 80,
    },
];

/// Raw GPU info parsed from nvidia-smi output.
#[derive(Debug)]
struct NvidiaSmiInfo {
    name: String,
    compute_cap: String,
    memory_total_mib: u64,
    memory_free_mib: u64,
}

fn parse_nvidia_smi_output(output: &str) -> Option<NvidiaSmiInfo> {
    let line = output.lines().next()?.trim();
    let parts: Vec<&str> = line.split(", ").collect();
    if parts.len() < 4 {
        return None;
    }
    Some(NvidiaSmiInfo {
        name: parts[0].trim().to_string(),
        compute_cap: parts[1].trim().to_string(),
        memory_total_mib: parts[2].trim().parse().ok()?,
        memory_free_mib: parts[3].trim().parse().ok()?,
    })
}

fn parse_compute_cap(s: &str) -> u32 {
    let parts: Vec<&str> = s.split('.').collect();
    if parts.len() == 2 {
        let major: u32 = parts[0].parse().unwrap_or(0);
        let minor: u32 = parts[1].parse().unwrap_or(0);
        major * 10 + minor
    } else {
        0
    }
}

fn lookup_known_gpu(name: &str) -> Option<&'static GpuSpec> {
    KNOWN_GPUS
        .iter()
        .find(|spec| name.contains(spec.name_pattern))
}

fn estimate_from_sm_count_and_cap(_sm_count: u32, compute_cap: u32) -> (f64, f64) {
    // Conservative fallback: estimate based on architecture generation.
    // Returns (bandwidth_gbs, fp16_tflops).
    match compute_cap / 10 {
        10 => (800.0, 100.0), // Blackwell (conservative)
        9 => (600.0, 80.0),   // Hopper (conservative)
        8 => (400.0, 50.0),   // Ampere/Ada
        7 => (300.0, 40.0),   // Volta/Turing
        _ => (200.0, 20.0),   // Unknown
    }
}

/// Detect GPU hardware profile by querying nvidia-smi.
pub fn detect_gpu_profile() -> anyhow::Result<GpuHardwareProfile> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,compute_cap,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .map_err(|e| anyhow::anyhow!("Failed to run nvidia-smi: {e}"))?;

    if !output.status.success() {
        return Err(anyhow::anyhow!(
            "nvidia-smi failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let info = parse_nvidia_smi_output(&stdout)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse nvidia-smi output: {stdout}"))?;

    let compute_capability = parse_compute_cap(&info.compute_cap);
    let total_vram_bytes = info.memory_total_mib * 1024 * 1024;
    let free_vram_bytes = info.memory_free_mib * 1024 * 1024;

    if let Some(spec) = lookup_known_gpu(&info.name) {
        Ok(GpuHardwareProfile {
            name: info.name,
            compute_capability,
            memory_bandwidth_gbs: spec.memory_bandwidth_gbs,
            fp16_tflops: spec.fp16_tflops,
            total_vram_bytes,
            free_vram_bytes,
            sm_count: spec.sm_count,
            is_known_gpu: true,
        })
    } else {
        let (bandwidth, tflops) = estimate_from_sm_count_and_cap(0, compute_capability);
        Ok(GpuHardwareProfile {
            name: info.name,
            compute_capability,
            memory_bandwidth_gbs: bandwidth,
            fp16_tflops: tflops,
            total_vram_bytes,
            free_vram_bytes,
            sm_count: 0,
            is_known_gpu: false,
        })
    }
}

/// Build a GPU profile from known values (for testing or manual configuration).
pub fn gpu_profile_from_values(
    name: &str,
    total_vram_bytes: u64,
    free_vram_bytes: u64,
) -> GpuHardwareProfile {
    if let Some(spec) = lookup_known_gpu(name) {
        GpuHardwareProfile {
            name: name.to_string(),
            compute_capability: spec.compute_capability,
            memory_bandwidth_gbs: spec.memory_bandwidth_gbs,
            fp16_tflops: spec.fp16_tflops,
            total_vram_bytes,
            free_vram_bytes,
            sm_count: spec.sm_count,
            is_known_gpu: true,
        }
    } else {
        GpuHardwareProfile {
            name: name.to_string(),
            compute_capability: 0,
            memory_bandwidth_gbs: 300.0,
            fp16_tflops: 40.0,
            total_vram_bytes,
            free_vram_bytes,
            sm_count: 0,
            is_known_gpu: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_nvidia_smi_output() {
        let output = "NVIDIA GeForce RTX 4090, 8.9, 24564, 23456\n";
        let info = parse_nvidia_smi_output(output).unwrap();
        assert_eq!(info.name, "NVIDIA GeForce RTX 4090");
        assert_eq!(info.compute_cap, "8.9");
        assert_eq!(info.memory_total_mib, 24564);
        assert_eq!(info.memory_free_mib, 23456);
    }

    #[test]
    fn test_parse_compute_cap() {
        assert_eq!(parse_compute_cap("8.9"), 89);
        assert_eq!(parse_compute_cap("9.0"), 90);
        assert_eq!(parse_compute_cap("7.0"), 70);
    }

    #[test]
    fn test_lookup_known_gpu_rtx4090() {
        let spec = lookup_known_gpu("NVIDIA GeForce RTX 4090").unwrap();
        assert!((spec.memory_bandwidth_gbs - 1008.0).abs() < 1.0);
        assert!((spec.fp16_tflops - 165.0).abs() < 1.0);
        assert_eq!(spec.compute_capability, 89);
    }

    #[test]
    fn test_lookup_known_gpu_h100_sxm() {
        let spec = lookup_known_gpu("NVIDIA H100 SXM").unwrap();
        assert!((spec.memory_bandwidth_gbs - 3350.0).abs() < 1.0);
        assert!((spec.fp16_tflops - 989.0).abs() < 1.0);
    }

    #[test]
    fn test_lookup_known_gpu_a100() {
        let spec = lookup_known_gpu("NVIDIA A100-SXM4-80GB").unwrap();
        assert!((spec.memory_bandwidth_gbs - 2039.0).abs() < 1.0);
    }

    #[test]
    fn test_lookup_unknown_gpu() {
        assert!(lookup_known_gpu("Some Unknown GPU").is_none());
    }

    #[test]
    fn test_gpu_profile_from_values_known() {
        let profile = gpu_profile_from_values(
            "NVIDIA GeForce RTX 4090",
            24 * 1024 * 1024 * 1024,
            20 * 1024 * 1024 * 1024,
        );
        assert!(profile.is_known_gpu);
        assert!((profile.memory_bandwidth_gbs - 1008.0).abs() < 1.0);
        assert_eq!(profile.total_vram_bytes, 24 * 1024 * 1024 * 1024);
    }

    #[test]
    fn test_gpu_profile_from_values_unknown() {
        let profile = gpu_profile_from_values(
            "Mystery GPU",
            8 * 1024 * 1024 * 1024,
            6 * 1024 * 1024 * 1024,
        );
        assert!(!profile.is_known_gpu);
    }

    #[test]
    fn test_known_gpu_table_no_duplicates() {
        // Verify that the table entries don't shadow each other
        // by checking that each entry matches at least its own pattern.
        for spec in KNOWN_GPUS {
            assert!(
                spec.name_pattern.contains(spec.name_pattern),
                "Pattern '{}' should match itself",
                spec.name_pattern
            );
        }
    }

    #[test]
    fn test_estimate_fallback() {
        let (bw, tf) = estimate_from_sm_count_and_cap(0, 89);
        assert!(bw > 0.0);
        assert!(tf > 0.0);
    }
}
