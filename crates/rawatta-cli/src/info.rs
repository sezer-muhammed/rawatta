// SPDX-License-Identifier: Apache-2.0
//! `rawatta info`: the GPU, its compute capability and memory, the driver's
//! CUDA version, whether RAWATTA and the exact engine can run, and whether
//! the runtime libraries and the weights are found.

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;

use anyhow::{Result, bail};
use rawatta_nn::Nn;
use rawatta_nn::gpu::{GpuInfo, probe};

use crate::support::{self, parse_cc, sm};

const MIB: f64 = 1024.0 * 1024.0;

/// The driver's CUDA API floor of this build (cudarc `cuda-12080`).
const MIN_DRIVER_CUDA: (i32, i32) = (12, 8);

pub fn yes_no(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}

fn device_lines(d: &GpuInfo) -> Vec<String> {
    let cc = parse_cc(&d.compute_capability).unwrap_or((0, 0));
    vec![
        format!("GPU {}: {}", d.ordinal, d.name),
        format!(
            "  Compute capability: {} ({})",
            d.compute_capability,
            sm(cc)
        ),
        format!(
            "  VRAM: {:.1} GiB ({:.0} MiB)",
            d.total_memory_bytes as f64 / MIB / 1024.0,
            d.total_memory_bytes as f64 / MIB
        ),
        format!("  Multiprocessors: {}", d.multiprocessors),
        format!(
            "  RAWATTA supported: {} (needs sm_89+)",
            yes_no(cc >= (8, 9))
        ),
        format!(
            "  Exact engine supported: {} (needs sm_80+){}",
            yes_no(cc >= (8, 0)),
            if ((8, 0)..(8, 9)).contains(&cc) {
                "; built and bit-checked for sm_80+, not yet run on Ampere hardware"
            } else {
                ""
            }
        ),
    ]
}

/// Opens device 0 and checks NVRTC (compiles an empty kernel) and cuBLASLt
/// (creates a handle). cudarc panics when a library is missing, so each
/// check runs under `catch_unwind`.
fn runtime_lines() -> Vec<String> {
    let nn = match catch_unwind(Nn::open) {
        Ok(Ok(nn)) => nn,
        Ok(Err(e)) => return vec![format!("Runtime: cannot open GPU 0: {e:#}")],
        Err(_) => return vec!["Runtime: cannot open GPU 0 (CUDA driver library failed)".into()],
    };
    let nvrtc = if unsafe { rawatta_nn::cudarc::nvrtc::sys::is_culib_present() } {
        match catch_unwind(AssertUnwindSafe(|| {
            nn.module(
                "rawatta_info_probe",
                "extern \"C\" __global__ void rawatta_info_probe() {}",
            )
        })) {
            Ok(Ok(_)) => format!("ok (target {})", nn.arch()),
            Ok(Err(e)) => format!("error: {e:#}"),
            Err(_) => "error: NVRTC failed to load".into(),
        }
    } else {
        "not found (install the CUDA 12.x NVRTC library, libnvrtc.so.12, 12.8 or a newer 12.x; CUDA 13 \
         libraries are not picked up; for a runfile install add its lib64 to LD_LIBRARY_PATH)"
            .into()
    };
    let cublas = match catch_unwind(AssertUnwindSafe(|| {
        rawatta_nn::gemm::Gemm::new(&nn, rawatta_nn::gemm::Tuning::Heuristic)
    })) {
        Ok(Ok(_)) => {
            let v = rawatta_nn::gemm::Gemm::version();
            format!("ok (version {}.{}.{})", v / 10000, v / 100 % 100, v % 100)
        }
        Ok(Err(e)) => format!("error: {e:#}"),
        Err(_) => "error: cuBLASLt failed to load".into(),
    };
    vec![format!("NVRTC: {nvrtc}"), format!("cuBLASLt: {cublas}")]
}

pub fn info(weights: Option<&Path>) -> Result<()> {
    println!("rawatta {}", env!("CARGO_PKG_VERSION"));
    let r = probe();
    match &r.driver_version {
        Some(v) => {
            let ok = parse_cc(v).is_some_and(|c| c >= MIN_DRIVER_CUDA);
            println!(
                "Driver CUDA version: {v}{}",
                if ok {
                    String::new()
                } else {
                    format!(
                        " (too old: rawatta needs a driver that supports CUDA {}.{} or newer)",
                        MIN_DRIVER_CUDA.0, MIN_DRIVER_CUDA.1
                    )
                }
            );
        }
        None => println!("Driver CUDA version: none (no NVIDIA driver found)"),
    }
    if r.devices.is_empty() {
        println!("RAWATTA supported: no (needs sm_89+)");
        println!("Exact engine supported: no (needs sm_80+)");
        bail!(
            "no CUDA GPU found{}",
            r.error.map(|e| format!(": {e}")).unwrap_or_default()
        );
    }
    for d in &r.devices {
        for l in device_lines(d) {
            println!("{l}");
        }
    }
    let cc0 = parse_cc(&r.devices[0].compute_capability).unwrap_or((0, 0));
    if r.devices.len() > 1 {
        println!(
            "rawatta runs on GPU 0 (select another with CUDA_VISIBLE_DEVICES): RAWATTA supported: {} (needs sm_89+)",
            yes_no(cc0 >= (8, 9))
        );
    }
    if cc0 >= (8, 0) {
        for l in runtime_lines() {
            println!("{l}");
        }
    }
    let w = support::weights_path(weights);
    println!(
        "Weights: {} ({})",
        w.display(),
        if w.is_file() {
            "found"
        } else {
            "not found: set RAWATTA_WEIGHTS or download them; `rawatta run` prints the commands"
        }
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_report_lines() {
        let d = GpuInfo {
            ordinal: 0,
            name: "Test GPU".into(),
            compute_capability: "8.6".into(),
            total_memory_bytes: 24 << 30,
            multiprocessors: 84,
        };
        let l = device_lines(&d).join("\n");
        assert!(l.contains("Compute capability: 8.6 (sm_86)"), "{l}");
        assert!(l.contains("VRAM: 24.0 GiB (24576 MiB)"), "{l}");
        assert!(l.contains("RAWATTA supported: no (needs sm_89+)"), "{l}");
        assert!(
            l.contains("Exact engine supported: yes (needs sm_80+)"),
            "{l}"
        );
        assert!(l.contains("not yet run on Ampere hardware"), "{l}");
        let ada = GpuInfo {
            compute_capability: "8.9".into(),
            ..d
        };
        let l = device_lines(&ada).join("\n");
        assert!(l.contains("RAWATTA supported: yes"), "{l}");
        assert!(!l.contains("Ampere"), "{l}");
    }
}
