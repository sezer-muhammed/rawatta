// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// Part of the native VGGT-Ω engine; see LICENSE-VGGT-OMEGA.
//! Every NVRTC module the engine can load, compiled with NVRTC for each
//! supported GPU family exactly as at run time and assembled with `ptxas`
//! for that GPU. The driver's JIT rejects a whole module when one
//! instruction is missing on the GPU (FP8 conversions below sm_89, for
//! example), and a GPU only shows that for its own architecture, so this is
//! how the sm_80+ claim of the exact engine is checked without the cards.

use std::path::{Path, PathBuf};
use std::process::Command;

use rawatta_nn::qkv::HEAD_DIM;
use rawatta_nn::sage::{self, Config, PvAccum, QkQuant};
use rayon::prelude::*;

/// Compute capability and `ptxas` target of each GPU family the engine
/// supports (sm_80+; sm_88 and other 8.x share compute_80's PTX).
const GPUS: [((i32, i32), &str); 7] = [
    ((8, 0), "sm_80"),   // A100, A30
    ((8, 6), "sm_86"),   // RTX 30 series, A10, A40, RTX A6000
    ((8, 7), "sm_87"),   // Jetson Orin
    ((8, 9), "sm_89"),   // RTX 40 series, L4, L40S
    ((9, 0), "sm_90"),   // H100, H200
    ((10, 0), "sm_100"), // B200
    ((12, 0), "sm_120"), // RTX 50 series
];

fn ptxas() -> PathBuf {
    for var in ["CUDA_HOME", "CUDA_PATH"] {
        if let Some(d) = std::env::var_os(var) {
            let p = PathBuf::from(d).join("bin").join("ptxas");
            if p.is_file() {
                return p;
            }
        }
    }
    if Command::new("ptxas")
        .arg("--version")
        .output()
        .is_ok_and(|o| o.status.success())
    {
        return PathBuf::from("ptxas");
    }
    let p = PathBuf::from("/usr/local/cuda/bin/ptxas");
    assert!(
        p.is_file(),
        "ptxas not found: install the CUDA toolkit (12.8+) or set CUDA_HOME"
    );
    p
}

/// Assembles `ptx` for `sm`; the error carries ptxas's message.
fn assemble(ptxas: &Path, dir: &Path, what: &str, sm: &str, ptx: &str) -> Result<(), String> {
    let file = dir.join(format!("{what}.{sm}.ptx"));
    std::fs::write(&file, ptx).map_err(|e| format!("write {}: {e}", file.display()))?;
    let out = Command::new(ptxas)
        .args(["-arch", sm, "-o"])
        .arg(dir.join(format!("{what}.{sm}.cubin")))
        .arg(&file)
        .output()
        .map_err(|e| format!("run ptxas: {e}"))?;
    if out.status.success() {
        Ok(())
    } else {
        let msg = String::from_utf8_lossy(&out.stderr);
        let lines: Vec<&str> = msg.lines().collect();
        Err(format!(
            "{what} for {sm}: {} error line(s), first: {}",
            lines.len(),
            lines.first().copied().unwrap_or("").trim()
        ))
    }
}

#[test]
#[ignore = "needs NVRTC and ptxas (CUDA 12.8+ toolkit, no GPU): run in the dev container"]
fn kernels_assemble_for_every_supported_gpu() {
    let ptxas = ptxas();
    let dir = std::env::temp_dir().join(format!("rawatta-ptxas-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();

    // The bf16 engine: rawatta-nn's ops and this crate's kernels, for every
    // family at the architecture `Nn` picks for it.
    let modules = crate::module_sources();
    let jobs: Vec<_> = modules
        .iter()
        .flat_map(|m| GPUS.iter().map(move |g| (m, *g)))
        .collect();
    let mut errors: Vec<String> = jobs
        .par_iter()
        .flat_map(|((name, src), (cc, sm))| {
            let arch = rawatta_nn::nvrtc_arch(*cc);
            let ptx = match rawatta_nn::compile_ptx(src, arch) {
                Ok(p) => p.to_src(),
                Err(e) => return vec![format!("NVRTC {name} for {arch}: {e:#}")],
            };
            let mut errs: Vec<String> = assemble(&ptxas, &dir, name, sm, &ptx)
                .err()
                .into_iter()
                .collect();
            // Below sm_89 no FP8 instruction may appear at all; on sm_89+
            // the LayerNorm keeps the hardware conversion.
            if *cc < (8, 9) && ptx.contains("e4m3") {
                errs.push(format!(
                    "{name} for {arch}: PTX uses an FP8 (e4m3) instruction"
                ));
            }
            if *cc >= (8, 9)
                && *name == "rawatta_nn_norm"
                && !ptx.contains("cvt.rn.satfinite.e4m3x2.f32")
            {
                errs.push(format!(
                    "{name} for {arch}: lost the hardware FP8 conversion"
                ));
            }
            errs
        })
        .collect();

    // The quantised attention (sm_89+ only, compiled for compute_89 and
    // JIT-compiled by the driver on newer GPUs): every P·V format, with the
    // default and the finest QK quantisation, and the peak probes.
    let finest = QkQuant {
        q_token: true,
        k_token: true,
        smooth_q: true,
    };
    let mut sage_jobs = vec![("sage_peak".to_string(), sage::PEAK_SOURCE.as_str(), vec![])];
    for pv in [PvAccum::F32, PvAccum::F16, PvAccum::Half, PvAccum::HalfF16] {
        for (tag, qk) in [("", QkQuant::default()), ("_fine", finest)] {
            let c = Config::with_precision(HEAD_DIM, pv, qk);
            let d = sage::defines(HEAD_DIM, c, &[]).unwrap();
            sage_jobs.push((format!("sage_{pv:?}{tag}"), sage::SOURCE.as_str(), d));
        }
    }
    let sage_sms: Vec<&str> = GPUS
        .iter()
        .filter(|(cc, _)| *cc >= (8, 9))
        .map(|(_, sm)| *sm)
        .collect();
    errors.extend(
        sage_jobs
            .par_iter()
            .flat_map(|(name, src, defines)| match sage::compile(src, defines) {
                Ok(p) => {
                    let ptx = p.to_src();
                    sage_sms
                        .iter()
                        .filter_map(|sm| assemble(&ptxas, &dir, name, sm, &ptx).err())
                        .collect::<Vec<_>>()
                }
                Err(e) => vec![format!("NVRTC {name}: {e:#}")],
            })
            .collect::<Vec<_>>(),
    );
    let _ = std::fs::remove_dir_all(&dir);

    println!(
        "ptxas: {} engine modules x {} GPU targets, {} sage variants x {} targets, {} failures",
        modules.len(),
        GPUS.len(),
        sage_jobs.len(),
        sage_sms.len(),
        errors.len()
    );
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}
