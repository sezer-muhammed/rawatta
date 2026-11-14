// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! The benchmark cell: runs one or more engine configs on a set of image
//! sequences and writes predictions plus timing.
//!
//! A [`Job`] is JSON `{"out": dir, "configs": [names], "reps": k,
//! "warmup": w, "depth": bool, "crop_aspect": h/w (optional),
//! "sequences": [{"name": s, "images": [paths]}]}`; `crop_aspect`
//! centre-crops every frame to that height/width ratio first (e.g. 0.65625
//! gives 512x336 from 4:3 frames; intrinsics then refer to the crop).
//!
//! Output per (config, sequence) in `<out>/<config>/<name>/`:
//! `extrinsics.npy` (N,3,4) w2c, `intrinsics.npy` (N,3,3) in preprocessed
//! pixels, `depth.npy` (N,H,W) and `conf.npy` when `depth`, and
//! `cell.json` (per-run CUDA-event times of the whole forward, median, IQR,
//! stage split, memory, geometry). `/` in a config name becomes `%` in the
//! directory name. Latency excludes image decode and preprocessing; it
//! includes the trunk, the camera head and the dense head.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use rawatta_nn::{CancelFlag, Nn};
use serde::Deserialize;

use super::configs::BenchConfig;
use crate::image::Rgb8Image;
use crate::output::write_npy;
use crate::{VggtConfig, VggtEngine, preprocess};

#[derive(Debug, Clone, Deserialize)]
pub struct Job {
    pub out: PathBuf,
    pub configs: Vec<String>,
    /// Timed runs per cell; 0 runs once, untimed.
    #[serde(default)]
    pub reps: usize,
    #[serde(default = "two")]
    pub warmup: usize,
    #[serde(default = "yes")]
    pub depth: bool,
    #[serde(default)]
    pub crop_aspect: Option<f64>,
    pub sequences: Vec<Seq>,
}

fn two() -> usize {
    2
}
fn yes() -> bool {
    true
}

#[derive(Debug, Clone, Deserialize)]
pub struct Seq {
    pub name: String,
    pub images: Vec<PathBuf>,
}

impl Job {
    pub fn from_file(path: &Path) -> Result<Self> {
        let text =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
    }
}

/// Centre crop of `img` to height/width = `aspect` (only ever shrinks).
pub fn crop_to(img: Rgb8Image, aspect: f64) -> Rgb8Image {
    let (w, h) = (img.width as usize, img.height as usize);
    let (cw, ch) = if (h as f64) > w as f64 * aspect {
        (w, (w as f64 * aspect).round() as usize)
    } else {
        ((h as f64 / aspect).round() as usize, h)
    };
    if (cw, ch) == (w, h) {
        return img;
    }
    let (x0, y0) = ((w - cw) / 2, (h - ch) / 2);
    let mut data = Vec::with_capacity(cw * ch * 3);
    for y in y0..y0 + ch {
        data.extend_from_slice(&img.data[(y * w + x0) * 3..(y * w + x0 + cw) * 3]);
    }
    Rgb8Image {
        width: cw as u32,
        height: ch as u32,
        data,
    }
}

/// `(free, total)` device memory in bytes (`cuMemGetInfo`).
pub fn mem_info() -> Result<(usize, usize)> {
    rawatta_nn::cudarc::driver::result::mem_get_info().map_err(|e| anyhow!("cuMemGetInfo: {e:?}"))
}

/// Linear-interpolated quantile of an ascending, non-empty slice.
pub fn quantile(sorted: &[f32], q: f64) -> f32 {
    let x = q * (sorted.len() - 1) as f64;
    let (i, f) = (x.floor() as usize, x.fract() as f32);
    let j = (i + 1).min(sorted.len() - 1);
    sorted[i] * (1.0 - f) + sorted[j] * f
}

fn write_json(dir: &Path, j: &serde_json::Value) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let p = dir.join("cell.json");
    std::fs::write(&p, serde_json::to_vec_pretty(j)?)
        .with_context(|| format!("write {}", p.display()))
}

/// Runs every (config, sequence) cell of `job` with the weights at
/// `weights`. A cell that does not fit or fails records the error in its
/// `cell.json` and the job goes on.
pub fn run_job(job: &Job, weights: &Path) -> Result<()> {
    let _ctx = Nn::open()?;
    let (free_start, total) = mem_info()?;
    let cancel = CancelFlag::new();
    for name in &job.configs {
        let bc = BenchConfig::named(name)?;
        // Measured from the start of the process: an engine's memory is only
        // partly returned to the driver when it drops.
        let free0 = free_start;
        let mut cfg = VggtConfig::default();
        bc.apply(&mut cfg);
        let mut e = VggtEngine::load(Arc::new(Nn::open()?), weights, cfg).context("load engine")?;
        let after_load = mem_info()?.0;
        for s in &job.sequences {
            let frames = s
                .images
                .iter()
                .map(|f| -> Result<Rgb8Image> {
                    let bytes =
                        std::fs::read(f).with_context(|| format!("read {}", f.display()))?;
                    let img = preprocess::decode(&bytes)
                        .with_context(|| format!("decode {}", f.display()))?;
                    Ok(match job.crop_aspect {
                        Some(a) => crop_to(img, a),
                        None => img,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let p = e.preprocess(&frames)?;
            let g = p.geom;
            // `/` separates layer lists in config names; keep one directory.
            let dir = job.out.join(name.replace('/', "%")).join(&s.name);
            let plan = match e.plan(g) {
                Ok(p) => p,
                Err(err) => {
                    // Does not fit: record the planner's need and go on.
                    let j = serde_json::json!({"config": name, "sequence": s.name, "frames": g.frames,
                        "width": g.width, "height": g.height, "error": format!("{err:#}"),
                        "weights_mib": (free0.saturating_sub(after_load)) as f64 / 1048576.0});
                    write_json(&dir, &j)?;
                    println!(
                        "bench cell: {name} {} N={} DOES NOT FIT: {err:#}",
                        s.name, g.frames
                    );
                    continue;
                }
            };
            let mut times = Vec::new();
            let mut stages = Vec::new();
            let mut min_free = usize::MAX;
            let mut cams = None;
            let runs = if job.reps == 0 {
                1
            } else {
                job.reps + job.warmup
            };
            let mut failed = None;
            for k in 0..runs {
                match e.run(&p, &cancel, &mut |_, _| {}) {
                    Ok(c) => cams = Some(c),
                    Err(err) => {
                        failed = Some(format!("{err:#}"));
                        break;
                    }
                }
                min_free = min_free.min(mem_info()?.0);
                if job.reps == 0 || k >= job.warmup {
                    times.push(e.times.total_ms);
                    stages.push(e.times.clone());
                }
            }
            let cams = match (failed, cams) {
                (None, Some(c)) => c,
                (err, _) => {
                    let err = err.unwrap_or_else(|| "no run".into());
                    let j = serde_json::json!({"config": name, "sequence": s.name, "frames": g.frames, "error": err});
                    write_json(&dir, &j)?;
                    println!("bench cell: {name} {} N={} FAILED: {err}", s.name, g.frames);
                    continue;
                }
            };
            let ex: Vec<f32> = cams.extrinsics.iter().flatten().copied().collect();
            write_npy(&dir.join("extrinsics.npy"), &[g.frames, 3, 4], &ex)?;
            let k: Vec<f32> = cams.intrinsics.iter().flatten().copied().collect();
            write_npy(&dir.join("intrinsics.npy"), &[g.frames, 3, 3], &k)?;
            if job.depth {
                let d = e.download(&g)?;
                write_npy(
                    &dir.join("depth.npy"),
                    &[g.frames, g.height, g.width],
                    &d.depth,
                )?;
                write_npy(
                    &dir.join("conf.npy"),
                    &[g.frames, g.height, g.width],
                    &d.conf,
                )?;
            }
            let mut sorted = times.clone();
            sorted.sort_by(f32::total_cmp);
            let med = quantile(&sorted, 0.5);
            let j = serde_json::json!({
                "config": name,
                "layers": bc.layers(),
                "fp8_blocks": format!("{:#x}", bc.fp8_blocks.0),
                "sage_pv16": bc.sage_pv16,
                "sage_prec": bc.prec(),
                "sequence": s.name,
                "frames": g.frames,
                "width": g.width,
                "height": g.height,
                "timed": job.reps > 0,
                "times_ms": times,
                "median_ms": med,
                "iqr_ms": quantile(&sorted, 0.75) - quantile(&sorted, 0.25),
                "stages_at_last": stages.last(),
                "engine_peak_mib": (free0.saturating_sub(min_free)) as f64 / 1048576.0,
                "weights_mib": (free0.saturating_sub(after_load)) as f64 / 1048576.0,
                "card_used_peak_mib": (total - min_free) as f64 / 1048576.0,
                "card_total_mib": total as f64 / 1048576.0,
                "plan": {"offload": plan.offload, "release_trunk": plan.release_trunk, "dense_chunk": plan.dense_chunk},
            });
            write_json(&dir, &j)?;
            println!(
                "bench cell: {name} {} N={} {}x{} median {med:.1} ms",
                s.name, g.frames, g.width, g.height
            );
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn job_defaults_and_crop() {
        let j: Job = serde_json::from_str(
            r#"{"out": "o", "configs": ["exact"], "sequences": [{"name": "s", "images": ["a.jpg"]}]}"#,
        )
        .unwrap();
        assert_eq!(
            (j.reps, j.warmup, j.depth, j.crop_aspect),
            (0, 2, true, None)
        );
        let img = Rgb8Image::new(640, 480);
        let c = crop_to(img.clone(), 0.65625);
        assert_eq!((c.width, c.height), (640, 420));
        assert_eq!(crop_to(img, 0.75).width, 640);
        assert_eq!(quantile(&[1.0, 2.0, 3.0, 4.0], 0.5), 2.5);
    }

    /// `$RAWATTA_BENCH_JOB` (a job file) with the weights of
    /// [`crate::default_weights_path`].
    #[test]
    #[ignore]
    fn gpu_bench_cell() {
        let path = std::env::var_os("RAWATTA_BENCH_JOB")
            .expect("set RAWATTA_BENCH_JOB=<job.json> (see bench::cell)");
        let job = Job::from_file(Path::new(&path)).unwrap();
        run_job(&job, &crate::default_weights_path()).unwrap();
    }
}
