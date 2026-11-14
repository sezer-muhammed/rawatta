// SPDX-License-Identifier: Apache-2.0
//! `rawatta run`: frames in, cameras / depth / point cloud / COLMAP model
//! and timing out.

use std::cell::Cell;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result, anyhow, ensure};
use clap::{ArgGroup, Args};
use rawatta_nn::{CancelFlag, Nn};
use rayon::prelude::*;
use vggt_omega::bench::cell::{crop_to, mem_info, quantile};
use vggt_omega::output::{
    cameras_doc, colmap_text, point_cloud, sample_indices, write_npy, write_points_ply,
};
use vggt_omega::{Prepared, Rgb8Image, VggtConfig, VggtEngine, preprocess};

use crate::support::{
    DEFAULT_CONFIG, Resolved, check_compute, check_weights, gpu0, parse_aspect, resolve_config,
    weights_path,
};
use crate::video::{TempDir, extract_frames};

/// Frames taken from a video when `--max-frames` is not given.
pub const VIDEO_DEFAULT_FRAMES: usize = 64;

const IMAGE_EXTS: [&str; 4] = ["jpg", "jpeg", "png", "webp"];

#[derive(Args, Debug)]
#[command(group(ArgGroup::new("input").required(true).args(["images", "video"])))]
pub struct RunArgs {
    /// Folder of photos (.jpg, .jpeg, .png, .webp), used in file-name order.
    /// The first frame is the reference: world coordinates are its camera.
    #[arg(long, value_name = "DIR")]
    pub images: Option<PathBuf>,
    /// A video file; frames are sampled uniformly (needs ffmpeg and ffprobe
    /// on PATH).
    #[arg(long, value_name = "FILE")]
    pub video: Option<PathBuf>,
    /// Use at most N frames, sampled uniformly (first and last kept).
    /// Default: every image of --images, 64 frames of --video.
    #[arg(long, value_name = "N", value_parser = clap::value_parser!(u32).range(1..))]
    pub max_frames: Option<u32>,
    /// rawatta, rawatta-fast, exact, a config grammar string, or the path
    /// of a .toml config file in the configs/ format (see
    /// configs/GRAMMAR.md).
    #[arg(long, default_value = DEFAULT_CONFIG)]
    pub config: String,
    /// Converted weights (.safetensors). Default: $RAWATTA_WEIGHTS, else
    /// $XDG_CACHE_HOME/rawatta/vggt_omega_1b_512.safetensors
    /// (~/.cache/rawatta/... without XDG_CACHE_HOME).
    #[arg(long, value_name = "FILE")]
    pub weights: Option<PathBuf>,
    /// Output folder (created if needed; files in it are overwritten).
    #[arg(long, value_name = "DIR")]
    pub out: PathBuf,
    /// Centre-crop every frame to this height/width ratio first, e.g.
    /// 0.65625 or 336/512 (intrinsics then refer to the crop).
    #[arg(long, value_name = "H/W", value_parser = parse_aspect)]
    pub crop_aspect: Option<f64>,
    /// Do not write depth.npy and conf.npy.
    #[arg(long)]
    pub no_depth: bool,
    /// At most this many points in points.ply and points3D.txt (thinned
    /// with a regular stride).
    #[arg(long, value_name = "N", default_value_t = 2_000_000)]
    pub max_points: usize,
    /// Keep points whose depth confidence is at or above this percentile
    /// of all pixels (0 keeps every pixel).
    #[arg(long, value_name = "P", default_value_t = 50.0)]
    pub conf_percentile: f32,
    /// Timed forward passes; timing.json reports their median and IQR. The
    /// default, 1, is a single sample (IQR 0). For timings you report, use
    /// the paper's protocol: --reps 5 --warmup 2 or more (median of 5 to 10
    /// passes after 2 warm-ups), on an otherwise idle GPU.
    #[arg(long, value_name = "K", default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    pub reps: u32,
    /// Untimed forward passes before the timed ones (the first pass also
    /// compiles kernels and captures the CUDA graph).
    #[arg(long, value_name = "K", default_value_t = 1)]
    pub warmup: u32,
}

fn is_image(p: &Path) -> bool {
    p.is_file()
        && p.extension()
            .and_then(|e| e.to_str())
            .is_some_and(|e| IMAGE_EXTS.contains(&e.to_ascii_lowercase().as_str()))
}

/// The photos of `dir`, sorted by file name.
pub fn list_images(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("read image folder {}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| is_image(p))
        .collect();
    files.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    ensure!(
        !files.is_empty(),
        "no .jpg, .jpeg, .png or .webp files in {}",
        dir.display()
    );
    Ok(files)
}

fn file_name(p: &Path) -> String {
    p.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Decodes one image, centre-cropped to `crop` when given.
fn load_frame(f: &Path, crop: Option<f64>) -> Result<Rgb8Image> {
    let bytes = std::fs::read(f).with_context(|| format!("read {}", f.display()))?;
    let img = preprocess::decode(&bytes).with_context(|| format!("decode {}", f.display()))?;
    Ok(match crop {
        Some(a) => crop_to(img, a),
        None => img,
    })
}

/// The input frames as files, in order, sampled to `--max-frames`.
struct Inputs {
    names: Vec<String>,
    files: Vec<PathBuf>,
    /// Frames extracted from `--video`, removed on drop.
    _tmp: Option<TempDir>,
}

fn inputs(a: &RunArgs) -> Result<Inputs> {
    if let Some(dir) = &a.images {
        let files = list_images(dir)?;
        let n = a.max_frames.map_or(files.len(), |m| m as usize);
        let files: Vec<PathBuf> = sample_indices(files.len(), n)
            .into_iter()
            .map(|i| files[i].clone())
            .collect();
        let names = files.iter().map(|f| file_name(f)).collect();
        return Ok(Inputs {
            names,
            files,
            _tmp: None,
        });
    }
    let video = a
        .video
        .as_ref()
        .ok_or_else(|| anyhow!("give --images <dir> or --video <file>"))?;
    let tmp = TempDir::new("frames")?;
    let n = a.max_frames.map_or(VIDEO_DEFAULT_FRAMES, |m| m as usize);
    let (names, files) = extract_frames(video, n, &tmp.0)?.into_iter().unzip();
    Ok(Inputs {
        names,
        files,
        _tmp: Some(tmp),
    })
}

/// The model-input size, from the first frame alone (every frame is
/// brought to it), so the memory plan can be checked before the others are
/// decoded.
fn target(inp: &Inputs, crop: Option<f64>, vc: &VggtConfig) -> Result<preprocess::Target> {
    let first = load_frame(&inp.files[0], crop)?;
    preprocess::Target::new(&first, vc.resolution, vc.resize)
}

/// Decodes, crops and resizes every frame, one image per worker thread at
/// a time (full-size frames are never all in memory).
fn prepare_frames(inp: &Inputs, crop: Option<f64>, t: &preprocess::Target) -> Result<Prepared> {
    preprocess::prepare_with(t, inp.files.len(), |i| load_frame(&inp.files[i], crop))
}

/// File names for the preprocessed frames under `colmap/images/`: the input
/// stem as `.png` (whitespace → `_`, as the COLMAP writer does), prefixed
/// with the frame number when two stems collide.
pub fn colmap_image_names(names: &[String]) -> Vec<String> {
    let base: Vec<String> = names
        .iter()
        .map(|n| {
            let stem = Path::new(n)
                .file_stem()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| n.clone());
            format!("{}.png", stem.replace(char::is_whitespace, "_"))
        })
        .collect();
    let mut seen = std::collections::HashSet::new();
    if base.iter().all(|b| seen.insert(b.as_str())) {
        return base;
    }
    base.iter()
        .enumerate()
        .map(|(i, b)| format!("{i:05}_{b}"))
        .collect()
}

fn write_file(path: &Path, bytes: &[u8]) -> Result<()> {
    std::fs::write(path, bytes).with_context(|| format!("write {}", path.display()))
}

fn write_colmap_images(dir: &Path, names: &[String], p: &Prepared) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("create {}", dir.display()))?;
    let g = p.geom;
    let n = g.pixels() * 3;
    names.par_iter().enumerate().try_for_each(|(i, name)| {
        let path = dir.join(name);
        image::save_buffer_with_format(
            &path,
            &p.pixels[i * n..(i + 1) * n],
            g.width as u32,
            g.height as u32,
            image::ExtendedColorType::Rgb8,
            image::ImageFormat::Png,
        )
        .with_context(|| format!("write {}", path.display()))
    })
}

/// Cheap checks of the input path, run before the weights and the GPU:
/// the image folder exists and holds photos, or the video is a file.
pub fn check_input(a: &RunArgs) -> Result<()> {
    if let Some(dir) = &a.images {
        ensure!(
            dir.is_dir(),
            "image folder {} not found (--images takes a folder of .jpg, .jpeg, .png or .webp files)",
            dir.display()
        );
        list_images(dir)?;
    } else if let Some(v) = &a.video {
        ensure!(v.is_file(), "video file {} not found", v.display());
    }
    Ok(())
}

pub fn run(a: &RunArgs) -> Result<()> {
    let started = Instant::now();
    let cfg: Resolved = resolve_config(&a.config)?;
    ensure!(
        (0.0..=100.0).contains(&a.conf_percentile),
        "--conf-percentile must be within [0, 100]"
    );
    check_input(a)?;
    let weights = weights_path(a.weights.as_deref());
    check_weights(&weights, cfg.needs_fp8_weights())?;
    let (gpu, cc) = gpu0()?;
    check_compute(&cfg, cc, &gpu)?;

    let mut vc = VggtConfig::default();
    cfg.config.apply(&mut vc);
    let t = Instant::now();
    let inp = inputs(a)?;
    let names = &inp.names;
    let target = target(&inp, a.crop_aspect, &vc)?;
    let g = target.geometry(inp.files.len());
    let inputs_s = t.elapsed().as_secs_f64();

    // As in the benchmark cell: memory is measured from a context opened
    // before the engine loads.
    let _ctx = Nn::open()?;
    let (free0, total) = mem_info()?;
    eprintln!(
        "rawatta: loading {} with config {} ({})",
        weights.display(),
        cfg.name,
        cfg.grammar
    );
    let t = Instant::now();
    let mut e = VggtEngine::load(Arc::new(Nn::open()?), &weights, vc)
        .with_context(|| format!("load VGGT-Omega weights from {}", weights.display()))?;
    let load_s = t.elapsed().as_secs_f64();
    let after_load = mem_info()?.0;
    // The plan needs only the geometry: fail before decoding every frame.
    let plan = e.plan(g).map_err(|err| {
        anyhow!(
            "{} frames at {}x{} do not fit in GPU memory: {err:#}\nUse fewer frames (--max-frames).",
            g.frames,
            g.width,
            g.height
        )
    })?;

    let t = Instant::now();
    let p = prepare_frames(&inp, a.crop_aspect, &target)?;
    let decode_preprocess_s = t.elapsed().as_secs_f64();
    eprintln!(
        "rawatta: {} frames decoded and resized to {}x{} in {decode_preprocess_s:.2} s",
        g.frames, g.width, g.height
    );
    debug_assert_eq!(p.geom, g);

    let cancel = CancelFlag::new();
    let runs = (a.warmup + a.reps) as usize;
    let min_free = Cell::new(usize::MAX);
    let sample = || {
        if let Ok((f, _)) = mem_info() {
            min_free.set(min_free.get().min(f));
        }
    };
    let mut times = Vec::with_capacity(a.reps as usize);
    let mut cams = None;
    for k in 0..runs {
        // Stage boundaries of the first pass catch buffers that the planner
        // releases before the pass ends; later passes are not disturbed.
        let mut progress = |_: &str, _: f32| {
            if k == 0 {
                sample();
            }
        };
        let c = e.run(&p, &cancel, &mut progress)?;
        sample();
        if k >= a.warmup as usize {
            times.push(e.times.total_ms);
        }
        cams = Some(c);
    }
    let cams = cams.ok_or_else(|| anyhow!("no forward pass ran"))?;
    let mut sorted = times.clone();
    sorted.sort_by(f32::total_cmp);
    let median = quantile(&sorted, 0.5);
    let iqr = quantile(&sorted, 0.75) - quantile(&sorted, 0.25);
    let peak_mib = free0.saturating_sub(min_free.get()) as f64 / 1048576.0;

    let out = &a.out;
    std::fs::create_dir_all(out).with_context(|| format!("create {}", out.display()))?;
    let mut wrote = vec!["cameras.json", "extrinsics.npy", "intrinsics.npy"];
    write_file(
        &out.join("cameras.json"),
        &serde_json::to_vec_pretty(&cameras_doc(names, &cams))?,
    )?;
    let ex: Vec<f32> = cams.extrinsics.iter().flatten().copied().collect();
    write_npy(&out.join("extrinsics.npy"), &[g.frames, 3, 4], &ex)?;
    let k: Vec<f32> = cams.intrinsics.iter().flatten().copied().collect();
    write_npy(&out.join("intrinsics.npy"), &[g.frames, 3, 3], &k)?;
    let dense = e.download(&g)?;
    if !a.no_depth {
        let shape = [g.frames, g.height, g.width];
        write_npy(&out.join("depth.npy"), &shape, &dense.depth)?;
        write_npy(&out.join("conf.npy"), &shape, &dense.conf)?;
        wrote.extend(["depth.npy", "conf.npy"]);
    }
    // World points come from the dense head: fp32 depth back-projected with
    // the predicted cameras, cam = ((u - cx)/fx * d, (v - cy)/fy * d, d) at
    // integer pixel (u, v), world = R^T (cam - t).
    let pts = point_cloud(&p, &dense, a.conf_percentile, a.max_points);
    {
        let path = out.join("points.ply");
        let mut w = BufWriter::new(
            std::fs::File::create(&path).with_context(|| format!("create {}", path.display()))?,
        );
        write_points_ply(&pts, &mut w)?;
        w.flush()?;
    }
    let cdir = out.join("colmap");
    let cnames = colmap_image_names(names);
    write_colmap_images(&cdir.join("images"), &cnames, &p)?;
    let (c_txt, i_txt, p_txt) = colmap_text(&cnames, &cams, &pts);
    write_file(&cdir.join("cameras.txt"), c_txt.as_bytes())?;
    write_file(&cdir.join("images.txt"), i_txt.as_bytes())?;
    write_file(&cdir.join("points3D.txt"), p_txt.as_bytes())?;
    wrote.extend(["points.ply", "colmap/", "timing.json"]);

    let timing = serde_json::json!({
        "config": cfg.name,
        "config_file": cfg.file.as_ref().map(|f| f.display().to_string()),
        "grammar": cfg.grammar,
        "layers": cfg.config.layers(),
        "sage_prec": cfg.config.prec(),
        "fp8_blocks": format!("{:#x}", cfg.config.fp8_blocks.0),
        "gpu": gpu,
        "compute_capability": format!("{}.{}", cc.0, cc.1),
        "frames": g.frames,
        "width": g.width,
        "height": g.height,
        "warmup": a.warmup,
        "reps": a.reps,
        "times_ms": times,
        "median_ms": median,
        "iqr_ms": iqr,
        "stages_at_last": e.times,
        "peak_device_mib": peak_mib,
        "weights_mib": free0.saturating_sub(after_load) as f64 / 1048576.0,
        "card_used_peak_mib": total.saturating_sub(min_free.get()) as f64 / 1048576.0,
        "card_total_mib": total as f64 / 1048576.0,
        "plan": {"offload": plan.offload, "release_trunk": plan.release_trunk, "dense_chunk": plan.dense_chunk},
        "inputs_s": inputs_s,
        "decode_preprocess_s": decode_preprocess_s,
        "load_s": load_s,
        "wall_s": started.elapsed().as_secs_f64(),
        "points": pts.len(),
        "rawatta_version": env!("CARGO_PKG_VERSION"),
        "notes": {
            "latency": "CUDA events around one forward pass: frame upload and normalisation, trunk, camera head, dense head. Excludes decoding, preprocessing (crop, resize) and writing outputs.",
            "protocol": "median_ms and iqr_ms are over the timed passes (reps) after the warm-up passes. The paper reports the median of 5 to 10 passes after 2 warm-ups (--reps 5 --warmup 2); with --reps 1 the median is a single sample.",
            "memory": "peak_device_mib: device memory in use above the free memory before the weights loaded (weights + workspaces); minimum free memory sampled with cuMemGetInfo at the stage boundaries of the first pass and after every pass. cuMemGetInfo is card-wide, so memory that other processes allocate on the same GPU during the run is counted too.",
        },
    });
    write_file(
        &out.join("timing.json"),
        &serde_json::to_vec_pretty(&timing)?,
    )?;

    println!(
        "rawatta run: {} frames at {}x{}, config {} ({}) on {gpu}",
        g.frames, g.width, g.height, cfg.name, cfg.grammar
    );
    println!(
        "forward: median {median:.1} ms (IQR {iqr:.1} ms) over {} timed pass(es) after {} warm-up; peak device memory {peak_mib:.0} MiB",
        a.reps, a.warmup
    );
    if a.reps < 5 || a.warmup < 2 {
        println!(
            "note: for timings to report, use --reps 5 --warmup 2 or more (the paper's protocol) on an idle GPU"
        );
    }
    println!(
        "wrote {}: {} ({} points)",
        out.display(),
        wrote.join(", "),
        pts.len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_listing_is_sorted_and_filtered() {
        let tmp = TempDir::new("list-test").unwrap();
        for n in ["b.JPG", "a.png", "c.webp", "notes.txt", "d.jpeg", "e.gif"] {
            std::fs::write(tmp.0.join(n), b"x").unwrap();
        }
        std::fs::create_dir(tmp.0.join("f.jpg")).unwrap();
        let got: Vec<String> = list_images(&tmp.0)
            .unwrap()
            .iter()
            .map(|p| file_name(p))
            .collect();
        assert_eq!(got, ["a.png", "b.JPG", "c.webp", "d.jpeg"]);
        let empty = TempDir::new("list-empty").unwrap();
        let e = format!("{:#}", list_images(&empty.0).unwrap_err());
        assert!(e.contains("no .jpg"), "{e}");
    }

    /// Folder of photos to preprocessed frames and the COLMAP image files,
    /// with a crop and frame sampling (no GPU). Frames decoded one at a time
    /// give the same pixels as the batch preprocessing.
    #[test]
    fn images_folder_to_prepared_frames() {
        use clap::Parser;
        let tmp = TempDir::new("frames-test").unwrap();
        for i in 0..5u8 {
            let img = image::RgbImage::from_fn(640, 480, |x, y| {
                image::Rgb([i * 40, (x % 256) as u8, (y % 256) as u8])
            });
            img.save(tmp.0.join(format!("img_{i}.png"))).unwrap();
        }
        let dir = tmp.0.display().to_string();
        let args = [
            "rawatta",
            "run",
            "--images",
            &dir,
            "--out",
            "o",
            "--max-frames",
            "3",
            "--crop-aspect",
            "336/512",
        ];
        let crate::Cmd::Run(a) = crate::Cli::try_parse_from(args).unwrap().cmd else {
            panic!("not run")
        };
        let inp = inputs(&a).unwrap();
        assert_eq!(inp.names, ["img_0.png", "img_2.png", "img_4.png"]);
        let f1 = load_frame(&inp.files[1], a.crop_aspect).unwrap();
        assert_eq!((f1.width, f1.height), (640, 420));
        assert_eq!(f1.pixel(0, 0)[0], 80);
        let vc = VggtConfig::default();
        let t = target(&inp, a.crop_aspect, &vc).unwrap();
        assert_eq!(
            t.geometry(3),
            vggt_omega::workspace::Geometry::new(3, 336, 512)
        );
        let p = prepare_frames(&inp, a.crop_aspect, &t).unwrap();
        assert_eq!((p.geom.frames, p.geom.width, p.geom.height), (3, 512, 336));
        let all: Vec<Rgb8Image> = inp
            .files
            .iter()
            .map(|f| load_frame(f, a.crop_aspect).unwrap())
            .collect();
        let batch = preprocess::prepare(&all, vc.resolution, vc.resize).unwrap();
        assert_eq!(batch.geom, p.geom);
        assert!(
            batch.pixels == p.pixels,
            "streamed and batch preprocessing differ"
        );
        let cnames = colmap_image_names(&inp.names);
        let cdir = tmp.0.join("colmap/images");
        write_colmap_images(&cdir, &cnames, &p).unwrap();
        let back = image::open(cdir.join("img_2.png")).unwrap().into_rgb8();
        assert_eq!(back.dimensions(), (512, 336));
        assert_eq!(
            back.as_raw()[..],
            p.pixels[512 * 336 * 3..2 * 512 * 336 * 3]
        );
    }

    /// A bad input path is reported before the weights are looked for.
    #[test]
    fn input_is_checked_before_the_weights() {
        use clap::Parser;
        let parse = |args: &[&str]| {
            let mut v = vec![
                "rawatta",
                "run",
                "--out",
                "o",
                "--weights",
                "/nonexistent/w.safetensors",
            ];
            v.extend_from_slice(args);
            let crate::Cmd::Run(a) = crate::Cli::try_parse_from(v).unwrap().cmd else {
                panic!("not run")
            };
            a
        };
        let e = format!(
            "{:#}",
            run(&parse(&["--images", "/nonexistent/photos"])).unwrap_err()
        );
        assert!(
            e.contains("image folder /nonexistent/photos not found"),
            "{e}"
        );
        let e = format!(
            "{:#}",
            run(&parse(&["--video", "/nonexistent/v.mp4"])).unwrap_err()
        );
        assert!(e.contains("video file /nonexistent/v.mp4 not found"), "{e}");
        let empty = TempDir::new("input-empty").unwrap();
        let dir = empty.0.display().to_string();
        let e = format!("{:#}", run(&parse(&["--images", &dir])).unwrap_err());
        assert!(e.contains("no .jpg"), "{e}");
        // A valid folder gets as far as the weights.
        std::fs::write(empty.0.join("a.jpg"), b"x").unwrap();
        let e = format!("{:#}", run(&parse(&["--images", &dir])).unwrap_err());
        assert!(
            e.contains("weights not found at /nonexistent/w.safetensors"),
            "{e}"
        );
    }

    #[test]
    fn colmap_names_are_unique_pngs() {
        let n = |v: &[&str]| v.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            colmap_image_names(&n(&["a.jpg", "my photo.JPG", "c"])),
            ["a.png", "my_photo.png", "c.png"]
        );
        assert_eq!(
            colmap_image_names(&n(&["a.jpg", "a.png"])),
            ["00000_a.png", "00001_a.png"]
        );
    }

    #[test]
    fn cli_parses_the_contract_flags() {
        use clap::Parser;
        let c = crate::Cli::try_parse_from([
            "rawatta",
            "run",
            "--images",
            "imgs",
            "--out",
            "o",
            "--max-frames",
            "10",
            "--config",
            "exact",
            "--crop-aspect",
            "336/512",
            "--no-depth",
        ])
        .unwrap();
        let crate::Cmd::Run(r) = c.cmd else {
            panic!("not run")
        };
        assert_eq!(r.images.as_deref(), Some(Path::new("imgs")));
        assert_eq!(r.max_frames, Some(10));
        assert_eq!(r.config, "exact");
        assert_eq!(r.crop_aspect, Some(0.65625));
        assert!(r.no_depth);
        assert_eq!((r.reps, r.warmup, r.max_points), (1, 1, 2_000_000));
        let d = crate::Cli::try_parse_from(["rawatta", "run", "--video", "v.mp4", "--out", "o"])
            .unwrap();
        let crate::Cmd::Run(r) = d.cmd else {
            panic!("not run")
        };
        assert_eq!(r.config, "rawatta");
        assert!(r.weights.is_none());
        // Exactly one input, named together in the error; --out is required.
        let e = crate::Cli::try_parse_from(["rawatta", "run", "--out", "o"])
            .unwrap_err()
            .to_string();
        assert!(e.contains("--images <DIR>|--video <FILE>"), "{e}");
        assert!(
            crate::Cli::try_parse_from([
                "rawatta", "run", "--images", "i", "--video", "v", "--out", "o"
            ])
            .is_err()
        );
        assert!(crate::Cli::try_parse_from(["rawatta", "run", "--images", "i"]).is_err());
        assert!(
            crate::Cli::try_parse_from([
                "rawatta", "run", "--images", "i", "--out", "o", "--reps", "0"
            ])
            .is_err()
        );
        let b = crate::Cli::try_parse_from(["rawatta", "bench", "--job", "j.json"]).unwrap();
        assert!(matches!(b.cmd, crate::Cmd::Bench(_)));
        assert!(matches!(
            crate::Cli::try_parse_from(["rawatta", "info"]).unwrap().cmd,
            crate::Cmd::Info(_)
        ));
    }
}
