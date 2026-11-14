// SPDX-License-Identifier: Apache-2.0
//! `rawatta`: multi-view 3D reconstruction with VGGT-Ω on a native CUDA
//! engine, with RAWATTA (Regime-AWare ATTention Approximation).
//!
//! - `rawatta info`: the GPU and whether RAWATTA (sm_89+) and the exact
//!   engine (sm_80+) run on it.
//! - `rawatta run`: images or a video in; cameras, depth, a point cloud, a
//!   COLMAP text model and timing out.
//! - `rawatta bench`: a benchmark job (the benchmark cell's job JSON and
//!   output layout).

mod info;
mod run;
mod support;
mod video;

#[cfg(test)]
mod config_files;

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use vggt_omega::bench::{Job, run_job};

const ABOUT: &str = "VGGT-Omega multi-view 3D reconstruction on a native CUDA engine, with RAWATTA \
                     (Regime-AWare ATTention Approximation).";

const AFTER_HELP: &str = "\
Configs: rawatta (default, needs sm_89+), rawatta-fast (sm_89+), exact (the exact engine,
sm_80+; built and bit-checked for sm_80+, not yet run on Ampere hardware), any config
grammar string, or the path of a .toml config file in the configs/ format; see
configs/GRAMMAR.md.
Weights: $RAWATTA_WEIGHTS or --weights, else
$XDG_CACHE_HOME/rawatta/vggt_omega_1b_512.safetensors (~/.cache/rawatta/...).
Download them once with the image's download-weights entrypoint or
tools/download_weights.sh (see the README).
The VGGT-Omega weights and engine code are under the FAIR Noncommercial Research
License: research use only.";

#[derive(Parser, Debug)]
#[command(name = "rawatta", version, about = ABOUT, after_help = AFTER_HELP)]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) cmd: Cmd,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Cmd {
    /// Show the GPU, its compute capability and memory, and whether RAWATTA
    /// (sm_89+) and the exact engine (sm_80+) can run on it.
    Info(InfoArgs),
    /// Reconstruct cameras, depth and a coloured point cloud from a folder
    /// of images or a video.
    Run(run::RunArgs),
    /// Run a benchmark job: every config on every sequence, writing
    /// <out>/<config>/<sequence>/{extrinsics,intrinsics,depth,conf}.npy and
    /// cell.json.
    Bench(BenchArgs),
}

#[derive(Args, Debug)]
pub(crate) struct InfoArgs {
    /// Weights file to look for (default: $RAWATTA_WEIGHTS, else the cache
    /// location).
    #[arg(long, value_name = "FILE")]
    weights: Option<PathBuf>,
}

#[derive(Args, Debug)]
pub(crate) struct BenchArgs {
    /// Job file: {"out": dir, "configs": [names, grammar strings or .toml
    /// config files], "reps": k, "warmup": w, "depth": bool, "crop_aspect":
    /// h/w, "sequences": [{"name": s, "images": [paths]}]}. A config file's
    /// results go under its grammar string.
    #[arg(long, value_name = "FILE")]
    job: PathBuf,
    /// Converted weights (default: $RAWATTA_WEIGHTS, else the cache
    /// location).
    #[arg(long, value_name = "FILE")]
    weights: Option<PathBuf>,
}

fn bench(a: &BenchArgs) -> Result<()> {
    let mut job = Job::from_file(&a.job)?;
    let configs = job
        .configs
        .iter()
        .map(|c| support::resolve_config(c))
        .collect::<Result<Vec<_>>>()?;
    anyhow::ensure!(!configs.is_empty(), "the job lists no configs");
    // The benchmark cell takes names and grammar strings: a config file
    // runs (and is written out) under its grammar string.
    for (c, r) in job.configs.iter_mut().zip(&configs) {
        if r.file.is_some() {
            c.clone_from(&r.grammar);
        }
    }
    let weights = support::weights_path(a.weights.as_deref());
    support::check_weights(&weights, configs.iter().any(|c| c.needs_fp8_weights()))?;
    let (gpu, cc) = support::gpu0()?;
    for c in &configs {
        support::check_compute(c, cc, &gpu)?;
    }
    run_job(&job, &weights)
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let r = match &cli.cmd {
        Cmd::Info(a) => info::info(a.weights.as_deref()),
        Cmd::Run(a) => run::run(a),
        Cmd::Bench(a) => bench(a),
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("rawatta: error: {e:#}");
            ExitCode::FAILURE
        }
    }
}
