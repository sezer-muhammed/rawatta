// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! Evaluation harness for the VGGT-Ω engine: named schedule × precision
//! configs ([`configs`]), multi-view metrics ([`metrics`]: pose AUC / RRA /
//! RTA, depth, point maps) and the GPU benchmark cell ([`cell`]).

pub mod cell;
pub mod configs;
pub mod metrics;

pub use cell::{Job, run_job};
pub use configs::{ALIASES, BenchConfig};
