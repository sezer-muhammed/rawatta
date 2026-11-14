// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! Named engine configurations: one implementation (this engine), many
//! attention schedules × precisions.
//!
//! A config name is `<schedule>[@<precision>][~<overrides>][+fp8[:<blocks>]]`, or one of
//! the aliases in [`ALIASES`] (`exact`, `rawatta`, `rawatta-fast`, …).
//! Schedules:
//!
//! | name | global layers (register layers 2, 6, 9, 14, 20 always stay register) |
//! |---|---|
//! | `dense` | all global |
//! | `v1` | frame 3, 4, 5; merge4 0, 1, 7, 19 (an early schedule, kept as a control) |
//! | `v1-wide` | `v1` + merge4 8, 22 |
//! | `avggt` | AVGGT-style: frame on every global layer before 9 (0, 1, 3, 4, 5, 7, 8), 2×2 merged K/V on the global layers from 10 on. GPA-VGGT's Ω layer split is the same in this engine (their r = 0.5 bipartite merge and shared cross-frame K/V need a flexible attention operator this engine does not have) |
//! | `avggt-frame` | frame on 0, 1, 3, 4, 5, 7, 8, rest dense |
//! | `frame<k>` | frame on the first k global layers |
//! | `L<i>:<op>,...` | dense except the listed layers (`op` ∈ frame, merge2, merge4); `;` also separates items, for comma-split config lists |
//!
//! Precisions: `bf16` (bf16 QK and PV), `sage` (INT8 QK, FP8 PV, FP32
//! accumulation), `sagepp` (INT8 QK, FP8 PV, FP16 block accumulation).
//! `sage` takes `.`-separated modifiers: `pv16` (= `sagepp`),
//! `pvh` (FP16 P and V, FP32 accumulate: no FP8 in P·V), `pvh16` (FP16 P
//! and V, FP16 block accumulation, two-level), `qt` (per-row Q
//! scales), `kt` (per-key K scales), `sq` (smooth Q), and for merged layers
//! (ablations) `nb` (no +log(count) bias) and `f0` (frame 0 merged too),
//! e.g. `sage.pvh.qt`, `~L19=sagepp.f0`.
//! Sage precision applies to the dense global layers; merged layers always
//! run on the quantised kernel and frame layers on the frame path (as in
//! the engine). Per-layer overrides follow `~`: `sage~L12=bf16/L13=sage.pvh`
//! (`bf16` keeps a dense layer on bf16 FlashAttention; a merged layer
//! takes only sage precisions). `+fp8`: FP8 GEMMs on [`LATE_FP8_BLOCKS`];
//! `+fp8:16-23/i5/f7` picks blocks (bare: frame and inter block, `i`/`f`:
//! one of them). Lists take `,` or `/` (`/` survives comma-separated config
//! lists).
//!
//! Not expressible (they need a flexible attention operator this engine
//! does not have):
//! exact GPA-VGGT (shared cross-frame K/V, r = 0.5 bipartite merge),
//! block-sparse top-k / CDF global attention (sparse-vggt style), FastVGGT
//! token merging and the SageAttention v1 INT8-QK / FP16-PV precision; those
//! names return an error that says so.

use anyhow::{Result, bail};

use crate::trunk::Fp8Blocks;
use crate::trunk::sched::{PvAccum, QkQuant, SagePrec};
use crate::{AttnSchedule, InterAttn, LATE_FP8_BLOCKS, REGISTER_LAYERS, VggtConfig};

/// Named settings → config strings (the paper's configuration-name table).
pub const ALIASES: [(&str, &str); 16] = [
    // The exact engine: every global layer dense, bf16 FlashAttention.
    ("exact", "dense@bf16"),
    // RAWATTA: SageAttention2++ (INT8 QKᵀ, FP8 P·V, FP16 block accumulation)
    // on every dense global layer, 16-bit P·V on the two layers the
    // calibration check flags.
    ("rawatta", "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16"),
    // RAWATTA-Fast: RAWATTA's precision plus the searched operator schedule
    // (greedy search, one calibration sample, budget -1).
    (
        "rawatta-fast",
        "L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16",
    ),
    // Threshold schedule and its frame variant.
    (
        "threshold",
        "L0:merge4/L1:merge4/L3:merge4/L4:merge4/L5:merge4/L7:merge4/L8:merge4/L19:merge4/L22:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16",
    ),
    (
        "threshold-frame",
        "L0:frame/L1:frame/L3:frame/L4:frame/L5:frame/L7:frame/L8:frame/L19:merge4/L22:merge4@sagepp~L15=sage.pvh16",
    ),
    // Greedy search with budget -2 (extra Pareto point).
    (
        "greedy-b2",
        "L0:frame/L1:merge4/L3:frame/L4:frame/L5:frame/L7:merge4/L8:merge4/L19:merge4@sagepp~L15=sage.pvh16",
    ),
    // 3-sample search: first run and seed-matched run.
    (
        "search3",
        "L0:frame/L1:merge4/L4:merge4/L5:frame/L19:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16",
    ),
    (
        "search3-seeded",
        "L0:merge4/L3:merge4/L4:merge4/L5:frame/L8:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16",
    ),
    // Earlier schedule point.
    (
        "earlier",
        "L0:merge4/L1:merge4/L3:merge4/L4:frame/L5:frame/L8:merge4@sagepp~L15=sage.pvh16",
    ),
    // Uniform low-precision kernels.
    ("sage2", "dense@sage"),
    ("sage2pp", "dense@sagepp"),
    ("pvh16", "dense@sage.pvh16"),
    // In-engine rival controls: the AVGGT/GPA-VGGT layer split in bf16; the
    // AVGGT published layers (frame on 0-8, 2×2 merge with +log(count) on
    // 10-23) and its frame half alone at RAWATTA's precision; a GPA-style
    // 2×2 merge on exactly RAWATTA-Fast's layers.
    ("avggt-split", "avggt@bf16"),
    (
        "avggt-pub",
        "L0:frame/L1:frame/L3:frame/L4:frame/L5:frame/L7:frame/L8:frame/L10:merge2/L11:merge2/L12:merge2/L13:merge2/L15:merge2/L16:merge2/L17:merge2/L18:merge2/L19:merge2/L21:merge2/L22:merge2/L23:merge2@sagepp~L15=sage.pvh16",
    ),
    (
        "avggt-pub-frame",
        "L0:frame/L1:frame/L3:frame/L4:frame/L5:frame/L7:frame/L8:frame@sagepp~L15=sage.pvh16",
    ),
    (
        "gpa-fast",
        "L0:merge2/L3:merge2/L4:merge2/L5:merge2/L19:merge2@sagepp~L4=sage.pvh16/L15=sage.pvh16",
    ),
];

const NEEDS_FLEX: [&str; 5] = ["gpa", "sparse", "fastvggt", "sage1", "flex"];

#[derive(Debug, Clone, PartialEq)]
pub struct BenchConfig {
    pub name: String,
    pub schedule: AttnSchedule,
    pub fp8_blocks: Fp8Blocks,
    pub sage_pv16: bool,
    pub sage_prec: [SagePrec; 24],
}

impl BenchConfig {
    pub fn named(name: &str) -> Result<Self> {
        let full = Self::grammar_of(name);
        let (body, fp8) = match full.split_once("+fp8") {
            Some((b, "")) => (b, Some(LATE_FP8_BLOCKS)),
            Some((b, f)) => (
                b,
                Some(fp8_list(f.strip_prefix(':').ok_or_else(|| {
                    anyhow::anyhow!("{name}: +fp8 or +fp8:<blocks>")
                })?)?),
            ),
            None => (full, None),
        };
        let (sched, prec) = body.split_once('@').unwrap_or((body, "bf16"));
        let (prec, overrides) = prec.split_once('~').unwrap_or((prec, ""));
        if NEEDS_FLEX.iter().any(|n| sched == *n || prec == *n) {
            bail!("{name}: needs a flexible attention operator that this engine does not have");
        }
        let base = precision(prec).map_err(|e| anyhow::anyhow!("{name}: {e}"))?;
        let mut s = schedule(sched)?;
        let mut sage_prec = [SagePrec::default(); 24];
        for (i, a) in s.0.iter_mut().enumerate() {
            if let Some(p) = base {
                if *a == InterAttn::Global {
                    *a = InterAttn::Sage;
                }
                sage_prec[i] = p;
            }
        }
        for item in overrides.split([',', '/']).filter(|x| !x.is_empty()) {
            let (l, p) = item
                .trim_start_matches('L')
                .split_once('=')
                .ok_or_else(|| {
                    anyhow::anyhow!("{name}: bad override {item:?} (L<i>=<precision>)")
                })?;
            let l: usize = l.parse()?;
            if l >= 24 || REGISTER_LAYERS.contains(&l) {
                bail!("{name}: layer {l} is not a global layer");
            }
            let a = &mut s.0[l];
            match (
                precision(p).map_err(|e| anyhow::anyhow!("{name}: {e}"))?,
                *a,
            ) {
                (None, InterAttn::Sage | InterAttn::Global) => *a = InterAttn::Global,
                (None, _) => bail!("{name}: layer {l} ({a:?}) has no bf16 path"),
                (Some(q), InterAttn::Global | InterAttn::Sage) => {
                    *a = InterAttn::Sage;
                    sage_prec[l] = q;
                }
                (Some(q), InterAttn::Merge2 | InterAttn::Merge4) => sage_prec[l] = q,
                (Some(_), _) => bail!("{name}: layer {l} ({a:?}) is not a sage layer"),
            }
        }
        Ok(Self {
            name: name.to_string(),
            schedule: s,
            fp8_blocks: fp8.unwrap_or(Fp8Blocks::NONE),
            sage_pv16: false,
            sage_prec,
        })
    }

    /// The grammar string `name` stands for: an alias's target, else `name`
    /// itself.
    pub fn grammar_of(name: &str) -> &str {
        ALIASES
            .iter()
            .find(|(a, _)| *a == name)
            .map_or(name, |(_, f)| *f)
    }

    /// Lowest compute capability that runs this config: 8.9 (FP8 tensor
    /// cores: Ada, Hopper, Blackwell; Hopper and sm_100 are only
    /// ptxas-checked, not yet run) when any layer runs on the quantised
    /// SageAttention kernel (sage or merged layers) or any GEMM runs in FP8;
    /// else 8.0 (bf16 tensor cores), the floor of the engine.
    pub fn min_compute_capability(&self) -> (i32, i32) {
        let quantised = self
            .schedule
            .0
            .iter()
            .any(|a| matches!(a, InterAttn::Sage | InterAttn::Merge2 | InterAttn::Merge4));
        if quantised || self.fp8_blocks != Fp8Blocks::NONE {
            (8, 9)
        } else {
            (8, 0)
        }
    }

    /// The non-default sage precisions, e.g. `L12:pvh.qt L13:pv16`.
    pub fn prec(&self) -> String {
        self.schedule
            .0
            .iter()
            .zip(&self.sage_prec)
            .enumerate()
            .filter(|(_, (a, p))| {
                matches!(a, InterAttn::Sage | InterAttn::Merge2 | InterAttn::Merge4)
                    && **p != SagePrec::default()
            })
            .map(|(i, (_, p))| format!("L{i}:{}", prec_name(p)))
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn apply(&self, cfg: &mut VggtConfig) {
        cfg.schedule = self.schedule;
        cfg.fp8_blocks = self.fp8_blocks;
        cfg.fp8 = self.fp8_blocks != Fp8Blocks::NONE;
        cfg.sage_pv16 = self.sage_pv16;
        cfg.sage_prec = self.sage_prec;
    }

    /// One letter per layer: G global, R register, F frame, 2/4 merge, S sage.
    pub fn layers(&self) -> String {
        self.schedule
            .0
            .iter()
            .map(|a| match a {
                InterAttn::Global => 'G',
                InterAttn::Register => 'R',
                InterAttn::Frame => 'F',
                InterAttn::Merge2 => '2',
                InterAttn::Merge4 => '4',
                InterAttn::Sage => 'S',
            })
            .collect()
    }
}

/// `bf16` → `None`; `sage[.mod…]` / `sagepp` → the sage precision.
fn precision(p: &str) -> Result<Option<SagePrec>> {
    if p == "bf16" {
        return Ok(None);
    }
    let mut it = p.split('.');
    let mut q = SagePrec::default();
    match it.next() {
        Some("sage") => {}
        Some("sagepp") => q.pv = PvAccum::F16,
        _ => bail!("unknown precision {p:?} (bf16, sage[.pv16|.pvh|.qt|.kt|.sq…], sagepp)"),
    }
    for m in it {
        match m {
            "pv16" => q.pv = PvAccum::F16,
            "pvh" => q.pv = PvAccum::Half,
            "pvh16" => q.pv = PvAccum::HalfF16,
            "qt" => q.qk.q_token = true,
            "kt" => q.qk.k_token = true,
            "sq" => q.qk.smooth_q = true,
            "nb" => q.no_bias = true,
            "f0" => q.merge_f0 = true,
            _ => bail!("unknown sage modifier {m:?} in {p:?} (pv16, pvh, qt, kt, sq, nb, f0)"),
        }
    }
    Ok(Some(q))
}

fn prec_name(p: &SagePrec) -> String {
    let QkQuant {
        q_token,
        k_token,
        smooth_q,
    } = p.qk;
    let mut s = String::from("sage");
    for (on, m) in [
        (p.pv == PvAccum::F16, ".pv16"),
        (p.pv == PvAccum::Half, ".pvh"),
        (p.pv == PvAccum::HalfF16, ".pvh16"),
        (q_token, ".qt"),
        (k_token, ".kt"),
        (smooth_q, ".sq"),
        (p.no_bias, ".nb"),
        (p.merge_f0, ".f0"),
    ] {
        if on {
            s.push_str(m);
        }
    }
    s
}

/// `16-23/i5/f7`: block ranges, both frame and inter unless prefixed
/// (register inter blocks never run FP8; their bits are ignored).
fn fp8_list(list: &str) -> Result<Fp8Blocks> {
    let mut b = Fp8Blocks::NONE;
    for item in list.split([',', '/']).filter(|x| !x.is_empty()) {
        let (which, r) = match item.as_bytes()[0] {
            b'i' => (1, &item[1..]),
            b'f' => (2, &item[1..]),
            _ => (3, item),
        };
        let (lo, hi) = r.split_once('-').unwrap_or((r, r));
        let (lo, hi): (usize, usize) = (lo.parse()?, hi.parse()?);
        if hi >= 24 || lo > hi {
            bail!("bad fp8 block range {item:?}");
        }
        for i in lo..=hi {
            if which & 2 != 0 {
                b = b.with_frame(i);
            }
            if which & 1 != 0 {
                b = b.with_inter(i);
            }
        }
    }
    Ok(b)
}

fn global_layers() -> impl Iterator<Item = usize> {
    (0..24).filter(|i| !REGISTER_LAYERS.contains(i))
}

fn schedule(name: &str) -> Result<AttnSchedule> {
    let mut s = AttnSchedule::dense();
    match name {
        "dense" => {}
        "v1" => s = AttnSchedule::v1(false),
        "v1-wide" => s = AttnSchedule::v1_wide(false),
        "avggt" | "avggt-frame" => {
            for i in global_layers() {
                if i < 9 {
                    s.0[i] = InterAttn::Frame;
                } else if name != "avggt-frame" {
                    s.0[i] = InterAttn::Merge2;
                }
            }
        }
        n if n.starts_with("frame") => {
            let k: usize = n[5..]
                .parse()
                .map_err(|_| anyhow::anyhow!("bad schedule {n:?}"))?;
            for i in global_layers().take(k) {
                s.0[i] = InterAttn::Frame;
            }
        }
        n if n.starts_with('L') => {
            for item in n.split([',', '/', ';']) {
                let (l, op) = item
                    .trim_start_matches('L')
                    .split_once(':')
                    .ok_or_else(|| anyhow::anyhow!("bad layer item {item:?} (L<i>:<op>)"))?;
                let l: usize = l.parse()?;
                if l >= 24 || REGISTER_LAYERS.contains(&l) {
                    bail!("layer {l} is not a global layer");
                }
                s.0[l] = match op {
                    "frame" => InterAttn::Frame,
                    "merge2" => InterAttn::Merge2,
                    "merge4" => InterAttn::Merge4,
                    "dense" => InterAttn::Global,
                    o => bail!("unknown layer op {o:?}"),
                };
            }
        }
        n => bail!("unknown schedule {n:?}"),
    }
    Ok(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aliases_and_layer_strings() {
        let d = BenchConfig::named("exact").unwrap();
        assert_eq!(d.layers(), "GGRGGGRGGRGGGGRGGGGGRGGG");
        assert_eq!(d.prec(), "");
        let r = BenchConfig::named("rawatta").unwrap();
        assert_eq!(r.layers(), "SSRSSSRSSRSSSSRSSSSSRSSS");
        assert_eq!(r.fp8_blocks, Fp8Blocks::NONE);
        for (i, p) in r.sage_prec.iter().enumerate() {
            let want = if i == 4 || i == 15 {
                PvAccum::HalfF16
            } else {
                PvAccum::F16
            };
            assert_eq!(p.pv, want, "layer {i}");
        }
        let f = BenchConfig::named("rawatta-fast").unwrap();
        assert_eq!(f.layers(), "FSR4FFRSSRSSSSRSSSS4RSSS");
        assert_eq!(f.sage_prec[15].pv, PvAccum::HalfF16);
        assert_eq!(f.sage_prec[4].pv, PvAccum::F16);
        let o = BenchConfig::named("v1@sage+fp8").unwrap();
        assert_eq!(o.schedule, AttnSchedule::v1(true));
        assert_eq!(o.fp8_blocks, LATE_FP8_BLOCKS);
        assert_eq!(
            BenchConfig::named("avggt").unwrap().layers(),
            "FFRFFFRFFR2222R22222R222"
        );
        assert_eq!(
            BenchConfig::named("avggt-pub").unwrap().layers(),
            BenchConfig::named("avggt@sagepp").unwrap().layers()
        );
        assert_eq!(
            BenchConfig::named("gpa-fast").unwrap().layers(),
            "2SR222RSSRSSSSRSSSS2RSSS"
        );
        assert_eq!(
            BenchConfig::named("frame3@sagepp").unwrap().layers(),
            "FFRFSSRSSRSSSSRSSSSSRSSS"
        );
        let pp = BenchConfig::named("frame3@sagepp").unwrap();
        assert_eq!(pp.sage_prec[12].pv, PvAccum::F16);
        assert_eq!(
            pp,
            BenchConfig::named("frame3@sage.pv16")
                .unwrap()
                .named_as("frame3@sagepp")
        );
        assert_eq!(
            BenchConfig::named("L12:frame").unwrap().layers().as_bytes()[12],
            b'F'
        );
        assert_eq!(
            BenchConfig::named("L0:merge4;L12:merge2@sage")
                .unwrap()
                .layers(),
            "4SRSSSRSSRSS2SRSSSSSRSSS"
        );
        assert!(BenchConfig::named("L2:frame").is_err());
        assert!(BenchConfig::named("gpa@bf16").is_err());
        assert!(BenchConfig::named("dense@sage1").is_err());
        let vs = BenchConfig::named(
            "L0:merge4,L1:merge4,L3:merge4,L4:frame,L5:frame,L8:merge4@sage+fp8",
        )
        .unwrap();
        assert_eq!(vs.layers(), "44R4FFRS4RSSSSRSSSSSRSSS");
        assert_eq!(vs.fp8_blocks, LATE_FP8_BLOCKS);
        for (a, _) in ALIASES {
            BenchConfig::named(a).unwrap();
        }
    }

    #[test]
    fn grammar_and_compute_capability() {
        assert_eq!(BenchConfig::grammar_of("exact"), "dense@bf16");
        assert_eq!(
            BenchConfig::grammar_of("rawatta"),
            "dense@sagepp~L4=sage.pvh16/L15=sage.pvh16"
        );
        assert_eq!(BenchConfig::grammar_of("dense@sage"), "dense@sage");
        let cc = |n: &str| BenchConfig::named(n).unwrap().min_compute_capability();
        assert_eq!(cc("exact"), (8, 0));
        assert_eq!(cc("rawatta"), (8, 9));
        assert_eq!(cc("rawatta-fast"), (8, 9));
        // Frame attention runs on the bf16 path; merging needs the sage kernel.
        assert_eq!(cc("frame3@bf16"), (8, 0));
        assert_eq!(cc("L3:merge4@bf16"), (8, 9));
        assert_eq!(cc("dense@bf16~L12=sage"), (8, 9));
        assert_eq!(cc("dense@bf16+fp8"), (8, 9));
    }

    impl BenchConfig {
        fn named_as(mut self, n: &str) -> Self {
            self.name = n.to_string();
            self
        }
    }

    #[test]
    fn per_layer_precision() {
        let c = BenchConfig::named(
            "L0:merge4/L4:frame@sage.pvh.qt~L12=bf16/L13=sage.kt/L0=sage.sq+fp8:16-23",
        )
        .unwrap();
        assert_eq!(c.layers(), "4SRSFSRSSRSSGSRSSSSSRSSS");
        assert_eq!(c.sage_prec[5].pv, PvAccum::Half);
        assert!(c.sage_prec[5].qk.q_token && !c.sage_prec[5].qk.k_token);
        assert!(c.sage_prec[13].qk.k_token && c.sage_prec[13].pv == PvAccum::F32);
        assert!(c.sage_prec[0].qk.smooth_q);
        assert_eq!(c.fp8_blocks, LATE_FP8_BLOCKS);
        assert_eq!(c.prec().split(' ').count(), 17);
        let b = BenchConfig::named("dense@bf16~L12=sage").unwrap();
        assert_eq!(b.layers(), "GGRGGGRGGRGGSGRGGGGGRGGG");
        assert_eq!(b.prec(), "");
        let f = BenchConfig::named("dense@bf16+fp8:i16/f3-4").unwrap();
        assert_eq!(f.fp8_blocks.count(), 3);
        assert!(f.fp8_blocks.inter(16) && f.fp8_blocks.frame(3) && !f.fp8_blocks.frame(16));
        assert!(BenchConfig::named("L4:frame@sage~L4=bf16").is_err());
        assert!(BenchConfig::named("dense@sage.xx").is_err());
        assert!(BenchConfig::named("dense@sage~L2=bf16").is_err());
        // Merge ablations: no +log(count) bias, frame 0 merged.
        let m =
            BenchConfig::named("L3:merge4/L19:merge4@sagepp~L3=sagepp.nb/L19=sagepp.f0").unwrap();
        assert!(m.sage_prec[3].no_bias && !m.sage_prec[3].merge_f0);
        assert!(m.sage_prec[19].merge_f0 && m.sage_prec[19].pv == PvAccum::F16);
        assert_eq!(
            m.prec()
                .split(' ')
                .filter(|x| x.contains(".nb") || x.contains(".f0"))
                .count(),
            2
        );
    }
}
