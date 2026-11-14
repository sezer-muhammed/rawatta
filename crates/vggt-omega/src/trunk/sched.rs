// SPDX-License-Identifier: LicenseRef-FAIR-Noncommercial-Research
// A port of Meta's VGGT-Ω architecture; see LICENSE-VGGT-OMEGA.
//! Approximate inter-frame attention for the attention schedule:
//! SageAttention (INT8 QKᵀ, FP8 P·V) for global layers, and KV merging
//! (2×2 / 4×4 mean-pooled patch keys and values with a +log(count) logit
//! bias; camera and register tokens and every key of frame 0 stay
//! unmerged) through the same quantised kernel. `Frame` layers need no
//! kernel of their own: the aggregator runs them as frame attention.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, anyhow};
use rawatta_nn::Nn;
use rawatta_nn::attention::{FlashAttention, VFormat};
use rawatta_nn::cudarc::driver::{DevicePtr, DevicePtrMut};
use rawatta_nn::qkv::HEAD_DIM;
use rawatta_nn::sage::{
    self, FrameLayout, KvMerge, Options, SageAttention, Shape, Strides, Tensor, TensorMut,
    Workspace, WsNeeds,
};

use super::{ATTN_SCALE, DEPTH, HEADS, InterAttention, InterKind, SPECIAL, TrunkGeom};

pub use rawatta_nn::sage::{PvAccum, QkQuant};

/// The precision of one SageAttention layer: P·V format and the INT8 QK
/// granularity.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct SagePrec {
    pub pv: PvAccum,
    pub qk: QkQuant,
    /// Merged layers only, ablation: drop the +log(count) logit bias.
    pub no_bias: bool,
    /// Merged layers only, ablation: pool frame 0's keys too (no
    /// reference-frame exemption).
    pub merge_f0: bool,
}

impl SagePrec {
    fn config(self) -> sage::Config {
        sage::Config::with_precision(HEAD_DIM, self.pv, self.qk)
    }

    /// The part that selects compiled kernels (the merge options do not).
    fn kernel_key(self) -> Self {
        Self {
            no_bias: false,
            merge_f0: false,
            ..self
        }
    }
}

/// Workspace capacity the sage layers of `sched` need with `prec`.
pub(crate) fn needs(sched: &[InterKind; DEPTH], prec: &[SagePrec; DEPTH]) -> WsNeeds {
    sched
        .iter()
        .zip(prec)
        .filter(|(k, _)| is_sage(**k))
        .fold(WsNeeds::default(), |n, (_, p)| n.union(p.config().needs()))
}

fn is_sage(k: InterKind) -> bool {
    matches!(k, InterKind::Sage | InterKind::Merge2 | InterKind::Merge4)
}

/// The KV merge one schedule entry asks for (`None`: not a sage layer).
pub(crate) fn merge_of(kind: InterKind, g: &TrunkGeom, p: SagePrec) -> Option<KvMerge> {
    let pool = |stride| KvMerge::Pool {
        stride,
        layout: FrameLayout {
            frames: g.frames,
            special: SPECIAL,
            grid_h: g.grid_h,
            grid_w: g.grid_w,
            dense_frames: usize::from(!p.merge_f0),
        },
    };
    match kind {
        InterKind::Sage => Some(KvMerge::Dense),
        // One frame has nothing to merge: frame 0 stays dense.
        InterKind::Merge2 | InterKind::Merge4 if g.frames == 1 => Some(KvMerge::Dense),
        InterKind::Merge2 => Some(pool(2)),
        InterKind::Merge4 => Some(pool(4)),
        _ => None,
    }
}

fn shape(g: &TrunkGeom) -> Shape {
    Shape {
        batch: 1,
        heads: HEADS,
        tokens: g.rows(),
    }
}

/// Device bytes of one sage workspace (what `SageAttention::workspace_with`
/// allocates). With `borrowed`, Q8, K8 and V8 live in the trunk's `big`
/// buffer and only the scales count.
pub(crate) fn workspace_bytes(
    g: &TrunkGeom,
    merge: KvMerge,
    needs: WsNeeds,
    borrowed: bool,
) -> usize {
    let Ok(p) = sage::Plan::new(shape(g), merge) else {
        return 0;
    };
    let full = sage::workspace_bytes(shape(g), merge, HEAD_DIM, needs).unwrap_or(0);
    if borrowed {
        full - HEADS * HEAD_DIM * (p.tq_pad + p.tk_pad * (1 + needs.v_bytes))
    } else {
        full
    }
}

/// Every distinct sage workspace a schedule needs at `g`.
pub(crate) fn merges(
    sched: &[InterKind; DEPTH],
    g: &TrunkGeom,
    prec: &[SagePrec; DEPTH],
) -> Vec<KvMerge> {
    let mut out: Vec<KvMerge> = Vec::new();
    for (&k, &p) in sched.iter().zip(prec) {
        if let Some(m) = merge_of(k, g, p)
            && !out.contains(&m)
        {
            out.push(m);
        }
    }
    out
}

/// Whether every sage workspace of `g` fits in `bytes` of borrowed memory
/// (the trunk's `big` buffer, idle while attention runs).
pub(crate) fn fits_borrowed(g: &TrunkGeom, merge: KvMerge, needs: WsNeeds, bytes: usize) -> bool {
    Workspace::borrow_bytes(shape(g), merge, HEAD_DIM, needs).is_ok_and(|n| n <= bytes)
}

/// The compiled sage kernels (one set per layer precision) and the
/// workspaces of the current geometry.
///
/// Q8, K8 and V8 of every workspace share one borrowed region (the trunk's
/// `big` buffer) when it is large enough: only one sage layer runs at a
/// time, and `big` is idle between the QKV split and the projection.
pub(crate) struct Sage {
    kernels: HashMap<SagePrec, SageAttention>,
    prec: [SagePrec; DEPTH],
    ws: HashMap<WsKey, Workspace>,
    /// The region the current workspaces borrow.
    arena: Option<(u64, usize)>,
}

type WsKey = (usize, usize, usize, usize, usize);

fn ws_key(g: &TrunkGeom, merge: KvMerge) -> WsKey {
    let (s, d) = match merge {
        KvMerge::Dense => (1, 0),
        KvMerge::Pool { stride, layout } => (stride, layout.dense_frames),
    };
    (g.frames, g.grid_h, g.grid_w, s, d)
}

impl Sage {
    pub fn new() -> Self {
        Self {
            kernels: HashMap::new(),
            prec: [SagePrec::default(); DEPTH],
            ws: HashMap::new(),
            arena: None,
        }
    }

    /// Per-layer precision of the sage layers (kernels compile on demand).
    pub fn set_prec(&mut self, prec: [SagePrec; DEPTH]) {
        self.prec = prec;
    }

    pub fn release(&mut self) {
        self.ws.clear();
        self.arena = None;
    }

    /// Compiles the kernels and allocates the workspaces `sched` needs at
    /// `g` (dropping those of other geometries). `arena` is device memory
    /// (pointer, bytes) the workspaces may borrow; it must stay allocated
    /// until the next `prepare` with a different arena or [`Self::release`].
    /// Call outside graph capture.
    pub fn prepare(
        &mut self,
        nn: &Nn,
        sched: &[InterKind; DEPTH],
        g: &TrunkGeom,
        arena: Option<(u64, usize)>,
    ) -> Result<()> {
        let need = merges(sched, g, &self.prec);
        if self.arena != arena {
            self.ws.clear();
            self.arena = arena;
        }
        if need.is_empty() {
            return Ok(());
        }
        for (i, &kind) in sched.iter().enumerate() {
            let p = self.prec[i].kernel_key();
            if is_sage(kind) && !self.kernels.contains_key(&p) {
                let k = SageAttention::with_config(Arc::clone(&nn.stream), HEAD_DIM, p.config())?;
                self.kernels.insert(p, k);
            }
        }
        let ws_needs = needs(sched, &self.prec);
        let keys: Vec<_> = need.iter().map(|&m| ws_key(g, m)).collect();
        self.ws
            .retain(|k, w| keys.contains(k) && w.needs() == ws_needs);
        let k = self
            .kernels
            .values()
            .next()
            .ok_or_else(|| anyhow!("no sage kernels"))?;
        for m in need {
            if let std::collections::hash_map::Entry::Vacant(e) = self.ws.entry(ws_key(g, m)) {
                let ws = match arena {
                    Some((ptr, bytes)) if fits_borrowed(g, m, ws_needs, bytes) => {
                        // SAFETY: `arena` is the trunk's `big` buffer, kept
                        // alive while these workspaces exist (a new buffer
                        // changes the arena and clears them above); sage
                        // layers only run while `big` is idle.
                        unsafe { k.workspace_in(shape(g), m, ws_needs, ptr, bytes)? }
                    }
                    _ => k.workspace_with(shape(g), m, ws_needs)?,
                };
                e.insert(ws);
            }
        }
        Ok(())
    }

    /// The attention for sage layer `i` of `kind` at `g`.
    pub fn layer(&mut self, i: usize, kind: InterKind, g: &TrunkGeom) -> Result<SageLayer<'_>> {
        let p = self.prec[i];
        let merge = merge_of(kind, g, p).ok_or_else(|| anyhow!("{kind:?} is not a sage layer"))?;
        let kernels = self
            .kernels
            .get(&p.kernel_key())
            .ok_or_else(|| anyhow!("sage kernels for layer {i} not prepared"))?;
        let ws = self
            .ws
            .get_mut(&ws_key(g, merge))
            .ok_or_else(|| anyhow!("sage workspace for {kind:?} at {g:?} not prepared"))?;
        Ok(SageLayer {
            kernels,
            ws: RefCell::new(ws),
            count_bias: !p.no_bias,
        })
    }
}

/// One global layer through SageAttention, reading the head-major bf16
/// Q/K/V the QKV prologue writes (QK-norm already applied).
pub(crate) struct SageLayer<'a> {
    kernels: &'a SageAttention,
    ws: RefCell<&'a mut Workspace>,
    count_bias: bool,
}

impl InterAttention for SageLayer<'_> {
    fn v_format(&self) -> VFormat {
        VFormat::Bf16
    }

    fn forward(
        &self,
        _nn: &Nn,
        _fa: &FlashAttention,
        q: &dyn DevicePtr<u16>,
        k: &dyn DevicePtr<u16>,
        v: &dyn DevicePtr<u16>,
        o: &mut dyn DevicePtrMut<u16>,
        n_seq: usize,
        seq_len: usize,
    ) -> Result<()> {
        let mut ws = self.ws.borrow_mut();
        let tokens = n_seq * seq_len;
        if n_seq != 1 || ws.plan().shape.tokens != tokens {
            return Err(anyhow!(
                "sage layer: workspace for {:?}, called with {n_seq} × {seq_len}",
                ws.plan().shape
            ));
        }
        let hm = Strides::bhtd(tokens, HEADS, HEAD_DIM);
        let opts = Options {
            softmax_scale: Some(ATTN_SCALE),
            count_bias: self.count_bias,
            ..Options::default()
        };
        self.kernels.forward(
            &mut ws,
            Tensor {
                data: q,
                strides: hm,
            },
            Tensor {
                data: k,
                strides: hm,
            },
            Tensor {
                data: v,
                strides: hm,
            },
            TensorMut {
                data: o,
                strides: Strides::bthd(tokens, HEADS, HEAD_DIM),
            },
            &opts,
        )
    }
}
