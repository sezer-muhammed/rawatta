// SPDX-License-Identifier: Apache-2.0
//! SageAttention2-style quantised attention for long global sequences: the global layers of
//! the VGGT-Ω engine, where one sequence holds every frame's tokens
//! (200 frames × 689 tokens = 137,800 at 512×336).
//!
//! [`SageAttention::prepare`] (`cuda/sage_quant.cu`, three launches) reads
//! bf16 Q, K and V in any `[batch][token][head][dim]`-strided layout, so the
//! packed output of the QKV GEMM works as is, and writes:
//!
//! - **Q:** INT8 with one scale per 16 rows (the rows one warp owns), with
//!   the softmax scale and log2(e) folded in;
//! - **K:** smoothed (the per-channel mean over the sequence is subtracted,
//!   exact under softmax), INT8 with one scale per 64-key block;
//! - **V:** FP8 E4M3 with one scale per channel, optionally smoothed too (the
//!   mean is added back to the output), transposed and key-permuted for the
//!   FP8 `mma.sync`.
//!
//! The per-head QK LayerNorm of VGGT-Ω (`use_qk_norm`) can run in the same
//! pass ([`QkNorm`]); VGGT-Ω's global layers have no RoPE, so the raw QKV
//! GEMM output can go straight in.
//!
//! **KV merging** ([`KvMerge::Pool`], a GPA-VGGT-style layer-choice
//! control, not GPA-VGGT itself): keys
//! and values of each frame's patch grid are mean-pooled in 2×2 or 4×4
//! cells, each pooled key gets a +log(count) logit bias so it keeps its
//! members' attention mass, camera and register tokens stay unmerged, and
//! queries are never merged. Pooling happens in the same prologue pass.
//!
//! [`SageAttention::attend`] (`cuda/sage_attn.cu`) computes INT8 QKᵀ with
//! `mma.sync m16n8k32 s8`, an exp2 online softmax, and FP8 PV with
//! `mma.sync m16n8k32 e4m3` accumulating in FP32 with two-level
//! accumulation ([`PvAccum::F16`] accumulates each 64-key block in FP16
//! instead, SageAttention2++), and writes bf16 in any strided layout.
//!
//! FP16 P·V runs at twice the FP32-accumulate rate on GeForce Ada, and the
//! quantised kernel beats bf16 FlashAttention-style attention on long
//! global sequences because INT8 and FP8 `mma.sync` have twice the bf16
//! throughput. The harness that measures speed and error against
//! FlashAttention-2, PyTorch SDPA and an fp64 reference on real VGGT-1B
//! activations is `tests.rs` `gpu_bench` with `torch_ref.py` and
//! `vggt_capture.py`; this release quotes no numbers from it.
//!
//! Merging is a per-layer decision: its error differs by an order of
//! magnitude between global layers, while the kernel adds only its own
//! quantisation error to whatever the merge does.
//!
//! Algorithm: SageAttention2 (arXiv 2411.10958) and SageAttention2++ (arXiv
//! 2505.21136), reference code Apache-2.0; KV merging in the
//! style of GPA-VGGT (arXiv 2610.37000). Our own implementation; needs compute capability 8.9 (Ada)
//! or newer.

pub mod reference;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Result, anyhow, bail};
use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{
    CudaFunction, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, DeviceRepr, LaunchConfig,
    PushKernelArg, SyncOnDrop,
};
use cudarc::nvrtc::{CompileOptions, Ptx};

/// Query rows per INT8 scale (one `mma` m-tile; mirrors `SG_QGROUP`).
pub const QGROUP: usize = 16;
/// Queries per prologue tile (mirrors `SG_QTILE`).
pub const QTILE: usize = 128;
/// Keys per K/V block and per K scale (mirrors `SG_KBLOCK`).
pub const KBLOCK: usize = 64;
/// Largest E4M3 magnitude; V is scaled to it.
pub const FP8_MAX: f32 = 448.0;

const LN_Q: i32 = 1;
const LN_K: i32 = 2;
const SMOOTH_K: i32 = 4;
const SMOOTH_V: i32 = 8;
const NO_KBIAS: i32 = 16;
/// Channel statistics per (batch, head); mirrors `SG_NSTAT`.
const NSTAT: usize = 5;

/// The NVRTC source: shared helpers, the prologue and the attention kernels.
pub static SOURCE: LazyLock<String> = LazyLock::new(|| {
    [
        include_str!("../../cuda/sage_common.cu"),
        "\n",
        include_str!("../../cuda/sage_quant.cu"),
        "\n",
        include_str!("../../cuda/sage_attn.cu"),
    ]
    .concat()
});

/// The tensor-core peak probes (`cuda/sage_peak.cu`), for benchmarks.
pub static PEAK_SOURCE: LazyLock<String> = LazyLock::new(|| {
    [
        include_str!("../../cuda/sage_common.cu"),
        "\n",
        include_str!("../../cuda/sage_peak.cu"),
    ]
    .concat()
});

trait Cuda<T> {
    fn cuda(self, what: &str) -> Result<T>;
}

impl<T, E: std::fmt::Debug> Cuda<T> for std::result::Result<T, E> {
    fn cuda(self, what: &str) -> Result<T> {
        self.map_err(|e| anyhow!("{what}: {e:?}"))
    }
}

/// How one CTA of the attention kernel is shaped. Rows per CTA are
/// `16 × m_tiles × warps`; keys stream through a `stages`-deep ring of
/// 64-key tiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Tile {
    pub warps: u32,
    pub m_tiles: u32,
    pub stages: u32,
}

impl Tile {
    /// Tuned on an RTX 4070 SUPER at head dim 64 (see the tests' benchmark).
    pub const DEFAULT: Tile = Tile {
        warps: 4,
        m_tiles: 2,
        stages: 3,
    };

    /// Head dim 128: two m-tiles per warp would spill (O alone is 128
    /// registers), so one.
    pub const D128: Tile = Tile {
        warps: 4,
        m_tiles: 1,
        stages: 3,
    };

    pub fn for_head_dim(head_dim: usize) -> Self {
        if head_dim > 64 {
            Self::D128
        } else {
            Self::DEFAULT
        }
    }

    pub fn rows(&self) -> usize {
        16 * (self.m_tiles * self.warps) as usize
    }

    fn shared_bytes(&self, head_dim: usize, v_bytes: usize) -> usize {
        self.rows() * head_dim + self.stages as usize * (1 + v_bytes) * KBLOCK * head_dim
    }

    fn validate(&self) -> Result<()> {
        if !matches!(self.warps, 2 | 4 | 8)
            || !matches!(self.m_tiles, 1 | 2 | 4)
            || !(2..=6).contains(&self.stages)
            || !QTILE.is_multiple_of(self.rows())
        {
            bail!("unsupported sage tile {self:?}");
        }
        Ok(())
    }
}

/// How each 64-key block's P·V is accumulated before it is folded into the
/// FP32 output accumulator (two-level accumulation either way).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum PvAccum {
    /// FP8 `mma` with FP32 accumulate (SageAttention2).
    #[default]
    F32,
    /// FP8 `mma` with FP16 accumulate (SageAttention2++): twice the tensor
    /// rate on GeForce Ada. V is quantised to ±[`PV16_V_RANGE`] so a block's
    /// partial sums (64 keys, P ≤ 448) stay below the FP16 maximum.
    F16,
    /// No FP8 in P·V: P and (smoothed) V in FP16, `mma` m16n8k16 with FP32
    /// accumulate (SageAttention v1's P·V; half the FP8 rate). V takes
    /// twice the workspace. Spill-free at head dim 64 only.
    Half,
    /// [`Self::Half`] with each 64-key block accumulated in FP16 (full
    /// tensor rate on GeForce Ada, the FP8 FP32-accumulate rate); V is
    /// scaled per channel to ±[`PVH16_V_RANGE`].
    HalfF16,
}

/// The range V is scaled to with [`PvAccum::HalfF16`]: 64 keys × P ≤ 1 ×
/// 256 = 16384 < 65504.
pub const PVH16_V_RANGE: f32 = 256.0;

impl PvAccum {
    /// Bytes per stored V element.
    pub fn v_bytes(self) -> usize {
        if matches!(self, Self::Half | Self::HalfF16) {
            2
        } else {
            1
        }
    }
}

/// The FP8 range V is scaled to with [`PvAccum::F16`]: 64 · 448 · 2.25 =
/// 64512 < 65504.
pub const PV16_V_RANGE: f32 = 2.25;

/// Kernel configuration: CTA shape, P·V precision and the QK quantisation
/// granularity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Config {
    pub tile: Tile,
    pub pv_accum: PvAccum,
    pub qk: QkQuant,
}

/// How finely Q and K are quantised to INT8 (all exact transformations
/// besides the rounding itself).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub struct QkQuant {
    /// One Q scale per query row instead of per 16 rows.
    pub q_token: bool,
    /// One K scale per key instead of per 64-key block.
    pub k_token: bool,
    /// Subtract Q's per-channel mean over the sequence before quantising and
    /// add `q_mean · k_j` back as an FP32 per-key logit bias.
    pub smooth_q: bool,
}

impl Config {
    pub fn for_head_dim(head_dim: usize) -> Self {
        Self {
            tile: Tile::for_head_dim(head_dim),
            pv_accum: PvAccum::F32,
            qk: QkQuant::default(),
        }
    }

    /// `pv_accum` and `qk` with a tile that does not spill: FP16 P·V and
    /// the per-element logit paths need more registers than two m-tiles
    /// per warp leave at head dim 64.
    pub fn with_precision(head_dim: usize, pv_accum: PvAccum, qk: QkQuant) -> Self {
        let mut tile = Tile::for_head_dim(head_dim);
        // HalfF16 keeps two m-tiles with the split softmax (SG_SPLIT, set in
        // `with_defines`): spill-free and faster than one m-tile.
        if pv_accum == PvAccum::Half || qk != QkQuant::default() {
            tile.m_tiles = 1;
        }
        Self { tile, pv_accum, qk }
    }

    /// Dynamic shared memory of the attention kernel.
    pub fn shared_bytes(&self, head_dim: usize) -> usize {
        self.tile.shared_bytes(head_dim, self.pv_accum.v_bytes())
    }

    /// What a workspace must hold for these kernels.
    pub fn needs(&self) -> WsNeeds {
        WsNeeds {
            v_bytes: self.pv_accum.v_bytes(),
            qbias: self.qk.smooth_q,
        }
    }
}

/// Optional workspace capacity: FP16 V and the smooth-Q bias. A workspace
/// built for the union of several configs serves each of them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct WsNeeds {
    pub v_bytes: usize,
    pub qbias: bool,
}

impl Default for WsNeeds {
    fn default() -> Self {
        Self {
            v_bytes: 1,
            qbias: false,
        }
    }
}

impl WsNeeds {
    pub fn union(self, o: Self) -> Self {
        Self {
            v_bytes: self.v_bytes.max(o.v_bytes),
            qbias: self.qbias || o.qbias,
        }
    }
}

/// Device bytes of a workspace for `shape` × `merge` (what
/// [`SageAttention::workspace_with`] allocates).
pub fn workspace_bytes(
    shape: Shape,
    merge: KvMerge,
    head_dim: usize,
    needs: WsNeeds,
) -> Result<usize> {
    let p = Plan::new(shape, merge)?;
    let bh = shape.batch * shape.heads;
    Ok(bh * p.tq_pad * head_dim
        + 4 * bh * p.tq_pad
        + bh * p.tk_pad * head_dim
        + 4 * bh * p.tk_pad
        + bh * p.tk_pad * head_dim * needs.v_bytes
        + 4 * bh * NSTAT * head_dim
        + 4 * p.tk_pad
        + 4 * if needs.qbias { bh * p.tk_pad } else { 1 }
        + 4 * bh * p.q_tiles() * NSTAT * head_dim)
}

/// Element strides of a bf16 tensor holding `[batch][token][head][dim]`
/// values with contiguous channels. Every stride and the view's start must
/// be a multiple of 8 elements (16-byte vector access).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strides {
    pub batch: usize,
    pub token: usize,
    pub head: usize,
}

impl Strides {
    /// `[B][T][H][D]`: a `[tokens][heads × dim]` activation, the layout the
    /// output projection reads.
    pub fn bthd(tokens: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch: tokens * heads * head_dim,
            token: heads * head_dim,
            head: head_dim,
        }
    }

    /// `[B][H][T][D]`, the layout PyTorch's SDPA takes.
    pub fn bhtd(tokens: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch: heads * tokens * head_dim,
            token: head_dim,
            head: tokens * head_dim,
        }
    }

    /// Q, K or V inside one packed QKV GEMM output `[B][T][3][H][D]`; view
    /// the buffer from element 0, `H × D` or `2 × H × D`.
    pub fn packed_qkv(tokens: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            batch: tokens * 3 * heads * head_dim,
            token: 3 * heads * head_dim,
            head: head_dim,
        }
    }

    /// Elements a tensor of this shape spans.
    fn extent(&self, shape: &Shape, head_dim: usize) -> usize {
        (shape.batch - 1) * self.batch
            + (shape.tokens - 1) * self.token
            + (shape.heads - 1) * self.head
            + head_dim
    }

    fn check(&self, what: &str) -> Result<()> {
        if !self.batch.is_multiple_of(8)
            || !self.token.is_multiple_of(8)
            || !self.head.is_multiple_of(8)
        {
            bail!("{what} strides must be multiples of 8 elements: {self:?}");
        }
        Ok(())
    }
}

/// One attention problem: `tokens` queries and keys per batch item and head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shape {
    pub batch: usize,
    pub heads: usize,
    pub tokens: usize,
}

/// How a frame's tokens are laid out: `special` camera and register tokens,
/// then a `grid_h × grid_w` patch grid in row-major order. VGGT-Ω at
/// 512×336: 17 special tokens and a 21 × 32 grid, 689 tokens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FrameLayout {
    pub frames: usize,
    pub special: usize,
    pub grid_h: usize,
    pub grid_w: usize,
    /// Leading frames whose keys stay unmerged (VGGT-Ω's reference frame:
    /// 1); 0 merges every frame.
    pub dense_frames: usize,
}

impl FrameLayout {
    pub fn tokens_per_frame(&self) -> usize {
        self.special + self.grid_h * self.grid_w
    }

    /// Keys per frame after `stride × stride` pooling.
    pub fn merged_per_frame(&self, stride: usize) -> usize {
        self.special + self.grid_h.div_ceil(stride) * self.grid_w.div_ceil(stride)
    }

    /// Keys of the whole sequence after pooling every frame past the first
    /// `dense_frames`.
    pub fn merged_keys(&self, stride: usize) -> usize {
        let d = self.dense_frames.min(self.frames);
        d * self.tokens_per_frame() + (self.frames - d) * self.merged_per_frame(stride)
    }
}

/// Which keys and values attention sees.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvMerge {
    /// Every key: the reference schedule.
    Dense,
    /// Patch keys and values pooled in `stride × stride` cells within each
    /// frame (`stride` 2 or 4), with a +log(count) bias per pooled key;
    /// the first `layout.dense_frames` frames keep every key.
    Pool { stride: usize, layout: FrameLayout },
}

/// Per-call options.
#[derive(Clone, Copy)]
pub struct Options<'a> {
    /// Defaults to `1 / sqrt(head_dim)`.
    pub softmax_scale: Option<f32>,
    /// Subtract the per-channel mean of K before quantising (exact).
    pub smooth_k: bool,
    /// Subtract the per-channel mean of V before quantising and add it back
    /// to the output (exact).
    pub smooth_v: bool,
    pub qk_norm: Option<QkNorm<'a>>,
    /// The +log(count) logit bias of merged keys (default on; off only for
    /// the ablation).
    pub count_bias: bool,
}

impl Default for Options<'_> {
    fn default() -> Self {
        Self {
            softmax_scale: None,
            smooth_k: true,
            smooth_v: true,
            qk_norm: None,
            count_bias: true,
        }
    }
}

/// The per-head LayerNorm VGGT-Ω applies to Q and K (`q_norm`, `k_norm`),
/// fused into the prologue. `params` holds `[q weight, q bias, k weight,
/// k bias]`, `head_dim` floats each.
#[derive(Clone, Copy)]
pub struct QkNorm<'a> {
    pub params: &'a CudaSlice<f32>,
    pub eps: f32,
}

/// Mirrors `SgGeom` in `cuda/sage_common.cu`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Geom {
    q_sb: i64,
    q_st: i64,
    q_sh: i64,
    k_sb: i64,
    k_st: i64,
    k_sh: i64,
    v_sb: i64,
    v_st: i64,
    v_sh: i64,
    o_sb: i64,
    o_st: i64,
    o_sh: i64,
    batch: i32,
    heads: i32,
    tokens: i32,
    tq_pad: i32,
    tk: i32,
    tk_pad: i32,
    merge: i32,
    frames: i32,
    frame_tokens: i32,
    special: i32,
    grid_h: i32,
    grid_w: i32,
    merged_frame: i32,
    merged_w: i32,
    q_tiles: i32,
    flags: i32,
    q_mul: f32,
    ln_eps: f32,
    v_range: f32,
    dense_frames: i32,
}

// SAFETY: plain `#[repr(C)]` data with the field order and types of SgGeom.
unsafe impl DeviceRepr for Geom {}

fn int(v: usize, what: &str) -> Result<i32> {
    i32::try_from(v).map_err(|_| anyhow!("{what} = {v} does not fit in i32"))
}

/// Sizes that follow from a [`Shape`] and a [`KvMerge`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plan {
    pub shape: Shape,
    pub merge: KvMerge,
    /// Queries per batch item rounded up to [`QTILE`].
    pub tq_pad: usize,
    /// Keys per batch item after merging.
    pub tk: usize,
    /// `tk` rounded up to [`KBLOCK`].
    pub tk_pad: usize,
}

impl Plan {
    pub fn new(shape: Shape, merge: KvMerge) -> Result<Self> {
        if shape.batch == 0 || shape.heads == 0 || shape.tokens == 0 {
            bail!("empty attention shape {shape:?}");
        }
        let tk = match merge {
            KvMerge::Dense => shape.tokens,
            KvMerge::Pool { stride, layout } => {
                if !matches!(stride, 2 | 4) {
                    bail!("KV merge stride must be 2 or 4, got {stride}");
                }
                if layout.grid_h == 0 || layout.grid_w == 0 {
                    bail!("KV merge needs a patch grid: {layout:?}");
                }
                if layout.frames * layout.tokens_per_frame() != shape.tokens {
                    bail!(
                        "frame layout {layout:?} covers {} tokens, the sequence has {}",
                        layout.frames * layout.tokens_per_frame(),
                        shape.tokens
                    );
                }
                layout.merged_keys(stride)
            }
        };
        let plan = Self {
            shape,
            merge,
            tq_pad: shape.tokens.next_multiple_of(QTILE),
            tk,
            tk_pad: tk.next_multiple_of(KBLOCK),
        };
        // Everything the kernels index with i32 must fit.
        int(plan.tq_pad, "padded queries")?;
        int(plan.tk_pad, "padded keys")?;
        int(shape.batch * shape.heads, "batch × heads")?;
        Ok(plan)
    }

    /// Keys after merging per key before: 1 when dense.
    pub fn key_ratio(&self) -> f64 {
        self.tk as f64 / self.shape.tokens as f64
    }

    fn q_tiles(&self) -> usize {
        self.tq_pad / QTILE
    }

    fn kblocks(&self) -> usize {
        self.tk_pad / KBLOCK
    }
}

/// Device buffers for one [`Plan`]: the quantised tensors and their scales.
/// Reuse it across layers and calls of the same shape.
///
/// A workspace from [`SageAttention::workspace_in`] borrows `q8`, `k8` and
/// `v8` from caller memory and never frees them.
pub struct Workspace {
    plan: Plan,
    head_dim: usize,
    /// `[b][h][tq_pad][D]` INT8 Q.
    pub q8: ManuallyDrop<CudaSlice<i8>>,
    /// `[b][h][tq_pad]` per row (equal within 16 rows unless
    /// [`QkQuant::q_token`]), softmax scale × log2(e) folded in.
    pub q_scale: CudaSlice<f32>,
    /// `[b][h][tk_pad][D]` INT8 smoothed (and pooled) K.
    pub k8: ManuallyDrop<CudaSlice<i8>>,
    /// `[b][h][tk_pad]` per key (equal within 64 keys unless
    /// [`QkQuant::k_token`]).
    pub k_scale: CudaSlice<f32>,
    /// `[b][h][tk_pad / 64][D][64]` E4M3 V, keys permuted within 16
    /// ([`PvAccum::Half`]: FP16, natural order, twice the bytes).
    pub v8: ManuallyDrop<CudaSlice<u8>>,
    /// `[b][h][5][D]`: K mean, V mean, V scale, 1 / V scale, Q mean.
    pub stats: CudaSlice<f32>,
    /// `[tk_pad]` log2 of each merged key's member count.
    pub kbias: CudaSlice<f32>,
    /// `[b][h][tk_pad]` smooth-Q logit bias (one element without
    /// [`WsNeeds::qbias`]).
    pub qbias: CudaSlice<f32>,
    needs: WsNeeds,
    part: CudaSlice<f32>,
    geom: Option<Geom>,
    /// `q8`, `k8` and `v8` live in caller memory.
    borrowed: bool,
}

impl Drop for Workspace {
    fn drop(&mut self) {
        // SAFETY: each field is taken once, here, and never used again.
        let (q8, k8, v8) = unsafe {
            (
                ManuallyDrop::take(&mut self.q8),
                ManuallyDrop::take(&mut self.k8),
                ManuallyDrop::take(&mut self.v8),
            )
        };
        if self.borrowed {
            q8.leak();
            k8.leak();
            v8.leak();
        }
    }
}

impl Workspace {
    pub fn plan(&self) -> &Plan {
        &self.plan
    }

    /// Device memory this workspace owns (borrowed buffers excluded).
    pub fn bytes(&self) -> usize {
        let big = self.q8.num_bytes() + self.k8.num_bytes() + self.v8.num_bytes();
        (if self.borrowed { 0 } else { big })
            + self.q_scale.num_bytes()
            + self.k_scale.num_bytes()
            + self.stats.num_bytes()
            + self.kbias.num_bytes()
            + self.qbias.num_bytes()
            + self.part.num_bytes()
    }

    pub fn needs(&self) -> WsNeeds {
        self.needs
    }

    /// Bytes of caller memory [`SageAttention::workspace_in`] needs for
    /// `shape` with `merge`, head dim `d` and `needs` (Q8, K8 and V8, each
    /// 256-byte aligned).
    pub fn borrow_bytes(shape: Shape, merge: KvMerge, d: usize, needs: WsNeeds) -> Result<usize> {
        let p = Plan::new(shape, merge)?;
        let (b, h) = (shape.batch, shape.heads);
        Ok((b * h * p.tq_pad * d).next_multiple_of(256)
            + (b * h * p.tk_pad * d).next_multiple_of(256)
            + (b * h * p.tk_pad * d * needs.v_bytes).next_multiple_of(256))
    }
}

/// A bf16 input tensor: any device buffer plus its [`Strides`].
pub struct Tensor<'a> {
    pub data: &'a dyn DevicePtr<u16>,
    pub strides: Strides,
}

/// The bf16 output tensor.
pub struct TensorMut<'a> {
    pub data: &'a mut dyn DevicePtrMut<u16>,
    pub strides: Strides,
}

/// Compiled sage kernels for one head dim and [`Config`] on one stream.
/// Every call enqueues work and returns without synchronising.
pub struct SageAttention {
    stream: Arc<CudaStream>,
    head_dim: usize,
    config: Config,
    prep: CudaFunction,
    finalize: CudaFunction,
    quant_kv: CudaFunction,
    quant_q: CudaFunction,
    dense: CudaFunction,
    merged: CudaFunction,
}

/// PTX per compile-option set, shared across contexts in the process.
static PTX: LazyLock<Mutex<HashMap<Vec<String>, Ptx>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Compiles `source` with NVRTC for compute_89 (FP8 needs Ada or newer; the
/// driver JITs it for newer GPUs) with extra `-D` options, once per process.
pub fn compile(source: &str, defines: &[String]) -> Result<Ptx> {
    // -default-device: generic lambdas in the kernels count as device code.
    let mut options = vec!["--std=c++17".to_string(), "-default-device".to_string()];
    options.extend(defines.iter().cloned());
    let mut cache = PTX.lock().map_err(|_| anyhow!("PTX cache lock poisoned"))?;
    if let Some(p) = cache.get(&options) {
        return Ok(p.clone());
    }
    let opts = CompileOptions {
        arch: Some("compute_89"),
        options: options.clone(),
        ..Default::default()
    };
    let ptx = cudarc::nvrtc::compile_ptx_with_opts(source, opts)
        .map_err(|e| anyhow!("NVRTC compile of the sage kernels failed: {e}"))?;
    // For SASS inspection: `ptxas -arch=sm_89` + `cuobjdump -sass` on the dump.
    if let Ok(dir) = std::env::var("SAGE_DUMP_PTX") {
        let name = options[2..].join("").replace(['-', '='], "");
        let _ = std::fs::write(format!("{dir}/sage{name}.ptx"), ptx.to_src());
    }
    cache.insert(options, ptx.clone());
    Ok(ptx)
}

/// The `-D` options that select the kernel variant for `head_dim` (64 or
/// 128) and `config`, plus `extra`: what [`compile`] gets for
/// [`SageAttention::with_config`]. Needs no GPU.
pub fn defines(head_dim: usize, config: Config, extra: &[String]) -> Result<Vec<String>> {
    if !matches!(head_dim, 64 | 128) {
        bail!("sage attention supports head dim 64 or 128, got {head_dim}");
    }
    let tile = config.tile;
    tile.validate()?;
    let mut defines = vec![
        format!("-DSG_D={head_dim}"),
        format!("-DSG_NW={}", tile.warps),
        format!("-DSG_MT={}", tile.m_tiles),
        format!("-DSG_STAGES={}", tile.stages),
        format!("-DSG_PV16={}", u8::from(config.pv_accum == PvAccum::F16)),
        format!("-DSG_PVH={}", u8::from(config.pv_accum.v_bytes() == 2)),
        format!(
            "-DSG_PVH16={}",
            u8::from(config.pv_accum == PvAccum::HalfF16)
        ),
        format!("-DSG_QTOK={}", u8::from(config.qk.q_token)),
        format!("-DSG_KTOK={}", u8::from(config.qk.k_token)),
        format!("-DSG_SQ={}", u8::from(config.qk.smooth_q)),
    ];
    // HalfF16 at two m-tiles needs the split softmax to stay spill-free.
    if config.pv_accum == PvAccum::HalfF16
        && tile.m_tiles > 1
        && !extra.iter().any(|e| e.starts_with("-DSG_SPLIT"))
    {
        defines.push("-DSG_SPLIT=1".to_string());
    }
    defines.extend(extra.iter().cloned());
    Ok(defines)
}

/// Fails unless the stream's GPU has FP8 tensor cores (compute 8.9+).
pub fn check_device(stream: &CudaStream) -> Result<()> {
    let cc = stream
        .context()
        .compute_capability()
        .cuda("compute capability")?;
    if cc < (8, 9) {
        bail!(
            "sage attention needs FP8 tensor cores (compute capability 8.9 or newer); this GPU is {}.{}",
            cc.0,
            cc.1
        );
    }
    Ok(())
}

impl SageAttention {
    /// Compiles the kernels for `head_dim` (64 or 128) with
    /// [`Config::for_head_dim`]: FP32 PV accumulation, the tuned tile.
    pub fn new(stream: Arc<CudaStream>, head_dim: usize) -> Result<Self> {
        Self::with_config(stream, head_dim, Config::for_head_dim(head_dim))
    }

    pub fn with_config(stream: Arc<CudaStream>, head_dim: usize, config: Config) -> Result<Self> {
        Self::with_defines(stream, head_dim, config, &[])
    }

    /// [`Self::with_config`] plus extra `-D` options for the kernel source
    /// (tuning experiments: `SG_SPLIT`, `SG_FAST_EXP`).
    pub(crate) fn with_defines(
        stream: Arc<CudaStream>,
        head_dim: usize,
        config: Config,
        extra: &[String],
    ) -> Result<Self> {
        let defines = defines(head_dim, config, extra)?;
        check_device(&stream)?;
        let ptx = compile(&SOURCE, &defines)?;
        let module = stream
            .context()
            .load_module(ptx)
            .cuda("load the sage kernels")?;
        let f = |name: &str| module.load_function(name).cuda(name);
        let me = Self {
            prep: f("sg_prep")?,
            finalize: f("sg_finalize")?,
            quant_kv: f("sg_quant_kv")?,
            quant_q: f("sg_quant_q")?,
            dense: f("sg_attn_dense")?,
            merged: f("sg_attn_merge")?,
            stream,
            head_dim,
            config,
        };
        let smem = config.shared_bytes(head_dim) as i32;
        for func in [&me.dense, &me.merged] {
            func.set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                smem,
            )
            .cuda("allow the sage tile's shared memory")?;
        }
        Ok(me)
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn config(&self) -> Config {
        self.config
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// Registers per thread and local (spill) bytes of the attention kernel.
    pub fn attn_resources(&self) -> Result<(i32, i32)> {
        Ok((
            self.dense.num_regs().cuda("registers")?,
            self.dense.local_size_bytes().cuda("local memory")?,
        ))
    }

    /// Allocates the buffers for `shape` with `merge`.
    pub fn workspace(&self, shape: Shape, merge: KvMerge) -> Result<Workspace> {
        self.workspace_with(shape, merge, self.config.needs())
    }

    /// Allocates the buffers for `shape` with `merge` and at least `needs`
    /// (share one workspace between kernels of several configs).
    pub fn workspace_with(
        &self,
        shape: Shape,
        merge: KvMerge,
        needs: WsNeeds,
    ) -> Result<Workspace> {
        let needs = needs.union(self.config.needs());
        let plan = Plan::new(shape, merge)?;
        let (b, h, d) = (shape.batch, shape.heads, self.head_dim);
        let s = &self.stream;
        let zeros_i8 = |n: usize| s.alloc_zeros::<i8>(n).cuda("allocate sage workspace");
        let zeros_f32 = |n: usize| s.alloc_zeros::<f32>(n).cuda("allocate sage workspace");
        Ok(Workspace {
            q8: ManuallyDrop::new(zeros_i8(b * h * plan.tq_pad * d)?),
            q_scale: zeros_f32(b * h * plan.tq_pad)?,
            k8: ManuallyDrop::new(zeros_i8(b * h * plan.tk_pad * d)?),
            k_scale: zeros_f32(b * h * plan.tk_pad)?,
            v8: ManuallyDrop::new(
                s.alloc_zeros::<u8>(b * h * plan.tk_pad * d * needs.v_bytes)
                    .cuda("allocate sage workspace")?,
            ),
            stats: zeros_f32(b * h * NSTAT * d)?,
            kbias: zeros_f32(plan.tk_pad)?,
            qbias: zeros_f32(if needs.qbias { b * h * plan.tk_pad } else { 1 })?,
            needs,
            part: zeros_f32(b * h * plan.q_tiles() * NSTAT * d)?,
            geom: None,
            head_dim: d,
            plan,
            borrowed: false,
        })
    }

    /// [`Self::workspace`] with Q8, K8 and V8 placed in caller memory
    /// starting at `mem` ([`Workspace::borrow_bytes`] bytes; the small
    /// scale buffers are still allocated). [`Self::prepare`] rewrites every
    /// byte of them, padding included, so the memory needs no clearing.
    ///
    /// # Safety
    /// `mem` must be a 256-byte aligned device allocation of at least
    /// `borrow_bytes` bytes on this stream's context that outlives the
    /// workspace and that nothing else uses between a [`Self::prepare`]
    /// and its [`Self::attend`].
    pub unsafe fn workspace_in(
        &self,
        shape: Shape,
        merge: KvMerge,
        needs: WsNeeds,
        mem: cudarc::driver::sys::CUdeviceptr,
        bytes: usize,
    ) -> Result<Workspace> {
        let needs = needs.union(self.config.needs());
        let need = Workspace::borrow_bytes(shape, merge, self.head_dim, needs)?;
        if bytes < need || !mem.is_multiple_of(256) {
            bail!("sage workspace_in: {bytes} bytes at {mem:#x}, needs {need} aligned");
        }
        let plan = Plan::new(shape, merge)?;
        let (b, h, d) = (shape.batch, shape.heads, self.head_dim);
        let s = &self.stream;
        let zeros_f32 = |n: usize| s.alloc_zeros::<f32>(n).cuda("allocate sage workspace");
        let nq = b * h * plan.tq_pad * d;
        let nk = b * h * plan.tk_pad * d;
        let k_at = mem + nq.next_multiple_of(256) as u64;
        let v_at = k_at + nk.next_multiple_of(256) as u64;
        // SAFETY: the three ranges are disjoint and inside the caller's
        // allocation (checked above); `Drop` leaks them instead of freeing.
        let (q8, k8, v8) = unsafe {
            (
                s.upgrade_device_ptr::<i8>(mem, nq),
                s.upgrade_device_ptr::<i8>(k_at, nk),
                s.upgrade_device_ptr::<u8>(v_at, nk * needs.v_bytes),
            )
        };
        Ok(Workspace {
            q8: ManuallyDrop::new(q8),
            q_scale: zeros_f32(b * h * plan.tq_pad)?,
            k8: ManuallyDrop::new(k8),
            k_scale: zeros_f32(b * h * plan.tk_pad)?,
            v8: ManuallyDrop::new(v8),
            stats: zeros_f32(b * h * NSTAT * d)?,
            kbias: zeros_f32(plan.tk_pad)?,
            qbias: zeros_f32(if needs.qbias { b * h * plan.tk_pad } else { 1 })?,
            needs,
            part: zeros_f32(b * h * plan.q_tiles() * NSTAT * d)?,
            geom: None,
            head_dim: d,
            plan,
            borrowed: true,
        })
    }

    /// Quantises (and merges) Q, K and V into `ws`: three launches.
    pub fn prepare(
        &self,
        ws: &mut Workspace,
        q: Tensor<'_>,
        k: Tensor<'_>,
        v: Tensor<'_>,
        opts: &Options<'_>,
    ) -> Result<()> {
        if ws.head_dim != self.head_dim {
            bail!(
                "workspace was made for head dim {}, kernels are {}",
                ws.head_dim,
                self.head_dim
            );
        }
        let need = self.config.needs();
        if ws.needs.union(need) != ws.needs {
            bail!(
                "workspace holds {:?}, these kernels need {need:?}",
                ws.needs
            );
        }
        let plan = ws.plan;
        let shape = plan.shape;
        let d = self.head_dim;
        for (t, what) in [(&q, "Q"), (&k, "K"), (&v, "V")] {
            t.strides.check(what)?;
            let need = t.strides.extent(&shape, d);
            if t.data.len() < need {
                bail!(
                    "{what} holds {} elements, its shape and strides need {need}",
                    t.data.len()
                );
            }
        }
        if let Some(n) = &opts.qk_norm
            && n.params.len() < 4 * d
        {
            bail!(
                "QK-norm params hold {} floats, need {}",
                n.params.len(),
                4 * d
            );
        }
        let (stride, layout) = match plan.merge {
            KvMerge::Dense => (1, None),
            KvMerge::Pool { stride, layout } => (stride, Some(layout)),
        };
        let mut flags = 0;
        if opts.qk_norm.is_some() {
            flags |= LN_Q | LN_K;
        }
        if opts.smooth_k {
            flags |= SMOOTH_K;
        }
        if opts.smooth_v {
            flags |= SMOOTH_V;
        }
        if !opts.count_bias {
            flags |= NO_KBIAS;
        }
        let scale = opts
            .softmax_scale
            .unwrap_or(1.0 / (self.head_dim as f32).sqrt());
        let lay = layout.unwrap_or(FrameLayout {
            frames: 1,
            special: 0,
            grid_h: 0,
            grid_w: 0,
            dense_frames: 0,
        });
        let geom = Geom {
            q_sb: q.strides.batch as i64,
            q_st: q.strides.token as i64,
            q_sh: q.strides.head as i64,
            k_sb: k.strides.batch as i64,
            k_st: k.strides.token as i64,
            k_sh: k.strides.head as i64,
            v_sb: v.strides.batch as i64,
            v_st: v.strides.token as i64,
            v_sh: v.strides.head as i64,
            batch: int(shape.batch, "batch")?,
            heads: int(shape.heads, "heads")?,
            tokens: int(shape.tokens, "tokens")?,
            tq_pad: int(plan.tq_pad, "padded queries")?,
            tk: int(plan.tk, "keys")?,
            tk_pad: int(plan.tk_pad, "padded keys")?,
            merge: stride as i32,
            frames: int(lay.frames, "frames")?,
            frame_tokens: int(lay.tokens_per_frame(), "frame tokens")?,
            special: int(lay.special, "special tokens")?,
            grid_h: int(lay.grid_h, "grid height")?,
            grid_w: int(lay.grid_w, "grid width")?,
            merged_frame: int(lay.merged_per_frame(stride), "merged frame")?,
            merged_w: int(lay.grid_w.div_ceil(stride), "merged grid width")?,
            dense_frames: int(lay.dense_frames.min(lay.frames), "dense frames")?,
            q_tiles: int(plan.q_tiles(), "query tiles")?,
            flags,
            q_mul: scale * std::f32::consts::LOG2_E,
            ln_eps: opts.qk_norm.map_or(1e-5, |n| n.eps),
            v_range: match self.config.pv_accum {
                PvAccum::F32 | PvAccum::Half => FP8_MAX,
                PvAccum::HalfF16 => PVH16_V_RANGE,
                PvAccum::F16 => PV16_V_RANGE,
            },
            ..Default::default()
        };

        let s = &*self.stream;
        let mut guards: Vec<SyncOnDrop<'_>> = Vec::with_capacity(4);
        let (qp, g) = q.data.device_ptr(s);
        guards.push(g);
        let (kp, g) = k.data.device_ptr(s);
        guards.push(g);
        let (vp, g) = v.data.device_ptr(s);
        guards.push(g);
        // Null when no LayerNorm runs: the kernels then never read it.
        let ln: u64 = match &opts.qk_norm {
            Some(n) => {
                let (p, g) = n.params.device_ptr(s);
                guards.push(g);
                p
            }
            None => 0,
        };
        let (b, h) = (shape.batch as u32, shape.heads as u32);

        let cfg = LaunchConfig {
            grid_dim: (plan.q_tiles() as u32, h, b),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(&self.prep);
        l.arg(&qp)
            .arg(&kp)
            .arg(&vp)
            .arg(&ln)
            .arg(&mut *ws.q8)
            .arg(&mut ws.q_scale)
            .arg(&mut ws.part)
            .arg(&geom);
        // SAFETY: argument types match sg_prep; bounds were checked above.
        unsafe { l.launch(cfg) }.cuda("launch sg_prep")?;

        let cfg = LaunchConfig {
            grid_dim: (h, b, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(&self.finalize);
        l.arg(&ws.part).arg(&mut ws.stats).arg(&geom);
        // SAFETY: argument types match sg_finalize.
        unsafe { l.launch(cfg) }.cuda("launch sg_finalize")?;

        let cfg = LaunchConfig {
            grid_dim: (plan.kblocks() as u32, h, b),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        let mut l = s.launch_builder(&self.quant_kv);
        l.arg(&kp)
            .arg(&vp)
            .arg(&ln)
            .arg(&ws.stats)
            .arg(&mut *ws.k8)
            .arg(&mut ws.k_scale)
            .arg(&mut *ws.v8)
            .arg(&mut ws.kbias)
            .arg(&mut ws.qbias)
            .arg(&geom);
        // SAFETY: argument types match sg_quant_kv; bounds were checked above
        // (qbias is written only with smooth Q, whose workspace holds it).
        unsafe { l.launch(cfg) }.cuda("launch sg_quant_kv")?;

        if self.config.qk.smooth_q {
            let cfg = LaunchConfig {
                grid_dim: (plan.q_tiles() as u32, h, b),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = s.launch_builder(&self.quant_q);
            l.arg(&qp)
                .arg(&ln)
                .arg(&ws.stats)
                .arg(&mut *ws.q8)
                .arg(&mut ws.q_scale)
                .arg(&geom);
            // SAFETY: argument types match sg_quant_q; bounds as sg_prep.
            unsafe { l.launch(cfg) }.cuda("launch sg_quant_q")?;
        }
        drop(guards);
        ws.geom = Some(geom);
        Ok(())
    }

    /// Attention over the tensors [`Self::prepare`] left in `ws`; writes
    /// `[batch][token][head][dim]` bf16 into `out`.
    pub fn attend(&self, ws: &Workspace, out: TensorMut<'_>) -> Result<()> {
        let Some(mut geom) = ws.geom else {
            bail!("sage workspace has not been prepared");
        };
        let need = self.config.needs();
        if ws.needs.union(need) != ws.needs {
            bail!(
                "workspace holds {:?}, these kernels need {need:?}",
                ws.needs
            );
        }
        let plan = ws.plan;
        out.strides.check("output")?;
        let need = out.strides.extent(&plan.shape, self.head_dim);
        if out.data.len() < need {
            bail!(
                "output holds {} elements, its shape and strides need {need}",
                out.data.len()
            );
        }
        geom.o_sb = out.strides.batch as i64;
        geom.o_st = out.strides.token as i64;
        geom.o_sh = out.strides.head as i64;
        let s = &*self.stream;
        let (op, guard) = out.data.device_ptr_mut(s);
        let func = match plan.merge {
            KvMerge::Dense => &self.dense,
            KvMerge::Pool { .. } => &self.merged,
        };
        let cfg = LaunchConfig {
            grid_dim: (
                (plan.tq_pad / self.config.tile.rows()) as u32,
                plan.shape.heads as u32,
                plan.shape.batch as u32,
            ),
            block_dim: (self.config.tile.warps * 32, 1, 1),
            shared_mem_bytes: self.config.shared_bytes(self.head_dim) as u32,
        };
        let mut l = s.launch_builder(func);
        l.arg(&*ws.q8)
            .arg(&ws.q_scale)
            .arg(&*ws.k8)
            .arg(&ws.k_scale)
            .arg(&*ws.v8)
            .arg(&ws.stats)
            .arg(&ws.kbias)
            .arg(&ws.qbias)
            .arg(&op)
            .arg(&geom);
        // SAFETY: argument types match sg_attn_*; the workspace was sized by
        // its plan and the output extent was checked above.
        unsafe { l.launch(cfg) }.cuda("launch sage attention")?;
        drop(guard);
        Ok(())
    }

    /// [`Self::prepare`] then [`Self::attend`].
    pub fn forward(
        &self,
        ws: &mut Workspace,
        q: Tensor<'_>,
        k: Tensor<'_>,
        v: Tensor<'_>,
        out: TensorMut<'_>,
        opts: &Options<'_>,
    ) -> Result<()> {
        self.prepare(ws, q, k, v, opts)?;
        self.attend(ws, out)
    }
}

/// Multiply-accumulate operations (×2 = FLOPs) of one attention call: QKᵀ
/// and PV over every (query, key) pair, `4 · B · H · Tq · Tk · D` FLOPs.
pub fn flops(plan: &Plan, head_dim: usize) -> f64 {
    4.0 * (plan.shape.batch * plan.shape.heads) as f64
        * plan.shape.tokens as f64
        * plan.tk as f64
        * head_dim as f64
}
