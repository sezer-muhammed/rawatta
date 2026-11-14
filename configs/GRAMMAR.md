# Config grammar

`rawatta run --config <config>` and the `configs` list of a `rawatta bench` job take either a
**named setting** or a **config string**. A config string sets, for each of the 24 global
(inter-frame) attention layers of VGGT-Ω, which attention operator runs and in which numeric
format. The named settings are short names for config strings.

| Name | Config string | Needs |
|---|---|---|
| `rawatta` (default) | `dense@sagepp~L4=sage.pvh16/L15=sage.pvh16` | sm_89+ |
| `rawatta-fast` | `L0:frame/L3:merge4/L4:frame/L5:frame/L19:merge4@sagepp~L15=sage.pvh16` | sm_89+ |
| `exact` | `dense@bf16` | sm_80+ |

`rawatta.toml`, `rawatta-fast.toml` and `exact.toml` in this folder describe the three settings
and where the paper defines them. The supplementary material's "Configuration names" table maps
every name used in the paper to its config string.

## Config files

`--config` (and a bench job's `configs` list) also takes the path of a `.toml` file in the format
of the files here, e.g. `--config configs/rawatta-fast.toml` or your own `my-schedule.toml`:

```toml
name = "my-schedule"                 # optional; default: the file name without .toml
grammar = "L0:frame/L19:merge4@sagepp"   # required: a config string or a named setting
min_compute_capability = "8.9"       # optional; may raise the gate, never lower it
```

`display_name`, `description` and `paper_reference` are optional and only descriptive; any other
key is an error. The run uses `name` in its output (`timing.json` records the file as well). A
bench job runs a config file under its config string, so its results go in a folder named after
that string.

## Form

```text
<schedule>[@<precision>][~<overrides>][+fp8[:<blocks>]]
```

Lists separate items with `/` or `,`. Prefer `/`: it survives where configs are themselves a
comma-separated list.

Layers are numbered 0 to 23. Layers 2, 6, 9, 14 and 20 are register layers: the checkpoint fixes
them to attention among the special tokens of all frames (N×17 tokens, in bf16), and no config
changes them. Every other
layer is a *global* layer and is dense unless the schedule says otherwise.

## Schedule

The operator of each global layer.

| Schedule | Layers |
|---|---|
| `dense` | every global layer dense: all tokens of all frames attend to each other |
| `L<i>:<op>/...` | dense except the listed layers; `<op>` is `frame`, `merge2`, `merge4` or `dense`; `;` also separates items |
| `frame<k>` | frame attention on the first k global layers |
| `avggt` | AVGGT-style layer split: frame attention on the global layers before 9 (0, 1, 3, 4, 5, 7, 8), `merge2` on the global layers from 10 on |
| `avggt-frame` | frame attention on 0, 1, 3, 4, 5, 7, 8, the rest dense |
| `v1` | frame attention on 3, 4, 5; `merge4` on 0, 1, 7, 19 (an early schedule, kept as a control) |
| `v1-wide` | `v1` plus `merge4` on 8 and 22 |

Operators:

- `frame`: each frame attends only to its own tokens. Runs on the bf16 path.
- `merge2`, `merge4`: queries attend to frame 0 complete; frames 1 onwards contribute their 17
  special tokens unpooled plus 2×2- or 4×4-mean-pooled patch keys and values (after QK-norm), with
  a +log(count) bias that keeps each pooled key's weight. Always runs on the quantised
  SageAttention kernel.
- `dense`: full global attention, in the layer's precision.

## Precision

The numeric format of the dense global layers. Merged layers take it when it is a `sage` format;
under `bf16` they run in plain `sage` (INT8 QKᵀ, FP8 P·V, FP32 accumulation), because they have
no bf16 path. Frame layers always run on the bf16 kernel, whatever the precision.

| Precision | QKᵀ | P·V |
|---|---|---|
| `bf16` (default) | bf16 | bf16 (FlashAttention-style kernel) |
| `sage` | INT8, per-block scales, key smoothing | FP8 E4M3, FP32 accumulation |
| `sagepp` | INT8 | FP8 E4M3, FP16 accumulation per 64-key block (SageAttention2++) |

`sage` takes modifiers joined with `.`, for example `sage.pvh.qt`:

| Modifier | Effect |
|---|---|
| `pv16` | FP16 accumulation per 64-key block (`sage.pv16` is `sagepp`) |
| `pvh` | 16-bit P·V: FP16 P and V, FP32 accumulation (no FP8 in P·V) |
| `pvh16` | 16-bit P·V: FP16 P and V, FP16 accumulation per 64-key block |
| `qt` | per-row (per-token) Q scales |
| `kt` | per-key K scales |
| `sq` | smooth Q |
| `nb` | merged layers only: no +log(count) bias (ablation) |
| `f0` | merged layers only: frame 0 is pooled too (ablation) |

## Per-layer overrides

`~` followed by `L<i>=<precision>` items sets the precision of single layers:

- `dense@sagepp~L4=sage.pvh16/L15=sage.pvh16`: SageAttention2++ everywhere, 16-bit P·V on 4 and 15.
- `dense@sage~L12=bf16`: layer 12 back on exact bf16 attention.
- `dense@bf16~L12=sage`: only layer 12 quantised.
- `L19:merge4@sagepp~L19=sagepp.f0`: an ablation on a merged layer.

A merged layer takes only `sage` precisions, a frame layer takes none, and register layers
cannot be overridden.

## FP8 GEMMs

`+fp8` runs the linear layers (QKV, projection, MLP) of trunk blocks 16 to 23 in FP8 E4M3.
`+fp8:<blocks>` picks blocks: `16-23/i5/f7` means frame and global blocks 16 to 23, global block
5 and frame block 7 (`i` = inter-frame/global block only, `f` = frame block only). Register blocks
always stay bf16. FP8 GEMMs need the FP8 side file `<weights>_fp8.safetensors` (create it with
`tools/convert_weights.py --fp8`). None of the named settings uses FP8 GEMMs.

## Hardware

A config needs compute capability 8.9 or newer (FP8 tensor cores: Ada, Hopper, Blackwell; run on
Ada (RTX 4070 SUPER) and, in the paper, Blackwell sm_120 (RTX PRO 6000); Hopper (sm_90) and sm_100
are built and ptxas-checked but not yet run) when any layer runs on the quantised kernel (a `sage`/`sagepp` precision or a merged layer) or `+fp8`
is set. Otherwise it needs 8.0 or newer (Ampere and later), the floor of the engine: built and
bit-checked for sm_80+, not yet run on Ampere hardware. `rawatta run` checks this before loading
the weights; `rawatta info` shows what the GPU supports.

## Other named configs

These aliases name the other configurations of the paper's "Configuration names" table
(supplementary material). All of them need sm_89+: each runs at least one layer on the quantised kernel,
`avggt-split` included (its `merge2` layers run there even under `@bf16`).

| Name | Paper name | Config string |
|---|---|---|
| `threshold` | threshold schedule | `L0:merge4/L1:merge4/L3:merge4/L4:merge4/L5:merge4/L7:merge4/L8:merge4/L19:merge4/L22:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16` |
| `threshold-frame` | threshold schedule, frame variant | `L0:frame/L1:frame/L3:frame/L4:frame/L5:frame/L7:frame/L8:frame/L19:merge4/L22:merge4@sagepp~L15=sage.pvh16` |
| `greedy-b2` | greedy search, budget −2 | `L0:frame/L1:merge4/L3:frame/L4:frame/L5:frame/L7:merge4/L8:merge4/L19:merge4@sagepp~L15=sage.pvh16` |
| `search3` | 3-sample search, first run | `L0:frame/L1:merge4/L4:merge4/L5:frame/L19:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16` |
| `search3-seeded` | 3-sample search, seed-matched | `L0:merge4/L3:merge4/L4:merge4/L5:frame/L8:merge4@sagepp~L4=sage.pvh16/L15=sage.pvh16` |
| `earlier` | earlier schedule point | `L0:merge4/L1:merge4/L3:merge4/L4:frame/L5:frame/L8:merge4@sagepp~L15=sage.pvh16` |
| `sage2` | uniform SageAttention2 | `dense@sage` |
| `sage2pp` | uniform SageAttention2++ | `dense@sagepp` |
| `pvh16` | uniform 16-bit P·V | `dense@sage.pvh16` |
| `avggt-split` | AVGGT/GPA-style layer split: engine approximation of the paper's bf16 PyTorch split (its `merge2` layers run on the quantised kernel), not comparable to Supp. Fig. 5 | `avggt@bf16` |
| `avggt-pub` | AVGGT, published layers (in-engine control) | frame on 0–8, `merge2` on 10–23 (non-register), RAWATTA's precision |
| `avggt-pub-frame` | AVGGT, frame half only | frame on 0–8 (non-register), rest dense, RAWATTA's precision |
| `gpa-fast` | GPA-style merge on RAWATTA-Fast's layers | `merge2` on 0, 3, 4, 5, 19, RAWATTA's precision |

## Not available

These need a flexible attention operator that this engine does not have, and are rejected with an
error: exact GPA-VGGT (shared cross-frame keys and values, r = 0.5 bipartite merging),
block-sparse top-k or CDF global attention (SparseVGGT style), FastVGGT token merging, and the
SageAttention v1 precision (INT8 QKᵀ with FP16 P·V). The paper's FP8 rounded-P row-sum ablation
(`dense@sage.pq`) is not available in this release either.
