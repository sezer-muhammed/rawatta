<!-- SPDX-License-Identifier: Apache-2.0 -->
# Research harnesses (unsupported)

These are the PyTorch harnesses the paper used for its generality experiments on **VGGT-1B** and
**π³**, and the runner for official PyTorch code, extracted from the authors' research codebase.
They are kept **as used for the paper**: the algorithms, defaults and record formats are
unchanged; only container names and host paths were replaced by options and environment
variables, upstream imports were deferred so that `--help` works without them, and the
calibration maps these scripts read by file name get a documented fallback (the recovered lists are in
`eval/manifests/paper/calibration.json`). They are not part of the supported
tool set (that is `tools/`, `eval/`, `calibration/` and the `rawatta` binary). The only test
(`research/tests/test_help.py`) checks that every script and subcommand prints its `--help`
without the upstream packages; the harnesses were not re-run for this release.

None of the models' code is vendored. `setup.sh` clones the pinned repositories (GPA-VGGT,
sparse-vggt with its `vggt`, `SpargeAttn` and `Pi3` submodules) and fetches the official VGGT-Ω
evaluator; check each repository's and model's licence before use. The replacement attention
forward in `vggt1b_sched.py` and `v3_generality.py` follows the structure of upstream
facebookresearch/vggt's `Attention.forward`.

## Files and what they produced

| file | role in the authors' research codebase | produced (in the paper) |
|---|---|---|
| `vggt1b_sched.py` | VGGT-1B generality harness | supplementary section "Generality: VGGT-1B": SageAttention2 alone on VGGT-1B, the earlier-protocol safe-budget schedule, the 50-frame PyTorch timings; the VGGT-1B dense bf16 and Sage-only seed-0 cells of Supp. Tab. 19. Its `official` subcommand ran `run_official.py` (pose-only loader) for the GPA-VGGT and SparseVGGT comparison on VGGT-1B in that paragraph. |
| `v3_generality.py` | π³ and threshold-schedule generality harness | π³ dense bf16 and Sage-only rows, and the threshold-schedule rows of both models (VGGT-Ω's thresholds on each model's statistics) in Supp. Tab. 19 and the corresponding sentences of main-text Sec. 5 ("Do the regimes and the method transfer to other models?"); also the model/operator library of `v31gen_search.py` and the held-out evaluation of its searched schedules |
| `v31gen_search.py` | greedy-search generality harness | the greedy schedule search with VGGT-Ω's protocol on VGGT-1B and π³: the searched-schedule rows of Supp. Tab. 19 and the search sentence of Sec. 5 |
| `run_official.py` | official-code runner | official PyTorch code runner: `--method dense|gpa|sparse` (upstream VGGT-1B, GPA-VGGT, sparse-vggt) for the VGGT-1B comparisons; `--method omega` (official VGGT-Ω `eval/model.py`) for Official-VGGT-Ω reference rows. The supported equivalent of `--method omega` is `eval/official/time_official.py`. |
| `tables_v3gen.py`, `tables_v31gen.py` | the analysis scripts of the two generality harnesses above | the markdown tables behind Supp. Tab. 19, from `results.jsonl` |
| `setup.sh`, `run.sh` | the setup and run scripts of these experiments | environment: pinned clones + SpargeAttn sm_89 wheel; containerised run with optional GPU lock |
| `export_maps.py` | (new) | writes the evaluation frame maps from `eval/manifests/` for `--frames` |

## Not included

- **The PyTorch emulation of VGGT-Ω schedules (single-layer sensitivity in PyTorch, early fixed
  schedules, precision probes; Supp. Fig. 5, Supp. Secs. E and K) is excluded.** It depended on a shared library module of the
  authors' research codebase that was lost. For the same reason the PyTorch attention-statistics
  harness (the regime statistics behind Fig. 2 and the supplementary regime plots) and the first
  multi-domain schedule search are not included; `calibration/calibrate.py` re-implements that search's
  algorithm on the native engine instead.
- The rival ports onto VGGT-Ω (SparseVGGT†, GPA-VGGT†, FastVGGT†, chunking in Tab. 4
  and the memory study) and the top-k routing control are not part of this release.

## How to run

```sh
research/setup.sh --vggt1b --pi3                      # clones, SpargeAttn wheel, evaluator, checkpoints
python3 research/export_maps.py --out $DATA/_frames   # evaluation frame maps from eval/manifests
export RAWATTA_DATA=$DATA

# VGGT-1B: dense, Sage-only and a schedule on CO3Dv2 (seed 0), poses only
research/run.sh --lock /path/to/gpu.lock vggt1b_sched.py eval --checkpoint /rh/ckpt/model.pt \
    --configs "dense@bf16;dense@sage" --data-root /data/co3d --frames /data/_frames/co3d_eval_s0.json \
    --dataset co3d --no-depth

# π3: the threshold schedule (merge variant) at Sage precision
research/run.sh --lock /path/to/gpu.lock v3_generality.py eval --model pi3 --checkpoint /rh/ckpt/pi3.safetensors \
    --configs "dense@bf16;dense@sage;pi3-rule-b" --data-root /data/7scenes \
    --frames /data/_frames/7scenes_s0.json --dataset 7scenes --no-depth --seed 0

# the greedy search (stops after --minutes with exit code 3; re-run until it prints the schedule)
research/run.sh --lock /path/to/gpu.lock v31gen_search.py --model vggt1b --checkpoint /rh/ckpt/model.pt \
    --state /results/v31gen/vggt1b.json --data /data

python3 research/tables_v31gen.py runs/research/results.jsonl
```

Every run appends one JSON record per config to `runs/research/results.jsonl` (`$RESEARCH_OUT`).
Configs are `<schedule>@<bf16|sage>[+fp8]` with schedule `dense` or `L<i>:<frame|merge2|merge4|merge4nx>,...`
(`merge4nx`: all frames merged, for π³), separated by `;` on the command line; `+fp8` is a simulated
FP8 quantise-dequantise of the linear layers (accuracy only, never timed).

Known differences from the paper's runs: these scripts read the RE10K evaluation subset (`re10k_eval_v2`,
38 sequences) and the searches' CO3Dv2/RE10K calibration maps (`co3d_train.json`, `re10k_train.json`) by file
name. The same lists are released in `eval/manifests/paper/` (see `eval/manifests/README.md`) but in the
supported tools' format, so export them to those file names before running these scripts. `v31gen_search.py` falls back to the banana/donut sequences of the CO3Dv2
manifest and the 10 RE10K calibration sequences of `eval/manifests/splits.json` when the files are not
in `--cal-maps`.
