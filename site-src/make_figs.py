# SPDX-License-Identifier: Apache-2.0
"""Figures of the RAWATTA project page (teaser, anatomy, scaling) and its social preview image.

This is the paper's figure script with these changes: it reads the small data files in site-src/data/
instead of the full results files, writes SVG and PNG into docs/static/img/ instead of PDF, draws the scaling legend
in two rows (the one-row legend is clipped in the paper PDF), and also writes docs/static/img/og.png (1200x630) from
the teaser. The plotting code is otherwise the paper's. The data files hold the paper's values rounded from the
paper's sources (latencies to 0.1 ms, delta AUC to 1e-4, attention statistics to 1e-6), which does not change the
figures visibly.

Run (any machine with Python 3.12, see requirements.txt):
    python -I site-src/make_figs.py [DATA_DIR] [OUT_DIR]
Defaults: DATA_DIR = site-src/data, OUT_DIR = docs/static/img (both relative to the repository root).

Data (site-src/data/, all from the paper's logged runs):
  regimes.json      per-block normalised entropy and frame-0 mass of VGGT-Omega at N = 10/50/200, regime labels at N = 10
  pareto.json       worst-dataset delta AUC@3/@30 against Official VGGT-Omega and N = 50 latency (RTX 4070 SUPER)
  sensitivity.json  single-layer delta AUC@3 on the held-out datasets (ETH3D, 7-Scenes)
  scaling.csv       latency vs frames, Official VGGT-Omega and the exact engine (RTX 4070 SUPER)
RTX PRO 6000 Blackwell latencies and the RAWATTA / RAWATTA-Fast end points are hard-coded below (paper supp. Tab. s-gpu2, Tab. 1).
"""
import argparse, csv, json, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
_ap.add_argument("data_dir", nargs="?", default=os.path.join(ROOT, "site-src", "data"), metavar="DATA_DIR",
                 help="folder with regimes.json, pareto.json, sensitivity.json, scaling.csv (default: site-src/data)")
_ap.add_argument("out_dir", nargs="?", default=os.path.join(ROOT, "docs", "static", "img"), metavar="OUT_DIR",
                 help="where the SVG/PNG figures and og.png go (default: docs/static/img)")
_args = _ap.parse_args()
DATA, OUT = _args.data_dir, _args.out_dir
for _f in ("regimes.json", "pareto.json", "sensitivity.json", "scaling.csv"):
    if not os.path.isfile(os.path.join(DATA, _f)):
        _ap.error(f"{os.path.join(DATA, _f)} not found (DATA_DIR must hold the four data files)")
os.makedirs(OUT, exist_ok=True)

# plotting libraries only after the arguments are checked, so --help works without them
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                     "legend.fontsize": 6.5, "axes.linewidth": 0.6, "pdf.fonttype": 42, "font.family": "DejaVu Sans",
                     "svg.fonttype": "path", "svg.hashsalt": "rawatta"})
SVG_META = {"Date": None, "Creator": None}
PNG_META = {"Software": None}


def save(fig, name, dpi, **kw):
    """SVG (deterministic ids, no date) plus a PNG fallback."""
    fig.savefig(os.path.join(OUT, name + ".svg"), metadata=SVG_META, **kw)
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=dpi, metadata=PNG_META, **kw)


# one colour per method in every figure
C = {"off": "#555555", "exact": "#2c7fb8", "rawatta": "#d95f0e", "fast": "#7a0177", "kern": "#999999",
     "sparse": "#1b9e77", "gpa": "#66a61e", "fastvggt": "#b8860b", "avggt": "#8c510a"}
R = {"context": "#9ecae1", "matching": "#cb181d", "anchoring": "#fec44f", "late": "#deebf7", "reg": "#d9d9d9"}
S = json.load(open(os.path.join(DATA, "regimes.json")))


def regimes(key, nblocks, reg=()):
    lab = S["labels"][key]; out = []; seen_match = False
    for b in range(nblocks):
        if b in reg: out.append("reg"); continue
        l = lab.get(str(b))
        if l == "matching": seen_match = True; out.append("matching")
        elif l == "anchoring": seen_match = True; out.append("anchoring")
        else: out.append("late" if seen_match else "context")
    return out


# ---------------- Pareto data (delta vs Official on the same frames; worst-dataset mean over seeds where 3 seeds ran,
# seed 0 otherwise; N=50 kitchen latency, one timing session)
par = json.load(open(os.path.join(DATA, "pareto.json")))["methods"]


def pt(m, c, mk, filled, g):
    v = par[m]; w3 = v["worst3"]
    return (v["latency_ms"], w3["mean"], w3["mean30"], c, mk, filled, g)


pts = {  # name: (ms, worst d3, worst d30, colour, marker, filled, group: main = settings and baselines of this work, prior = other methods)
    "Official VGGT-$\\Omega$": pt("Official VGGT-Omega", C["off"], "o", True, "main"),
    "Exact engine": pt("Exact engine", C["exact"], "o", True, "main"),
    "RAWATTA": pt("RAWATTA", C["rawatta"], "o", True, "main"),
    "RAWATTA-Fast": pt("RAWATTA-Fast", C["fast"], "o", True, "main"),
    "threshold schedule": pt("threshold schedule", C["fast"], "s", False, "main"),
    "search, budget $-2$": pt("greedy search, budget -2", C["fast"], "D", False, "main"),
    "SageAttn2, uniform": pt("SageAttention2, uniform", C["kern"], "D", False, "prior"),
    # seed 0 (uniform SageAttention2++, numeric-format study)
    "SageAttn2++, uniform": (1712.7, -1.25, -1.03, C["kern"], "D", True, "prior"),
    "SparseVGGT": pt("SparseVGGT, default", C["sparse"], "P", True, "prior"),
    "SparseVGGT, 0.90": pt("SparseVGGT, CDF 0.90", C["sparse"], "P", False, "prior"),
    "GPA-VGGT": pt("GPA-VGGT, default", C["gpa"], "X", True, "prior"),
    "GPA-VGGT, $r{=}0.5$": pt("GPA-VGGT, r=0.5 on 9-23", C["gpa"], "X", False, "prior"),
    "GPA-style, Fast layers": pt("GPA-style merge, RAWATTA-Fast layers", C["gpa"], "p", False, "prior"),
    "FastVGGT": pt("FastVGGT", C["fastvggt"], "v", True, "prior"),
    "AVGGT/GPA split": pt("AVGGT/GPA layer split, RAWATTA precision", C["avggt"], "<", True, "prior"),
    "frame 0–8 only": pt("frame 0-8 only, RAWATTA precision", C["avggt"], "<", False, "prior"),
}
# N=50 latencies from the paper's Tab. 1 (timing sessions)
for _k, _ms in {"Exact engine": 2237.0, "RAWATTA": 1734.0, "RAWATTA-Fast": 1616.0, "threshold schedule": 1544.0, "search, budget $-2$": 1547.0}.items():
    pts[_k] = (_ms,) + pts[_k][1:]


def pareto_panel(fig, gs, idx, ylab, top_lim, bot_lim, title):
    top = fig.add_subplot(gs[0, idx]); bot = fig.add_subplot(gs[1, idx], sharex=top)
    for name, v in pts.items():
        ms, d3, d30, c, mk, filled, g = v; y = d3 if idx == 1 else d30
        for ax in (top, bot):
            ax.scatter(ms / 1000, y, s=22 if g == "main" else 18, marker=mk, facecolors=c if filled else "white", edgecolors=c, linewidths=0.9, zorder=5 if (g == "main" and filled) else 3)
    top.set_ylim(*top_lim); bot.set_ylim(*bot_lim)
    top.spines["bottom"].set_visible(False); bot.spines["top"].set_visible(False)
    top.tick_params(bottom=False, labelbottom=False)
    d = .012; kw = dict(color="k", clip_on=False, lw=0.6)
    top.plot((-d, +d), (-d * 3, +d * 3), transform=top.transAxes, **kw); bot.plot((-d, +d), (1 - d * 9, 1 + d * 9), transform=bot.transAxes, **kw)
    for ax in (top, bot): ax.grid(alpha=0.3, lw=0.4); ax.set_xlim(1.25, 3.8); ax.set_xticks([1.5, 2, 2.5, 3, 3.5])
    bot.set_xlabel("latency, 50 frames (s)")
    top.set_ylabel(ylab); top.set_title(title, loc="left", pad=2)
    return top, bot


fig = plt.figure(figsize=(6.9, 2.75))
gs = fig.add_gridspec(2, 3, height_ratios=[3.2, 1], width_ratios=[1.25, 1, 1], hspace=0.07, wspace=0.42, left=0.02, right=0.99, top=0.92, bottom=0.43)
# (a) regime atlas
ax = fig.add_subplot(gs[:, 0])
models = [("VGGT-$\\Omega$", "omega@10", 24, (2, 6, 9, 14, 20)), ("VGGT-1B", "vggt1b@10", 24, ()), ("$\\pi^3$", "pi3@10", 18, ())]
for row, (nm, key, nb, reg) in enumerate(models):
    rg = regimes(key, nb, reg)
    for b, r in enumerate(rg):
        x0 = b / nb
        ax.add_patch(plt.Rectangle((x0, -row - 0.38), 1 / nb, 0.76, color=R[r], ec="white", lw=0.4))
        if r == "reg": ax.text(x0 + 0.5 / nb, -row, "R", ha="center", va="center", fontsize=4.8, color="#777")
    ax.text(-0.02, -row, nm, ha="right", va="center", fontsize=7)
ax.set_xlim(-0.33, 1.0); ax.set_ylim(-2.85, 0.45); ax.axis("off")
ax.annotate("", xy=(1.0, -2.5), xytext=(0.0, -2.5), arrowprops=dict(arrowstyle="->", lw=0.6, color="#555")); ax.text(0.5, -2.62, "layer (global blocks, first to last)", fontsize=6.5, color="#555", ha="center", va="top")
ax.set_title("(a) three regimes of global attention", loc="left", pad=2, x=0.0)
ax.legend(handles=[Patch(color=R["context"], label="context"), Patch(color=R["matching"], label="matching"),
                   Patch(color=R["anchoring"], label="anchoring"), Patch(color=R["late"], label="late context"),
                   Patch(color=R["reg"], label="register-only (R)")], ncol=2, loc="upper left", bbox_to_anchor=(-0.02, -0.06),
          frameon=False, handlelength=1.0, columnspacing=0.5, fontsize=6.5, handletextpad=0.3)
pareto_panel(fig, gs, 1, "worst-dataset $\\Delta$AUC@3", (-5.0, 0.4), (-40, -18), "(b) same GPU: pose AUC@3")
pareto_panel(fig, gs, 2, "worst-dataset $\\Delta$AUC@30", (-2.1, 0.15), (-31, -7), "(c) same GPU: pose AUC@30")
handles = [Line2D([], [], ls="", marker=v[4], markersize=4.2, markerfacecolor=v[3] if v[5] else "white", markeredgecolor=v[3], label=k) for k, v in pts.items()]
fig.legend(handles=handles, ncol=4, loc="lower left", bbox_to_anchor=(0.32, -0.02), frameon=False, fontsize=6.3, columnspacing=0.45, handletextpad=0.2, labelspacing=0.2)
save(fig, "teaser", 220); plt.close(fig)

# ---------------- Fig: VGGT-Omega anatomy + single-layer sensitivity
T = S["tables"]; reg = {2, 6, 9, 14, 20}
fig, (a, b) = plt.subplots(2, 1, figsize=(6.9, 2.75), gridspec_kw=dict(height_ratios=[1.25, 1], hspace=0.12), sharex=True)
rg = regimes("omega@10", 24, reg)
for bl, r in enumerate(rg): a.axvspan(bl - .5, bl + .5, color=R[r], alpha=0.55 if r != "reg" else 0.8, lw=0)
for bl in reg: a.text(bl, 0.06, "R", ha="center", fontsize=6, color="#666")
a2 = a.twinx()
for N, ls in (("10", "-"), ("50", "--"), ("200", ":")):
    tb = T[f"omega@{N}"]["blocks"]; xs = sorted(int(k) for k in tb)
    a.plot(xs, [tb[str(x)]["ent"] for x in xs], ls, color="k", lw=0.9, marker="o", ms=1.8)
    a2.plot(xs, [tb[str(x)]["f0"] for x in xs], ls, color="#b2182b", lw=0.9, marker="s", ms=1.8)
a.set_ylim(0, 1.05); a.set_ylabel("normalised entropy"); a2.set_ylim(0, 0.7); a2.set_ylabel("mass on frame 0", color="#b2182b")
a2.tick_params(axis="y", colors="#b2182b")
a.legend(handles=[Line2D([], [], color="k", ls=ls, lw=0.9, label=f"$N$={n}") for n, ls in (("10", "-"), ("50", "--"), ("200", ":"))] +
         [Line2D([], [], color="k", marker="o", ms=2.5, ls="", label="entropy"), Line2D([], [], color="#b2182b", marker="s", ms=2.5, ls="", label="frame-0 mass")] +
         [Patch(color=R[k], alpha=.6, label=l) for k, l in (("context", "context"), ("matching", "matching"), ("anchoring", "anchoring"), ("late", "late context"))],
         ncol=9, loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, fontsize=6, columnspacing=0.8, handlelength=1.6)
sens = json.load(open(os.path.join(DATA, "sensitivity.json")))
ops = ["frame", "merge2", "merge4", "merge8"]
cmap = LinearSegmentedColormap.from_list("d", ["#d7301f", "#fdd0a2", "#f0f0f0", "#c6dbef", "#2171b5"])
M = np.full((4, 24), np.nan)
for bl in range(24):
    for i, o in enumerate(ops):
        r = sens["held"].get(f"{bl}:{o}")
        if r is not None and bl not in reg: M[i, bl] = r["mean3"]
im = b.imshow(np.clip(M, -10, 10), cmap=cmap, vmin=-10, vmax=10, aspect="auto")
for bl in reg:
    b.add_patch(plt.Rectangle((bl - .5, -.5), 1, 4, color=R["reg"]))
    for i in range(4): b.text(bl, i, "–", ha="center", va="center", fontsize=6, color="#666")
for i in range(4):
    for bl in range(24):
        if not np.isnan(M[i, bl]) and M[i, bl] <= -1.5: b.text(bl, i, f"{M[i,bl]:.0f}", ha="center", va="center", fontsize=6)
b.set_yticks(range(4)); b.set_yticklabels(["frame", "merge 2×2", "merge 4×4", "merge 8×8"])
b.set_xticks(range(24)); b.set_xlabel("global block of VGGT-$\\Omega$")
cb = fig.colorbar(im, ax=[a, b], fraction=0.015, pad=0.09); cb.set_label("single-layer $\\Delta$AUC@3", fontsize=6.5)
save(fig, "anatomy", 220, bbox_inches="tight"); plt.close(fig)

# ---------------- Fig: latency vs frames on two GPU classes
rows = list(csv.DictReader(open(os.path.join(DATA, "scaling.csv"))))


def curve(cfg):
    d = {int(r["N"]): float(r["median_ms"]) for r in rows if r["config"] == cfg and r["median_ms"] not in ("", "nan") and not r["error"]}
    return sorted(d.items())


# web version: legend in two rows (the single-row legend of the paper PDF is clipped at both ends), so the figure is a bit taller
fig, (a, b) = plt.subplots(1, 2, figsize=(3.35, 2.25), sharey=True)
xs, ys = zip(*curve("official"))  # last point: 90 frames, 8.36 s (largest N that fits)
a.plot(xs, np.array(ys) / 1000, "-o", ms=2.4, color=C["off"], lw=1)
a.plot([90], [8.36], marker="X", ms=6, color="#cb181d", ls="")
a.annotate("max. 90\n(OOM at 100)", (90, 8.36), xytext=(-6, 0), textcoords="offset points", ha="right", va="center", fontsize=6, color="#cb181d", linespacing=0.95)
xs, ys = zip(*curve("exact")); a.plot(xs, np.array(ys) / 1000, "-o", ms=2.4, color=C["exact"], lw=1)
a.plot([50, 200], [1.738, 12.511], "-o", ms=2.6, color=C["rawatta"], lw=1)
a.plot([50, 200], [1.617, 10.556], "-o", ms=2.6, color=C["fast"], lw=1)
a.set_title("(a) RTX 4070 SUPER", loc="left", pad=2)
bw = {"off": ([2, 10, 50, 100, 200], [33.4, 112.9, 826.6, 2130.3, 6274.9]), "exact": ([2, 10, 50, 100, 200], [14.6, 67.0, 585.2, 1773.4, 5953.6]),
      "rawatta": ([2, 10, 50, 100, 200], [14.7, 62.5, 443.4, 1184.6, 3563.4])}
for k, (x, y) in bw.items(): b.plot(x, np.array(y) / 1000, "-o", ms=2.4, color=C[k], lw=1)
b.set_title("(b) RTX PRO 6000", loc="left", pad=2)
for ax in (a, b):
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("frames $N$"); ax.grid(alpha=0.3, lw=0.4, which="major")
    ax.set_xticks([2, 10, 50, 200]); ax.set_xticklabels(["2", "10", "50", "200"]); ax.set_ylim(0.01, 40); ax.set_xlim(1.6, 260)
    ax.set_yticks([0.01, 0.1, 1, 10]); ax.set_yticklabels(["0.01", "0.1", "1", "10"]); ax.minorticks_off()
a.set_ylabel("latency (s)")
fig.legend(handles=[Line2D([], [], color=C[k], marker="o", ms=2.5, lw=1, label=l) for k, l in
                    (("off", "Official VGGT-$\\Omega$"), ("exact", "Exact engine"), ("rawatta", "RAWATTA"), ("fast", "RAWATTA-Fast (4070 only)"))],
           ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=6.3, columnspacing=0.7, handlelength=1.4)
fig.tight_layout(rect=(0, 0, 1, 0.84), w_pad=0.4); save(fig, "scaling", 300); plt.close(fig)

# ---------------- Social preview (OpenGraph / Twitter card): 1200x630 px, title over the teaser
fig = plt.figure(figsize=(12, 6.3), dpi=100, facecolor="white")
fig.text(0.5, 0.925, "RAWATTA", ha="center", va="center", fontsize=40, weight="bold", color="#1a1a1a")
fig.text(0.5, 0.835, "Training-Free Acceleration of Multi-View 3D Transformers", ha="center", va="center", fontsize=22, color="#1a1a1a")
fig.text(0.5, 0.775, "via Regime-Aware Attention Approximation", ha="center", va="center", fontsize=22, color="#1a1a1a")
ax = fig.add_axes([0.02, 0.0, 0.96, 0.73]); ax.axis("off")
ax.imshow(mpimg.imread(os.path.join(OUT, "teaser.png")), interpolation="lanczos")
fig.savefig(os.path.join(OUT, "og.png"), dpi=100, facecolor="white", metadata=PNG_META); plt.close(fig)
print("ok:", ", ".join(sorted(f for f in os.listdir(OUT) if f.endswith((".svg", ".png")))))
