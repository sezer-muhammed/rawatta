#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a latency table from the timing.json files of several `rawatta run` outputs.

  python3 timing_table.py runs/cmp/exact runs/cmp/rawatta runs/cmp/rawatta-fast
  python3 timing_table.py --baseline exact runs/cmp/*        # speed-up column relative to the exact engine
  python3 timing_table.py --json runs/cmp/*                  # machine-readable rows

Only standard library. The numbers are whatever the runs measured; nothing is filled in.
"""

import argparse
import json
import os
import sys

# timing.json is written by `rawatta run`. These are the field names we look for, first match wins
# (dotted names walk into nested objects), so small schema changes do not break the table.
FIELDS = {
    "latency_ms": ("median_ms", "latency_ms.median", "latency_ms", "forward_ms.median", "forward_ms",
                   "forward.median_ms", "latency.median_ms", "timing.median_ms"),
    "iqr_ms": ("iqr_ms", "latency_ms.iqr", "forward_ms.iqr", "forward.iqr_ms", "latency.iqr_ms", "timing.iqr_ms"),
    # rawatta writes peak_device_mib (memory the engine used); card_used_peak_mib is the whole card, other
    # processes included, so it is only a last resort.
    "peak_mib": ("peak_device_mib", "peak_mib", "engine_peak_mib", "peak_memory_mib", "peak_mem_mib",
                 "memory.peak_mib", "card_used_peak_mib"),
    "peak_bytes": ("peak_bytes", "peak_memory_bytes", "memory.peak_bytes"),
    "gpu": ("gpu", "gpu_name", "device", "gpu.name", "device.name"),
    "config": ("config", "config_name", "settings"),
    "frames": ("frames", "n_frames", "num_frames"),
    "reps": ("reps", "repetitions"),
    "warmup": ("warmup", "warmup_reps"),
}


def _get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def _first(d, names, kind):
    for n in names:
        v = _get(d, n)
        if kind == "num" and isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if kind == "str" and isinstance(v, str) and v:
            return v
        if kind == "str" and isinstance(v, dict) and isinstance(v.get("name"), str):
            return v["name"]
    return None


def read_timing(path):
    """Normalise one timing.json (file or run directory) into a flat dict."""
    if os.path.isdir(path):
        path = os.path.join(path, "timing.json")
    with open(path) as f:
        d = json.load(f)
    peak = _first(d, FIELDS["peak_mib"], "num")
    if peak is None:
        b = _first(d, FIELDS["peak_bytes"], "num")
        peak = b / 1048576.0 if b is not None else None
    frames = _first(d, FIELDS["frames"], "num")
    reps = _first(d, FIELDS["reps"], "num")
    warmup = _first(d, FIELDS["warmup"], "num")
    return {
        "path": path,
        "config": _first(d, FIELDS["config"], "str"),
        "frames": int(frames) if frames is not None else None,
        "latency_ms": _first(d, FIELDS["latency_ms"], "num"),
        "iqr_ms": _first(d, FIELDS["iqr_ms"], "num"),
        "peak_mib": peak,
        "gpu": _first(d, FIELDS["gpu"], "str"),
        "reps": int(reps) if reps is not None else None,
        "warmup": int(warmup) if warmup is not None else None,
        "synthetic": d.get("synthetic") is True,
    }


def _fmt(v, spec="%.1f"):
    return "-" if v is None else (spec % v if isinstance(v, float) else str(v))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (or timing.json files)")
    ap.add_argument("--baseline", default=None, help="config name (or run dir name) used for the speed-up column")
    ap.add_argument("--json", action="store_true", help="print JSON rows instead of a table")
    a = ap.parse_args(argv)

    rows = []
    for r in a.runs:
        name = os.path.basename(os.path.normpath(r if os.path.isdir(r) else os.path.dirname(r)))
        try:
            t = read_timing(r)
        except (OSError, ValueError) as e:
            t = {"path": r, "error": str(e)}
        t["run"] = name
        rows.append(t)
    if a.json:
        print(json.dumps(rows, indent=2))
        return 0

    base = None
    if a.baseline:
        for t in rows:
            if a.baseline in (t.get("run"), t.get("config")) and t.get("latency_ms"):
                base = t["latency_ms"]
    head = ["run", "config", "frames", "reps", "median ms", "IQR ms", "peak MiB"]
    if base:
        head.append("speed-up vs %s" % a.baseline)
    head.append("GPU")
    lines = [head]
    for t in rows:
        if "error" in t:
            lines.append([t["run"], "missing timing.json"] + [""] * (len(head) - 2))
            continue
        # One timed pass has no spread: show "-" rather than an IQR of 0 that looks perfectly stable.
        iqr = t["iqr_ms"] if (t["reps"] or 2) >= 2 else None
        row = [t["run"], _fmt(t["config"]), _fmt(t["frames"]), _fmt(t["reps"]), _fmt(t["latency_ms"]), _fmt(iqr),
               _fmt(t["peak_mib"], "%.0f")]
        if base:
            row.append("%.2fx" % (base / t["latency_ms"]) if t["latency_ms"] else "-")
        row.append(_fmt(t["gpu"]))
        lines.append(row)
    widths = [max(len(str(r[i])) for r in lines) for i in range(len(head))]
    print("| " + " | ".join(h.ljust(w) for h, w in zip(head, widths)) + " |")
    print("|" + "|".join("-" * (w + 2) for w in widths) + "|")
    for r in lines[1:]:
        print("| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |")
    few = [t["run"] for t in rows if "error" not in t and (t["reps"] is None or t["reps"] < 3)]
    if few:
        print("note: %s: fewer than 3 timed passes (or unknown); rerun with --reps 10 for a stable median"
              % ", ".join(few))
    if any(t.get("synthetic") for t in rows):
        print("note: SYNTHETIC timings (tests/fake_rawatta), not measurements")
    if any("error" in t for t in rows):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
