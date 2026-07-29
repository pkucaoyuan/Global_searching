#!/usr/bin/env python3
"""Build per-step VT thresholds (median of step_best across calibration images)."""
import re, sys, json, statistics

log_path, out_path, num_steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
pat = re.compile(r"\[SD\]\[VT\] step (\d+): .*step_best=([-\d.]+)")
per_step = [[] for _ in range(num_steps)]
for line in open(log_path, errors="ignore"):
    m = pat.search(line)
    if m:
        s, v = int(m.group(1)), float(m.group(2))
        if s < num_steps:
            per_step[s].append(v)
tau = [statistics.median(v) if v else float("inf") for v in per_step]
n_imgs = min(len(v) for v in per_step) if all(per_step) else 0
json.dump(tau, open(out_path, "w"))
print(f"wrote {out_path}: {num_steps} thresholds from {n_imgs} calib images; tau[0]={tau[0]:.4f} tau[-1]={tau[-1]:.4f}")
