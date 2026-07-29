#!/usr/bin/env python3
"""Water-filling K_t from a --log_gain calibration run.

Sensitivity proxy per step t: mean over images of (sum of positive per-iter gains
+ std of per-iter gains) -- captures both achievable improvement and dispersion.
Allocation: K_t = 1 + waterfill(budget - T) proportional to sensitivity.
Usage: make_ir_quota.py <calib_run.log> <out.json> <num_steps> <budget>
"""
import re, sys, json
import statistics as st

log, out, T, B = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
text = open(log, errors="ignore").read().replace("\r", "\n")
per_step = [[] for _ in range(T)]
for line in text.splitlines():
    m = re.search(r"\[SD\]\[EPS_GREEDY\] step (\d+): K_used=\d+, gains=\[([^\]]*)\]", line)
    if m:
        s = int(m.group(1))
        gains = [float(x) for x in m.group(2).split(",") if x.strip()]
        if s < T and gains:
            pos = sum(g for g in gains if g > 0)
            disp = st.stdev(gains) if len(gains) > 1 else 0.0
            per_step[s].append(pos + disp)
n_img = min(len(v) for v in per_step) if all(per_step) else 0
if n_img == 0:
    sys.exit("no complete images parsed")
sens = [st.mean(v) for v in per_step]
tot = sum(sens)
alloc = B - T  # 1 NFE floor per step
raw = [1 + alloc * s / tot for s in sens]
K = [int(x) for x in raw]
# distribute remainder to largest fractional parts
rem = B - sum(K)
order = sorted(range(T), key=lambda i: raw[i] - K[i], reverse=True)
for i in order[:rem]:
    K[i] += 1
json.dump(K, open(out, "w"))
q = lambda a, b: sum(K[a:b])
print(f"images={n_img} total={sum(K)} | pool: steps0-9={q(0,10)} 10-19={q(10,20)} 20-29={q(20,30)} 30-39={q(30,40)} 40-49={q(40,50)}")
print("K_t:", K)
