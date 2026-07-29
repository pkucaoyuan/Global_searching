#!/usr/bin/env python3
"""Paired evaluation of online-hyperparameter pilots vs offline-only baseline (same prompt+seed)."""
import re, os, glob, statistics

D = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day0")

def per_image(cfg):
    """Return {(prompt_idx, seed): score} and mean measured NFE per image."""
    path = f"{D}/{cfg}/run.log"
    text = open(path, errors="ignore").read().replace("\r", "\n")
    scores, cur_p, cur_seed = {}, None, None
    for line in text.splitlines():
        m = re.search(r"\[SD\]\[(\d+)\]\[r\d+\] Prompt:", line)
        if m: cur_p = int(m.group(1)); continue
        m = re.search(r"\[SD\] Seed: (\d+)", line)
        if m: cur_seed = int(m.group(1)); continue
        m = re.search(r"^Score: ([\d.]+)", line.strip())
        if m and cur_p is not None and cur_seed is not None:
            scores[(cur_p, cur_seed)] = float(m.group(1))
    # measured NFE: sum K_used per image (images delimited by step 0)
    spends, cur = [], None
    for line in text.splitlines():
        m = re.search(r"EPS_GREEDY_ONLINE\] step (\d+): K_used=(\d+)", line)
        if m:
            s, k = int(m.group(1)), int(m.group(2))
            if s == 0:
                cur = [0]; spends.append(cur)
            if cur is not None: cur[0] += k
    done = [s[0] for s in spends[:-1]] if len(spends) > 1 else []
    nfe = sum(done)/len(done) if done else float("nan")
    return scores, nfe

base, _ = per_image("e1_sd_eps1_brightness_400")
print(f"baseline offline-B images parsed: {len(base)}")
for d in sorted(glob.glob(f"{D}/tune*")):
    cfg = os.path.basename(d)
    s, nfe = per_image(cfg)
    common = sorted(set(s) & set(base))
    if not common:
        print(f"{cfg}: no overlap yet (n={len(s)})"); continue
    diffs = [s[k] - base[k] for k in common]
    mean_d = statistics.mean(diffs)
    se = (statistics.stdev(diffs) / len(diffs) ** 0.5) if len(diffs) > 1 else 0
    mean_abs = statistics.mean([s[k] for k in common])
    print(f"{cfg}: n_pair={len(common)} mean={mean_abs:.4f} paired_diff={mean_d:+.4f} (SE {se:.4f}) NFE~{nfe:.0f}")
