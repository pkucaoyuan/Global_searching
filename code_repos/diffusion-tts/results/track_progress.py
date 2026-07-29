#!/usr/bin/env python3
"""Rebuttal experiment tracker: parse run.logs, compare running scores vs plan expectations."""
import re, glob, os, statistics, datetime

DAY0 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "day0")

# (expected_lo, expected_hi, target_images, note) — anchors from rebuttal plan §2
EXPECT = {
    "e1_sd_eps1_brightness_400":       (0.7108, 0.7208, 200, "校验点0.7158; U=0.7025 G=0.7248"),
    "e1_sd_eps1_brightness_500":       (0.714, 0.734, 200, "预期0.724±.010; U=0.7094 G=0.7346"),
    "e1_sd_eps1_brightness_800":       (0.760, 0.780, 200, "预期0.770±.010; U=0.7511 G=0.7832"),
    "e1_sd_eps1_compressibility_400":  (0.8823, 0.8923, 200, "校验点0.8873; U=0.8833 G=0.8946"),
    "e1_sd_eps1_compressibility_500":  (0.884, 0.894, 200, "预期0.889±.005; U=0.8825 G=0.9004"),
    "e1_sd_eps1_compressibility_800":  (0.888, 0.898, 200, "预期0.893±.005; U=0.8892 G=0.9008"),
    "e1_edm_eps1_brightness_144":      (0.969, 0.985, 20, "预期0.977±.008; U=0.9507 G=0.9887"),
    "e1_edm_eps1_brightness_180":      (0.975, 0.989, 20, "预期0.982±.007; U=0.9617 G=0.9904"),
    "e1_edm_eps1_brightness_288":      (0.989, 0.997, 20, "预期0.993±.004; U=0.9787 G=0.9988"),
    "e1_edm_eps1_compressibility_144": (0.674, 0.686, 20, "预期0.680±.006; U=0.6714 G=0.6845"),
    "e1_edm_eps1_compressibility_180": (0.687, 0.699, 20, "预期0.693±.006; U=0.6774 G=0.7003"),
    "e1_edm_eps1_compressibility_288": (0.703, 0.715, 20, "预期0.709±.006; U=0.6901 G=0.7165"),
    "e6_sd_gainonly_brightness_400":   (0.720, 0.723, 200, "offline0.7158+gain信号(~60-70%增量)"),
    "e6_sd_varonly_brightness_400":    (0.718, 0.721, 200, "offline0.7158+var信号(~40-50%增量)"),
    "e6_sd_gainonly_compressibility_400": (0.891, 0.893, 200, "offline0.8873+gain信号"),
    "e6_sd_varonly_compressibility_400":  (0.890, 0.892, 200, "offline0.8873+var信号"),
    "e4_db_uniform_brightness_400":    (None, None, 300, "DrawBench基线,绝对值仅参考"),
    "e4_db_eps1_brightness_400":       (None, None, 300, "看与uniform差:预期为正"),
    "e4_db_online_brightness_400":     (None, None, 300, "GAINS-U预期+0.015~0.030"),
    "e4_db_uniform_compressibility_400": (None, None, 300, "DrawBench基线"),
    "e4_db_eps1_compressibility_400":  (None, None, 300, "看与uniform差:预期为正"),
    "e4_db_online_compressibility_400": (None, None, 300, "GAINS-U预期+0.008~0.018"),
    "e2_sd_rbf_brightness_400":        (0.705, 0.725, 200, "预期0.710-0.720,应<GAINS 0.7248"),
    "e2_sd_vt_brightness_400":         (0.700, 0.720, 200, "预期0.705-0.715"),
    "e2_sd_rbf_compressibility_400":   (0.883, 0.895, 200, "预期0.885-0.892,应<GAINS 0.8946"),
    "e2_sd_vt_compressibility_400":    (0.882, 0.893, 200, "预期0.884-0.890"),
    "e1_sd_uniform_brightness_400":    (None, None, 200, "自洽锚:与旧表U 0.7025对照"),
    "e1_sd_uniform_compressibility_400": (None, None, 200, "自洽锚(新C刻度)"),
    "e1_sd_gains_brightness_400":      (0.710, 0.735, 200, "K1=11严格400NFE;旧表G=0.7248"),
    "e1_sd_gains_compressibility_400": (None, None, 200, "K1=11;应>U-C 0.7971和offline 0.8096"),
    "e2_sd_rbffs_brightness_400":      (None, None, 200, "RBF强制花满400;应>RBF 0.6138"),
    "e2_sd_rbffs_compressibility_400": (None, None, 200, "RBF-FS;RBF已0.8022"),
    "e2_sd_vtfs_brightness_400":       (None, None, 200, "VT强制花满;应>VT 0.6836"),
    "e2_sd_vtfs_compressibility_400":  (None, None, 200, "VT-FS;VT 238NFE时0.7742"),
    "e3b_sd_naive_imagereward":        (None, None, 200, "IR基线(50 NFE)"),
    "e3b_sd_uniform_imagereward_400":  (None, None, 200, "IR@400 uniform"),
    "e3b_sd_eps1_imagereward_400":     (None, None, 200, "IR offline;预期U+0.02-0.05"),
    "e3b_sd_gains_imagereward_400":    (None, None, 200, "IR GAINS;预期U+0.04-0.10"),
    "e5_sd_window_early_brightness_400": (None, None, 200, "早窗基线;预期0.710-0.718<offline"),
    "e5_sd_window_early_compressibility_400": (None, None, 200, "早窗C"),
}

score_re = re.compile(r"(?:^|\n)Score: ([-\d.]+)")           # SD per-image
edm_re = re.compile(r"(?:^|\n)Average score: ([-\d.]+)")      # EDM per-run (36-img grid)

def parse_log(path):
    try:
        text = open(path, errors="ignore").read().replace("\r", "\n")
    except FileNotFoundError:
        return []
    scores = [float(x) for x in score_re.findall(text)]
    if not scores:
        scores = [float(x) for x in edm_re.findall(text)]
    return scores

def main():
    rows = []
    alerts = []
    for d in sorted(glob.glob(os.path.join(DAY0, "*", ""))):
        cfg = os.path.basename(d.rstrip("/"))
        scores = parse_log(os.path.join(d, "run.log"))
        lo, hi, target, note = EXPECT.get(cfg, (None, None, None, ""))
        n = len(scores)
        if n == 0:
            rows.append((cfg, 0, target, "-", "-", "启动中/无逐图分", note)); continue
        mean = statistics.mean(scores)
        std = statistics.stdev(scores) if n > 1 else 0.0
        status = "跑中"
        if target and n >= target: status = "完成"
        flag = ""
        judge_n = 15 if target == 20 else 150  # EDM judges at 15 runs; SD only near-complete (prompt heterogeneity)
        if lo is not None and n >= judge_n:
            if mean < lo: flag = "⚠低于预期"
            elif mean > hi: flag = "高于预期"
            else: flag = "符合"
            if mean < lo:
                alerts.append(f"{cfg}: n={n} mean={mean:.4f} < 预期下界{lo} ({note})")
        rows.append((cfg, n, target, f"{mean:.4f}", f"{std:.4f}", f"{status} {flag}".strip(), note))

    now = datetime.datetime.now().strftime("%F %T")
    out = [f"# Day0 实验状态  ({now})", "",
           "| config | 图数/目标 | mean | std | 状态 | 预期参照 |", "|---|---|---|---|---|---|"]
    for cfg, n, target, mean, std, status, note in rows:
        out.append(f"| {cfg} | {n}/{target or '?'} | {mean} | {std} | {status} | {note} |")
    out.append("")
    if alerts:
        out.append("## ⚠ 偏离告警")
        out += [f"- {a}" for a in alerts]
    text = "\n".join(out)
    with open(os.path.join(DAY0, "..", "tracker_status.md"), "w") as f:
        f.write(text + "\n")
    print(text)

if __name__ == "__main__":
    main()
