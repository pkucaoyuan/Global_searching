import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def compute_step_gains(rows):
    """
    每个时间步收益 = (该步最后一次迭代得分) - (该步第一次迭代得分)
    返回 list 对应 timestep 顺序。
    """
    per_step = defaultdict(list)
    for r in rows:
        if r["timestep"] == "baseline":
            continue
        step = int(r["timestep"])
        it = int(r["iter"])
        score = float(r["score"])
        per_step[step].append((it, score))

    gains = []
    steps_sorted = sorted(per_step.keys())
    for step in steps_sorted:
        iters = sorted(per_step[step], key=lambda x: x[0])
        first_score = iters[0][1]
        last_score = iters[-1][1]
        gains.append(last_score - first_score)
    return steps_sorted, gains


def compute_iter_avg_gains(rows):
    """
    每次迭代收益 = 同一时间步 (迭代 n+1 得分 - 迭代 n 得分)，然后跨时间步取平均
    假设每步迭代次数一致。
    返回 (iter_index, avg_gain) 列表，其中 iter_index 对应 n（从0开始，对应“n -> n+1”的差分）。
    """
    per_step = defaultdict(list)
    for r in rows:
        if r["timestep"] == "baseline":
            continue
        step = int(r["timestep"])
        it = int(r["iter"])
        score = float(r["score"])
        per_step[step].append((it, score))

    # 取最小共同迭代次数，防止个别步缺失
    min_iters = min(len(v) for v in per_step.values())
    # 计算差分并累积
    diff_sums = [0.0] * (min_iters - 1)
    for step, iters in per_step.items():
        iters_sorted = sorted(iters, key=lambda x: x[0])[:min_iters]
        for n in range(min_iters - 1):
            diff = iters_sorted[n + 1][1] - iters_sorted[n][1]
            diff_sums[n] += diff

    step_count = len(per_step)
    avg_gains = [s / step_count for s in diff_sums]
    return list(range(len(avg_gains))), avg_gains


def plot_step_gains(steps, gains, out_path):
    plt.figure(figsize=(10, 4))
    plt.plot(steps, gains, marker="o")
    plt.title("Per-step gain (last - first)")
    plt.xlabel("Timestep")
    plt.ylabel("Gain")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[eps1_gain_plot] saved per-step gains -> {out_path}")


def plot_iter_avg_gains(iter_indices, gains, out_path):
    plt.figure(figsize=(8, 4))
    plt.bar(iter_indices, gains)
    plt.title("Avg gain per iteration (n+1 - n)")
    plt.xlabel("Iteration index (n -> n+1)")
    plt.ylabel("Average gain")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[eps1_gain_plot] saved per-iter avg gains -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot gains from eps1_gain_probe CSV.")
    parser.add_argument("--csv", required=True, help="输入的 eps1_gain_probe 结果 CSV")
    parser.add_argument(
        "--out_dir", default="outputs", help="输出目录（默认 outputs）"
    )
    parser.add_argument(
        "--step_fig", default="step_gains.png", help="时间步收益图文件名"
    )
    parser.add_argument(
        "--iter_fig", default="iter_avg_gains.png", help="迭代平均收益图文件名"
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_csv(args.csv)

    steps, step_gains = compute_step_gains(rows)
    iters, iter_gains = compute_iter_avg_gains(rows)

    plot_step_gains(steps, step_gains, out_dir / args.step_fig)
    plot_iter_avg_gains(iters, iter_gains, out_dir / args.iter_fig)


if __name__ == "__main__":
    main()

