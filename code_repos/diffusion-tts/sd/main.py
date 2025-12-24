import os
import sys
import importlib.util
import torch
import argparse
import csv
from pathlib import Path
import numpy as np
from tqdm import tqdm
from PIL import Image
from functools import partial
import torch.multiprocessing as mp
import math

def load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, os.path.abspath(relative_path))
    sys.modules[name] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules[name])

# Lazy import diffusers
load_module("diffusers", "diffusers/src/diffusers/__init__.py")
from diffusers import StableDiffusionPipeline, DDIMScheduler
from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer


def main():
    parser = argparse.ArgumentParser(description="Stable Diffusion T2I with search methods (align args with EDM)")
    parser.add_argument('--prompt', type=str, default="YOUR PROMPT HERE", help='Text prompt (ignored if prompt_csv set)')
    parser.add_argument('--prompt_csv', type=str, default=None, help='CSV file with prompts (first column). If set, iterate prompts.')
    parser.add_argument('--output', type=str, default=None, help='Output file name (multi-prompt: will append index)')
    parser.add_argument('--scorer', type=str, choices=['brightness', 'compressibility', 'clip'], default='brightness', help='Scorer')
    parser.add_argument('--method', type=str, default='naive', help='Sampling method (naive, rejection, beam, mcts, zero_order, eps_greedy, epsilon_1)')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of denoising steps (default 50, align with SD baseline)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs; if >1 report mean±std')
    # master params
    parser.add_argument('--N', type=int, default=4, help='Master param N')
    parser.add_argument('--lambda_', type=float, default=0.15, help='Master param lambda')
    parser.add_argument('--eps', type=float, default=0.4, help='Master param eps')
    parser.add_argument('--K', type=int, default=20, help='Master param K (eps_greedy/zero_order)')
    parser.add_argument('--K1', type=int, default=25, help='epsilon_1: K for head 2 + tail 4 steps')
    parser.add_argument('--K2', type=int, default=15, help='epsilon_1: K for middle steps')
    parser.add_argument('--revert_on_negative', action='store_true', help='epsilon_1: revert pivot when gain<0 (after first iter)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    model_id = "runwayml/stable-diffusion-v1-5"
    local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    local_pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=local_scheduler,
        torch_dtype=torch.float16,
    ).to(args.device)

    scorer_map = {
        "brightness": BrightnessScorer(),
        "compressibility": CompressibilityScorer(),
        "clip": CLIPScorer(),
    }
    scorer = scorer_map[args.scorer]

    # map edm-style name to sd pipeline name
    method_map = {
        "naive": "naive",
        "rejection": "rejection",
        "beam": "beam",
        "mcts": "mcts",
        "zero_order": "zero_order",
        "eps_greedy": "eps_greedy",
        "epsilon_1": "eps_greedy_1",
    }
    if args.method not in method_map:
        raise ValueError(f"Unknown method {args.method}")
    sd_method = method_map[args.method]

    base_params = {
        'N': args.N,
        'lambda': args.lambda_,
        'eps': args.eps,
        'K': args.K,
        'K1': args.K1,
        'K2': args.K2,
        'revert_on_negative': args.revert_on_negative,
    }

    def run_once(run_idx: int, prompt_text: str, seed_base: int):
        if seed_base is not None:
            torch.manual_seed(seed_base + run_idx)
        best_result, best_score = None, float('-inf')
        # rejection runs multiple candidates; other methods single
        repeats = base_params['N'] if sd_method == "rejection" else 1
        for _ in range(repeats):
            result, score = local_pipe(
                prompt=prompt_text,
                num_inference_steps=args.num_steps,
                score_function=scorer,
                method=sd_method,
                params=base_params,
            )
            score_value = score.item() if torch.is_tensor(score) else float(score)
            if score_value > best_score:
                best_result, best_score = result, score_value
        return best_result, best_score

    # Load prompts
    prompts = []
    # Interpretation:
    # - n_runs 表示：从 CSV 取多少个 prompt；每个 prompt 只跑一次。
    # - 若未提供 prompt_csv，则只用单个 prompt，忽略 n_runs。
    n_runs_effective = 1
    if args.prompt_csv:
        if args.n_runs is None or args.n_runs <= 0:
            raise ValueError("When using prompt_csv, please set --n_runs to a positive number (prompt count).")
        with open(args.prompt_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)
            # skip header if contains 'prompt'
            if rows and rows[0] and 'prompt' in rows[0][0].lower():
                rows = rows[1:]
            prompts = [row[0] for row in rows if row]
        if not prompts:
            raise ValueError("prompt_csv provided but no prompts found")
        prompts = prompts[:args.n_runs]
        if len(prompts) < args.n_runs:
            print(f"[SD] Warning: requested n_runs={args.n_runs} prompts but CSV only has {len(prompts)}; using available prompts.")
    else:
        prompts = [args.prompt]
        n_runs_effective = 1  # 单 prompt 场景，不复用 n_runs

    def make_outname(idx: int):
        if args.output:
            stem = Path(args.output)
            return f"{stem.stem}_{idx}{stem.suffix or '.png'}"
        return f"sd_{args.method}_{args.scorer}_{idx}.png" if len(prompts) > 1 else f"sd_{args.method}_{args.scorer}.png"

    all_scores = []
    for idx, prompt_text in enumerate(prompts):
        best_result, best_score = run_once(idx, prompt_text, args.seed)
        outname = make_outname(idx)
        best_result.images[0].save(outname)
        print(f"\n[SD][{idx}] Prompt: {prompt_text}")
        print(f"[SD] Saved: {outname}\nBest score: {best_score}\n")
        all_scores.append(best_score)

    if len(prompts) > 1:
        overall_mean = float(np.mean(all_scores))
        overall_std = float(np.std(all_scores))
        print(f"[SD] Overall {len(prompts)} prompts mean score: {overall_mean:.4f} ± {overall_std:.4f}")


if __name__ == "__main__":
    main()