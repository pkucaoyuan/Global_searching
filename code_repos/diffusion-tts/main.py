"""
This script allows you to generate images using either the Stable Diffusion (SD) or EDM backend,
with a choice of scorer and sampling method

Usage examples:
    python main.py --backend sd --scorer brightness --method naive --prompt "A beautiful landscape"
    python main.py --backend edm --scorer imagenet --method zero_order

Arguments:
    --backend   : 'sd' or 'edm' (required)
    --scorer    : 'brightness', 'compressibility', 'clip', or 'imagenet' (required)
    --method    : Sampling method (available: 'naive', 'rejection', 'beam', 'mcts', 'zero_order', 'eps_greedy', 'epsilon_1') (default: 'naive')
    --prompt    : Prompt for SD (default: 'A beautiful landscape')
    --output    : Output filename
    --N, --lambda_, --eps, --K, --B, --S : sampling parameters (see code for defaults)
    --seed      : Random seed (default: 0)
    --device    : Device (default: 'cuda')
"""

import os
import sys
import argparse
import importlib.util
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import importlib

# =========================
# EDM Import Helper
# =========================
def import_edm():
    """Dynamically import EDM modules and scorers."""
    edm_dir = Path(__file__).parent / 'edm'
    sys.path.insert(0, str(edm_dir))
    dnnlib = importlib.import_module('dnnlib')
    dnnlib_util = importlib.import_module('dnnlib.util')
    from scorers import BrightnessScorer, CompressibilityScorer, ImageNetScorer
    return dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer

# =========================
# SD Import Helper
# =========================
def import_sd():
    """Dynamically import SD pipeline and scorers."""
    sd_dir = Path(__file__).parent / 'sd'
    diffusers_path = sd_dir / 'diffusers' / 'src' / 'diffusers' / '__init__.py'
    spec = importlib.util.spec_from_file_location('diffusers', str(diffusers_path.resolve()))
    sys.modules['diffusers'] = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sys.modules['diffusers'])
    from diffusers import StableDiffusionPipeline, DDIMScheduler
    sys.path.insert(0, str(sd_dir))
    from scorers import BrightnessScorer, CompressibilityScorer, CLIPScorer
    return StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer

# =========================
# Scorer Factory
# =========================
def get_scorer(backend, scorer_name, BrightnessScorer, CompressibilityScorer, CLIPScorer=None, ImageNetScorer=None):
    """Return the appropriate scorer instance for the backend and scorer name."""
    if scorer_name == 'brightness':
        return BrightnessScorer(dtype=torch.float32)
    elif scorer_name == 'compressibility':
        return CompressibilityScorer(dtype=torch.float32)
    elif scorer_name == 'clip' and backend == 'sd':
        return CLIPScorer(dtype=torch.float32)
    elif scorer_name == 'imagenet' and backend == 'edm':
        return ImageNetScorer(dtype=torch.float32)
    else:
        raise ValueError(f"Unknown or invalid scorer '{scorer_name}' for backend '{backend}'")

# =========================
# Main Logic
# =========================
def main():
    # -----------
    # CLI Arguments
    # -----------
    parser = argparse.ArgumentParser(
        description='Unified Diffusion Image Generator (EDM/SD)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--backend', type=str, choices=['edm', 'sd'], required=True, help='Backend: edm or sd')
    parser.add_argument('--scorer', type=str, choices=['brightness', 'compressibility', 'clip', 'imagenet'], required=True, help='Scorer name')
    parser.add_argument('--method', type=str, default='naive', help='Sampling method (naive, rejection, beam, mcts, zero_order, eps_greedy, epsilon_1, epsilon_online)')
    parser.add_argument('--prompt', type=str, default='YOUR PROMPT HERE', help='Prompt for SD (ignored if prompt_csv set)')
    parser.add_argument('--prompt_csv', type=str, default=None, help='CSV with prompts (first column); SD only')
    parser.add_argument('--output', type=str, default=None, help='Output filename (default: auto)')
    # Master params (with SD defaults)
    parser.add_argument('--N', type=int, default=4, help='Master param N')
    parser.add_argument('--lambda_', type=float, default=0.15, help='Master param lambda')
    parser.add_argument('--eps', type=float, default=0.4, help='Master param eps')
    parser.add_argument('--K', type=int, default=20, help='Master param K')
    parser.add_argument('--K1', type=int, default=25, help='Master param K1 (epsilon_1: steps 0-1 & last 4)')
    parser.add_argument('--K2', type=int, default=15, help='Master param K2 (epsilon_1: remaining middle steps)')
    parser.add_argument('--revert_on_negative', action='store_true', help='epsilon_1: 若本迭代增益为负则保持上一轮 pivot')
    parser.add_argument('--B', type=int, default=2, help='Master param B')
    parser.add_argument('--S', type=int, default=8, help='Master param S')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', help='Device')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of denoising steps (SD only)')
    parser.add_argument('--n_runs', type=int, default=1, help='Number of runs (if >1, output mean±std as in paper)')
    parser.add_argument('--log_gain', action='store_true', help='Log per-timestep gains for EPS_GREEDY')
    parser.add_argument('--thresh_gain_coef', type=float, default=1.0, help='epsilon_online: gain threshold coefficient (hist_mean_gain / hist_mean_var * coef)')
    parser.add_argument('--thresh_var_coef', type=float, default=1.0, help='epsilon_online: variance threshold coefficient (hist_mean_gain / hist_mean_var * coef)')
    args = parser.parse_args()

    # -----------
    # Validation
    # -----------
    if args.backend == 'sd' and args.scorer == 'imagenet':
        raise ValueError('imagenet scorer is only available for edm backend')
    if args.backend == 'edm' and args.scorer == 'clip':
        raise ValueError('clip scorer is only available for sd backend')

    # -----------
    # SD Backend
    # -----------
    if args.backend == 'sd':
        if args.prompt_csv and args.backend != 'sd':
            raise ValueError('prompt_csv is only supported for sd backend')
        StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer = import_sd()
        scorer = get_scorer('sd', args.scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer)

        model_id = "runwayml/stable-diffusion-v1-5"
        local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
        local_pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            scheduler=local_scheduler,
            torch_dtype=torch.float16,
        ).to(args.device)

        # map EDM name to SD internal name
        method_map_sd = {
            'naive': 'naive',
            'rejection': 'rejection',
            'beam': 'beam',
            'mcts': 'mcts',
            'zero_order': 'zero_order',
            'eps_greedy': 'eps_greedy',
            'epsilon_1': 'eps_greedy_1',
            'epsilon_online': 'eps_greedy_online',
        }
        if args.method not in method_map_sd:
            raise ValueError(f"Unknown method for sd backend: {args.method}")
        method = method_map_sd[args.method]
        MASTER_PARAMS = {
            'N': args.N,
            'lambda': args.lambda_,
            'eps': args.eps,
            'K': args.K,
            'K1': args.K1,
            'K2': args.K2,
            'revert_on_negative': args.revert_on_negative,
            'log_gain': args.log_gain,
            'thresh_gain_coef': args.thresh_gain_coef,
            'thresh_var_coef': args.thresh_var_coef,
            'B': args.B,
            'S': args.S,
        }

        # prompt handling: if prompt_csv provided, n_runs = number of prompts to take; each prompt runs once
        prompts = []
        if args.prompt_csv:
            import csv
            with open(args.prompt_csv, 'r', encoding='utf-8') as f:
                rows = list(csv.reader(f))
            if rows and rows[0] and 'prompt' in rows[0][0].lower():
                rows = rows[1:]
            prompts = [row[0] for row in rows if row]
            if not prompts:
                raise ValueError("prompt_csv provided but no prompts found")
            prompt_count = args.n_runs if args.n_runs and args.n_runs > 0 else len(prompts)
            prompts = prompts[:prompt_count]
            repeat_per_prompt = 1  # each prompt once
        else:
            prompts = [args.prompt]
            repeat_per_prompt = args.n_runs  # for single prompt, n_runs keeps repeat semantics

        def run_one_prompt(prompt_text, seed_offset=0):
            best_result, best_score = None, float('-inf')
            repeats = MASTER_PARAMS['N'] if method == "rejection" else repeat_per_prompt
            for r in range(repeats):
                if args.seed is not None:
                    torch.manual_seed(args.seed + seed_offset + r)
                result, score = local_pipe(
                    prompt=prompt_text,
                    num_inference_steps=args.num_steps,
                    score_function=scorer,
                    method=method,
                    params=MASTER_PARAMS,
                )
                score_value = score.item() if torch.is_tensor(score) else float(score)
                if score_value > best_score:
                    best_result, best_score = result, score_value
            return best_result, best_score

        all_scores = []
        for idx, ptxt in enumerate(prompts):
            best_result, best_score = run_one_prompt(ptxt, seed_offset=idx)
            outname = args.output or (f"sd_{method}_{args.scorer}_{idx}.png" if len(prompts) > 1 else f"sd_{method}_{args.scorer}.png")
            best_result.images[0].save(outname)
            print(f"\n[SD][{idx}] Prompt: {ptxt}")
            print(f"[SD] Saved: {outname}\nBest score: {best_score}\n")
            all_scores.append(best_score)
        if len(all_scores) > 1:
            import numpy as np
            print(f"[SD] Overall {len(all_scores)} prompts mean score: {np.mean(all_scores):.4f} ± {np.std(all_scores):.4f}")

    # -----------
    # EDM Backend
    # -----------
    elif args.backend == 'edm':
        dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer = import_edm()
        scorer = get_scorer('edm', args.scorer, BrightnessScorer, CompressibilityScorer, ImageNetScorer=ImageNetScorer)

        # EDM defaults
        model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
        network_pkl = f'{model_root}/edm-imagenet-64x64-cond-adm.pkl'
        gridw = gridh = 6  # 生成36张图 (6x6 grid)
        num_images = gridw * gridh  # 36
        device = torch.device(args.device)
        num_steps = 18

        # EDM method mapping
        from edm.main import SamplingMethod, generate_image_grid
        method_map = {
            'naive': SamplingMethod.NAIVE,
            'rejection': SamplingMethod.REJECTION_SAMPLING,
            'beam': SamplingMethod.BEAM_SEARCH,
            'mcts': SamplingMethod.MCTS,
            'zero_order': SamplingMethod.ZERO_ORDER,
            'eps_greedy': SamplingMethod.EPS_GREEDY,
            'epsilon_1': SamplingMethod.EPS_GREEDY_1,
        }
        if args.method not in method_map:
            raise ValueError(f"Unknown method: {args.method}")
        sampling_method = method_map[args.method]
        sampling_params = {'scorer': scorer}

        # Add master params if relevant for method
        if args.method in ['rejection', 'zero_order', 'eps_greedy', 'epsilon_1', 'beam', 'mcts']:
            if args.N is not None:
                sampling_params['N'] = args.N
            if args.K is not None:
                sampling_params['K'] = args.K
            if args.lambda_ is not None:
                sampling_params['lambda_param'] = args.lambda_
            if args.eps is not None:
                sampling_params['eps'] = args.eps
            if args.B is not None:
                sampling_params['B'] = args.B
            if args.S is not None:
                sampling_params['S'] = args.S
            # For epsilon_1, add K1 and K2
            if args.method == 'epsilon_1':
                if args.K1 is not None:
                    sampling_params['K1'] = args.K1
                if args.K2 is not None:
                    sampling_params['K2'] = args.K2
                if args.revert_on_negative:
                    sampling_params['revert_on_negative'] = True

        outname = args.output or f"edm_{args.method}_{args.scorer}.png"
        
        # Run multiple times if n_runs > 1
        if args.n_runs == 1:
            # Single run
            torch.manual_seed(args.seed)
            latents = torch.randn([num_images, 3, 64, 64])
            class_labels = torch.eye(1000)[torch.randint(1000, size=[num_images])]
            
            avg_score = generate_image_grid(
                network_pkl,
                outname,
                latents,
                class_labels,
                seed=args.seed,
                gridw=gridw,
                gridh=gridh,
                device=device,
                num_steps=num_steps,
                S_churn=40,
                S_min=0.05,
                S_max=50,
                S_noise=1.003,
                sampling_method=sampling_method,
                sampling_params=sampling_params,
                log_gain=args.log_gain,
            )
            print(f"\n[EDM] Score: {avg_score:.4f}")
            print(f"[EDM] Saved: {outname}\n")
        else:
            # Multiple runs: collect scores and compute mean±std (as in paper)
            import numpy as np
            all_scores = []
            
            print(f"\nRunning {args.n_runs} times with different seeds...")
            for run_idx in range(args.n_runs):
                print(f"\n=== Run {run_idx + 1}/{args.n_runs} ===")
                run_seed = args.seed + run_idx
                
                # Generate new latents and class_labels for each run
                torch.manual_seed(run_seed)
                run_latents = torch.randn([num_images, 3, 64, 64])
                run_class_labels = torch.eye(1000)[torch.randint(1000, size=[num_images])]
                
                # Run with output file (only save last run's image)
                temp_outname = outname if run_idx == args.n_runs - 1 else None
                
                score = generate_image_grid(
                    network_pkl,
                    temp_outname,
                    run_latents,
                    run_class_labels,
                    seed=run_seed,
                    gridw=gridw,
                    gridh=gridh,
                    device=device,
                    num_steps=num_steps,
                    S_churn=40,
                    S_min=0.05,
                    S_max=50,
                    S_noise=1.003,
                    sampling_method=sampling_method,
                sampling_params=sampling_params,
                log_gain=args.log_gain,
                )
                all_scores.append(score)
            
            # Compute mean and std (as in paper Table 1)
            all_scores = np.array(all_scores)
            mean_score = np.mean(all_scores)
            std_score = np.std(all_scores)
            
            print(f"\n[EDM] Final results ({args.n_runs} runs):")
            print(f"[EDM] Score: {mean_score:.4f} ± {std_score:.4f}")
            if outname:
                print(f"[EDM] Saved last run image: {outname}\n")

if __name__ == '__main__':
    main()