import argparse
import csv
from copy import deepcopy
from pathlib import Path
from typing import List

import torch
import numpy as np

# Reuse helper import functions from the root main script
from main import import_sd, import_edm, get_scorer


def run_sd_probe(args, output_csv: Path):
    StableDiffusionPipeline, DDIMScheduler, BrightnessScorer, CompressibilityScorer, CLIPScorer = import_sd()
    scorer = get_scorer("sd", args.scorer, BrightnessScorer, CompressibilityScorer, CLIPScorer=CLIPScorer)

    model_id = "runwayml/stable-diffusion-v1-5"
    local_scheduler = DDIMScheduler.from_pretrained(model_id, subfolder="scheduler")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        scheduler=local_scheduler,
        torch_dtype=torch.float16,
    ).to(args.device)

    master_params = {
        "N": args.N,
        "lambda": args.lambda_,
        "eps": args.eps,
        "K": args.K,
        "K1": args.K1,
        "K2": args.K2,
        "revert_on_negative": args.revert_on_negative,
        "log_gain": False,
        "record_eps1_trace": True,
    }

    # Baseline run: record per-step/iter noises
    torch.manual_seed(args.seed)
    output, score = pipe(
        prompt=args.prompt,
        num_inference_steps=args.num_steps,
        score_function=safer_score(scorer),
        method="eps_greedy_1",
        params=master_params,
    )
    baseline_score = float(score)
    trace = getattr(output, "eps1_trace", None)
    if trace is None:
        raise RuntimeError("eps1_trace not returned; ensure pipeline supports record_eps1_trace=True")

    final_noise_per_step: List[torch.Tensor] = trace["final_noise_per_step"]
    best_noises_per_step: List[List[torch.Tensor]] = trace["best_noises_per_step"]
    total_replays = sum(len(x) for x in best_noises_per_step)
    print(
        f"[eps1_gain_probe][SD] baseline={baseline_score:.6f}, "
        f"steps={len(best_noises_per_step)}, total_replays={total_replays}"
    )

    rows = []
    rows.append(
        {
            "backend": "sd",
            "timestep": "baseline",
            "iter": "baseline",
            "score": baseline_score,
        }
    )

    # Replay each timestep/iter noise with others fixed to final noise
    for step_idx, iter_list in enumerate(best_noises_per_step):
        print(f"[eps1_gain_probe][SD] replay step {step_idx} iters={len(iter_list)}")
        for iter_idx, noise in enumerate(iter_list):
            plan = [n.clone() for n in final_noise_per_step]
            plan[step_idx] = noise
            replay_params = deepcopy(master_params)
            replay_params["record_eps1_trace"] = False
            replay_params["replay_noise_plan"] = plan

            torch.manual_seed(args.seed)  # keep deterministic
            _, replay_score = pipe(
                prompt=args.prompt,
                num_inference_steps=args.num_steps,
                score_function=safer_score(scorer),
                method="eps_greedy_1",
                params=replay_params,
            )
            rows.append(
                {
                    "backend": "sd",
                    "timestep": step_idx,
                    "iter": iter_idx,
                    "score": float(replay_score),
                }
            )

    write_rows(output_csv, rows)


def run_edm_probe(args, output_csv: Path):
    dnnlib, dnnlib_util, BrightnessScorer, CompressibilityScorer, ImageNetScorer = import_edm()
    scorer = get_scorer("edm", args.scorer, BrightnessScorer, CompressibilityScorer, ImageNetScorer=ImageNetScorer)

    from edm.main import SamplingMethod, generate_image_grid

    model_root = "https://nvlabs-fi-cdn.nvidia.com/edm/pretrained"
    network_pkl = f"{model_root}/edm-imagenet-64x64-cond-adm.pkl"
    gridw = gridh = 6
    num_images = gridw * gridh
    device = torch.device(args.device)
    num_steps = args.num_steps

    sampling_params = {
        "scorer": scorer,
        "N": args.N,
        "K1": args.K1,
        "K2": args.K2,
        "lambda_param": args.lambda_,
        "eps": args.eps,
        "revert_on_negative": args.revert_on_negative,
        "record_eps1_trace": True,
    }

    sampling_method = SamplingMethod.EPS_GREEDY_1

    torch.manual_seed(args.seed)
    latents = torch.randn([num_images, 3, 64, 64])
    class_labels = torch.eye(1000)[torch.randint(1000, size=[num_images])]

    baseline_result = generate_image_grid(
        network_pkl,
        dest_path=None,
        latents=latents,
        class_labels=class_labels,
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
        log_gain=False,
    )
    if not isinstance(baseline_result, tuple) or len(baseline_result) != 2:
        raise RuntimeError("EDM baseline result did not return (score, trace)")
    baseline_score, trace = baseline_result
    final_noise_per_step = trace["final_noise_per_step"]
    best_noises_per_step = trace["best_noises_per_step"]
    total_replays = sum(len(x) for x in best_noises_per_step)
    print(
        f"[eps1_gain_probe][EDM] baseline={float(baseline_score):.6f}, "
        f"steps={len(best_noises_per_step)}, total_replays={total_replays}"
    )

    rows = [
        {"backend": "edm", "timestep": "baseline", "iter": "baseline", "score": float(baseline_score)}
    ]

    for step_idx, iter_list in enumerate(best_noises_per_step):
        print(f"[eps1_gain_probe][EDM] replay step {step_idx} iters={len(iter_list)}")
        for iter_idx, noise in enumerate(iter_list):
            plan = [n.clone() for n in final_noise_per_step]
            plan[step_idx] = noise
            replay_params = deepcopy(sampling_params)
            replay_params["record_eps1_trace"] = False
            replay_params["replay_noise_plan"] = plan

            torch.manual_seed(args.seed)
            latents_re = torch.randn([num_images, 3, 64, 64])
            class_labels_re = torch.eye(1000)[torch.randint(1000, size=[num_images])]
            replay_result = generate_image_grid(
                network_pkl,
                dest_path=None,
                latents=latents_re,
                class_labels=class_labels_re,
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
                sampling_params=replay_params,
                log_gain=False,
            )
            if isinstance(replay_result, tuple):
                replay_score = replay_result[0]
            else:
                replay_score = replay_result
            rows.append(
                {
                    "backend": "edm",
                    "timestep": step_idx,
                    "iter": iter_idx,
                    "score": float(replay_score),
                }
            )

    write_rows(output_csv, rows)


def safer_score(score_fn):
    # Wrap scorer to keep output on CPU float for logging
    def _wrapped(**kwargs):
        out = score_fn(**kwargs)
        if torch.is_tensor(out):
            return out.detach().cpu().float()
        return out

    return _wrapped


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["backend", "timestep", "iter", "score"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"[eps1_gain_probe] wrote {len(rows)} rows to {path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Probe epsilon_1 per-step/iter gains via replay.")
    parser.add_argument("--backend", choices=["sd", "edm"], required=True)
    parser.add_argument("--scorer", choices=["brightness", "compressibility", "clip", "imagenet"], required=True)
    parser.add_argument("--prompt", type=str, default="A beautiful landscape", help="SD prompt (ignored for EDM)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_steps", type=int, default=50, help="Denoising steps")
    parser.add_argument("--N", type=int, default=4)
    parser.add_argument("--lambda_", type=float, default=0.15)
    parser.add_argument("--eps", type=float, default=0.4)
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--K1", type=int, default=25)
    parser.add_argument("--K2", type=int, default=15)
    parser.add_argument("--revert_on_negative", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_csv", type=str, default="eps1_probe.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    out_csv = Path(args.output_csv)
    if args.backend == "sd":
        run_sd_probe(args, out_csv)
    else:
        run_edm_probe(args, out_csv)


if __name__ == "__main__":
    main()


