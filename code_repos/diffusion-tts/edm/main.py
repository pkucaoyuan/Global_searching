# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Minimal standalone example to reproduce the main results from the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os

import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
import matplotlib.pyplot as plt
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from scorers import Scorer, CompressibilityScorer, BrightnessScorer, ImageNetScorer
from torchvision import transforms

class SamplingMethod(Enum):
    MCTS = auto()
    BEAM_SEARCH = auto()
    ZERO_ORDER = auto()
    NAIVE = auto()
    REJECTION_SAMPLING = auto()
    EPS_GREEDY = auto()
    EPS_GREEDY_1 = auto()
    EPS_GREEDY_ONLINE = auto()

@dataclass
class SamplingParams:
    B: int = 2
    N: int = 4
    K: int = 20
    K1: int = 25  # For EPS_GREEDY_1: K for头两步+最后4步
    K2: int = 15  # For EPS_GREEDY_1: K for其余中间步
    revert_on_negative: bool = False  # EPS_GREEDY_1: 负增益时是否回退到上一轮 pivot（首迭代不回退）
    lambda_param: float = 0.15
    eps: float = 0.4
    # Online/early-stop related (ignored when不适用)
    total_budget: Optional[float] = None
    high_slack: int = 2
    thresh_gain_coef: float = 1.0
    thresh_var_coef: float = 1.0
    # 动态调度相关参数（默认关闭，extra_budget<=0 或 tau0<=0 时不启用）
    extra_budget: float = 0.0  # 允许的额外 NFE 预算（相对 baseline）
    gamma_reserve: float = 0.5  # 预算预留系数 γ
    tau0: float = 0.0  # 动态阈值起点 τ0
    alpha: float = 1.0  # 阈值随预算收紧的斜率 α
    probe_M: int = 0  # 每步用于决策的 probe 噪声数 M（0 表示不做 probe）
    S: int = 8
    scorer: Scorer = field(default_factory=lambda: CompressibilityScorer(dtype=torch.float32))
    # eps1_trace & replay support (ignored unless SamplingMethod.EPS_GREEDY_1)
    record_eps1_trace: bool = False
    replay_noise_plan: Optional[Any] = None

#----------------------------------------------------------------------------

def generate_image_grid(
    network_pkl, dest_path, latents, class_labels,
    seed=0, gridw=8, gridh=8, device=torch.device('cuda'),
    num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
    S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
    sampling_method: SamplingMethod = SamplingMethod.NAIVE,
    sampling_params: Optional[Dict[str, Any]] = None,
    precomputed_noise: Optional[Dict[int, torch.Tensor]] = None,
    log_gain: bool = False,
):
    # Set up environment and seed
    batch_size = gridw * gridh
    torch.manual_seed(seed)
    
    # Initialize sampling parameters
    if sampling_params is None:
        sampling_params = {}
    params_class = SamplingParams
    method_params = params_class(**sampling_params)
    print(f'Using sampling method: {sampling_method.name}')

    # Load network
    print(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl) as f:
        net = pickle.load(f)['ema'].to(device)
        
    # Move data to device
    latents = latents.to(device)
    if class_labels is not None:
        class_labels = class_labels.to(device)
        
    # Time step discretization
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])
    
    def step(x_cur, t_cur, t_next, i, eps_i, class_labels_for_step=None):
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
        t_hat = net.round_sigma(t_cur + gamma * t_cur)
        x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * eps_i

        denoised = net(x_hat, t_hat, class_labels_for_step).to(torch.float64)
        d_cur = (x_hat - denoised) / t_hat
        x_next = x_hat + (t_next - t_hat) * d_cur

        if i < num_steps - 1:
            denoised = net(x_next, t_next, class_labels_for_step).to(torch.float64)
            d_prime = (x_next - denoised) / t_next
            x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next, denoised

    # Main sampling loop
    x_next = latents.to(torch.float64) * t_steps[0]
    
    if sampling_method == SamplingMethod.REJECTION_SAMPLING:
        # Get rejection sampling parameters
        N = method_params.N
        batch_size = len(latents)
        
        # Do the whole denoising loop for N initial candidate noise vectors, then choose the best
        x_next_expanded = x_next.repeat_interleave(N, dim=0)  # [batch_size * N, C, H, W]
        class_labels_expanded = class_labels.repeat_interleave(N, dim=0)

        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'):
            x_cur = x_next_expanded  # Use expanded tensor
            
            # Use precomputed noise if provided, otherwise generate new noise
            if precomputed_noise is not None and i in precomputed_noise:
                # The precomputed noise has shape [batch_size, max_N, C, H, W]
                # Take the first N vectors for each batch element
                eps_i = precomputed_noise[i][:, :N].reshape(batch_size * N, *x_cur.shape[1:])
            else:
                # Generate random noise vectors
                eps_i = torch.randn_like(x_cur)
                
            x_next_expanded, _ = step(x_cur, t_cur, t_next, i, eps_i, class_labels_expanded)
        
        # Convert to images for scoring but don't normalize to [0,1]
        # Just convert to uint8 in [0,255] range which is what the model expects
        image_for_scoring = (x_next_expanded * 127.5 + 128).clip(0, 255).to(torch.uint8)
        # Use timesteps=0 for fully denoised images
        timesteps = torch.zeros(image_for_scoring.shape[0], device=device)
        scores = method_params.scorer(image_for_scoring, class_labels_expanded, timesteps)
        
        # Reshape scores and original tensors to select best candidates
        scores = scores.view(batch_size, N)  # [batch_size, N]
        x_next_reshaped = x_next_expanded.view(batch_size, N, *x_next_expanded.shape[1:])  # [batch_size, N, C, H, W]
        
        # Select best candidates from the original floating-point values
        best_indices = scores.argmax(dim=1)  # [batch_size]
        x_next = torch.stack([x_next_reshaped[i, idx] for i, idx in enumerate(best_indices)])  # [batch_size, C, H, W]
    elif sampling_method == SamplingMethod.BEAM_SEARCH:
        # Get beam search parameters
        b, k = method_params.b, method_params.k
        batch_size = len(latents)
        print(f"Beam Search Parameters - b: {b}, k: {k}, batch_size: {batch_size}")
        
        # Dynamically adjust batch size based on beam width
        # For larger beam widths, use smaller batches
        max_sub_batch = max(1, min(8, 32 // b))  # Dynamically scale batch size inversely with beam width
        print(f"Using sub-batch size of {max_sub_batch} for processing")
        
        # Micro-batch size for model inference
        micro_batch = max(1, min(4, 16 // b))  # Even smaller batches for inference
        print(f"Using micro-batch size of {micro_batch} for model inference")
        
        # Initialize with batch_size samples, each with k candidates
        # Repeat each sample k times to start
        x_next_expanded = x_next.repeat_interleave(k, dim=0)  # [batch_size * k, C, H, W]
        
        x_curs = [(x_next_expanded, None)]  # [(x, x0)] format
        sample_indices = torch.arange(batch_size, device=device).repeat_interleave(k)  # Track which sample each beam belongs to
        
        # Create a step function with micro-batching for large beam widths
        def step_with_microbatch(x_cur, t_cur, t_next, i, eps_i, class_labels_for_step=None, micro_batch_size=micro_batch):
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * eps_i
            
            # Process in micro-batches to avoid OOM
            full_batch_size = x_hat.shape[0]
            all_denoised = []
            
            for mb_start in range(0, full_batch_size, micro_batch_size):
                mb_end = min(mb_start + micro_batch_size, full_batch_size)
                x_hat_micro = x_hat[mb_start:mb_end]
                class_labels_micro = None if class_labels_for_step is None else class_labels_for_step[mb_start:mb_end]
                
                # Directly call the model without autocast
                denoised_micro = net(x_hat_micro, t_hat, class_labels_micro).to(torch.float64)
                all_denoised.append(denoised_micro)
                
                # Clean up to avoid memory buildup
                del x_hat_micro, denoised_micro
                torch.cuda.empty_cache()
                
            denoised = torch.cat(all_denoised, dim=0)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            
            if i < num_steps - 1:
                # Also process second denoising step in micro-batches
                all_denoised_2 = []
                
                for mb_start in range(0, full_batch_size, micro_batch_size):
                    mb_end = min(mb_start + micro_batch_size, full_batch_size)
                    x_next_micro = x_next[mb_start:mb_end]
                    class_labels_micro = None if class_labels_for_step is None else class_labels_for_step[mb_start:mb_end]
                    
                    # Directly call the model without autocast
                    denoised_micro = net(x_next_micro, t_next, class_labels_micro).to(torch.float64)
                    all_denoised_2.append(denoised_micro)
                    
                    # Clean up to avoid memory buildup
                    del x_next_micro, denoised_micro
                    torch.cuda.empty_cache()
                    
                denoised_2 = torch.cat(all_denoised_2, dim=0)
                d_prime = (x_next - denoised_2) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                
                del denoised_2, all_denoised_2
            
            del x_hat, all_denoised, d_cur
            torch.cuda.empty_cache()
            return x_next, denoised
        
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit="step"):
            # For each existing trajectory (should be only one after the first iteration)
            next_candidates = []
            
            # Calculate total batch size for current beam
            x_cur = x_curs[0]  # Just use the single element in x_curs
            cur_batch_size = x_cur[0].shape[0]  # This is batch_size * k
            
            # Process in smaller batches to avoid OOM
            all_x_candidates = []
            all_x0_candidates = []
            
            for sub_batch_start in range(0, cur_batch_size, max_sub_batch):
                sub_batch_end = min(sub_batch_start + max_sub_batch, cur_batch_size)
                sub_batch_size = sub_batch_end - sub_batch_start
                
                # Get the current subset of beams
                x_sub_batch = x_cur[0][sub_batch_start:sub_batch_end]
                
                # For each beam, process candidates in smaller sub-batches
                x_candidates_sub = []
                x0_candidates_sub = []
                
                # Process candidate generation in sub-batches
                for beam_batch_start in range(0, sub_batch_size, 1):  # Process one beam at a time
                    beam_batch_end = min(beam_batch_start + 1, sub_batch_size)
                    beam_batch_size = beam_batch_end - beam_batch_start
                    
                    # Get current beam
                    x_beam = x_sub_batch[beam_batch_start:beam_batch_end]
                    
                    # Expand current beam to b candidates
                    x_expanded = x_beam.repeat(b, 1, 1, 1)  # Shape: [b, C, H, W]
                    
                    if class_labels is not None:
                        # Get the class label for this beam
                        sub_idx = sample_indices[sub_batch_start + beam_batch_start]
                        beam_label = class_labels[sub_idx:sub_idx+1]
                        class_labels_expanded = beam_label.repeat(b, 1)
                    else:
                        class_labels_expanded = None
                    
                    # Generate b noise vectors for this beam
                    eps_i = torch.randn_like(x_expanded)
                    
                    # Process all candidates using micro-batched version
                    x_candidate_beam, x0_candidate_beam = step_with_microbatch(
                        x_expanded, t_cur, t_next, i, eps_i, class_labels_expanded
                    )
                    
                    # Reshape to [1, b, C, H, W] for consistent dimensions
                    x_candidate_beam = x_candidate_beam.unsqueeze(0)  # Shape: [1, b, C, H, W]
                    x0_candidate_beam = x0_candidate_beam.unsqueeze(0)  # Shape: [1, b, C, H, W]
                    
                    x_candidates_sub.append(x_candidate_beam)
                    x0_candidates_sub.append(x0_candidate_beam)
                    
                    # Free memory
                    del x_expanded, eps_i, class_labels_expanded, x_candidate_beam, x0_candidate_beam
                    torch.cuda.empty_cache()
                
                # Combine all beams in this sub-batch
                x_candidates_cat = torch.cat(x_candidates_sub, dim=0)  # [sub_batch_size, b, C, H, W]
                x0_candidates_cat = torch.cat(x0_candidates_sub, dim=0)  # [sub_batch_size, b, C, H, W]
                
                all_x_candidates.append(x_candidates_cat)
                all_x0_candidates.append(x0_candidates_cat)
                
                # Free memory
                del x_sub_batch, x_candidates_sub, x0_candidates_sub, x_candidates_cat, x0_candidates_cat
                torch.cuda.empty_cache()
            
            # Combine all candidates
            x_candidate = torch.cat(all_x_candidates, dim=0)  # [cur_batch_size, b, C, H, W]
            x0_candidate = torch.cat(all_x0_candidates, dim=0)  # [cur_batch_size, b, C, H, W]
            
            # Free memory for intermediate results
            del all_x_candidates, all_x0_candidates
            torch.cuda.empty_cache()
            
            # Score all candidates in small batches
            all_scores = []
            
            # Very small scoring batch size
            max_score_batch = max(1, 4 // b)  # Adjust scoring batch size based on beam width
            
            for score_batch_start in range(0, cur_batch_size, max_score_batch):
                score_batch_end = min(score_batch_start + max_score_batch, cur_batch_size)
                score_batch_size = score_batch_end - score_batch_start
                
                x_candidate_batch = x_candidate[score_batch_start:score_batch_end]
                x0_candidate_batch = x0_candidate[score_batch_start:score_batch_end]
                
                # Score one beam at a time to minimize memory usage
                beam_scores = []
                
                for beam_idx in range(score_batch_size):
                    # Get candidates for this beam
                    x_beam_candidates = x_candidate_batch[beam_idx]  # [b, C, H, W]
                    x0_beam_candidates = x0_candidate_batch[beam_idx]  # [b, C, H, W]
                    
                    # Variables for deletion
                    x_for_scoring = None
                    x_flat_var = None
                    
                    # Score based on denoised predictions (x0)
                    x_flat_var = x0_beam_candidates  # Already [b, C, H, W]
                    
                    # Normalize for scoring
                    x_for_scoring = (x_flat_var * 127.5 + 128).clip(0, 255).to(torch.uint8)
                    
                    # Use timesteps=0 for denoised images
                    timesteps = torch.zeros(x_for_scoring.shape[0], device=device)
                    
                    if class_labels is not None:
                        # Get class label for this beam
                        beam_orig_idx = sample_indices[score_batch_start + beam_idx]
                        class_labels_flat = class_labels[beam_orig_idx:beam_orig_idx+1].repeat(b, 1)
                    else:
                        class_labels_flat = None
                    
                    # Score candidates for this beam
                    beam_score = method_params.scorer(x_for_scoring, class_labels_flat, timesteps)
                    beam_scores.append(beam_score)
                    
                    # Free memory
                    del x_for_scoring, x_flat_var, timesteps
                    if class_labels_flat is not None:
                        del class_labels_flat
                    torch.cuda.empty_cache()
                
                # Combine scores for all beams in this batch
                batch_scores = torch.cat(beam_scores, dim=0).view(score_batch_size, b)
                all_scores.append(batch_scores)
                
                del beam_scores, batch_scores, x_candidate_batch, x0_candidate_batch
                torch.cuda.empty_cache()
            
            # Combine all scores
            scores = torch.cat(all_scores, dim=0)  # [cur_batch_size, b]
            
            # Free memory
            del all_scores
            torch.cuda.empty_cache()
            
            # Reshape scores for sample-level selection
            scores = scores.view(batch_size, k * b)  # [batch_size, k*b]
            
            # Select top k for each sample
            topk_scores, topk_indices = torch.topk(scores, k=k, dim=1)  # [batch_size, k]
            
            # Gather top k candidates for each sample - process in small batches
            new_x = []
            new_x0 = []
            
            for sample_idx in range(batch_size):
                sample_beam_indices = torch.arange(k, device=device) + sample_idx * k
                sample_indices_k = topk_indices[sample_idx]  # [k]
                
                for beam_idx in range(k):
                    # Calculate proper indices in the candidate tensors
                    beam_base_idx = sample_beam_indices[beam_idx] // k  # Current beam index
                    candidate_idx = sample_indices_k[beam_idx] % b  # Which of the b candidates was selected
                    beam_batch_idx = beam_base_idx % cur_batch_size  # Simpler calculation for batch index
                    
                    x_next_sample = x_candidate[beam_batch_idx, candidate_idx]  # [C, H, W]
                    x0_next_sample = x0_candidate[beam_batch_idx, candidate_idx]  # [C, H, W]
                    
                    new_x.append(x_next_sample)
                    new_x0.append(x0_next_sample)
                
                # Free intermediate memory every few samples
                if sample_idx % 8 == 7:
                    torch.cuda.empty_cache()
            
            # Stack all selected candidates
            x_next_all = torch.stack(new_x, dim=0)  # [batch_size * k, C, H, W]
            x0_next_all = torch.stack(new_x0, dim=0)  # [batch_size * k, C, H, W]
            
            # Free major tensors
            del x_candidate, x0_candidate, scores, topk_scores, topk_indices, new_x, new_x0
            torch.cuda.empty_cache()
            
            # Reset x_curs to a single element with the updated beams
            x_curs = [(x_next_all, x0_next_all)]
            
            # Update sample indices for the next iteration
            sample_indices = torch.arange(batch_size, device=device).repeat_interleave(k)
        
        # After all steps, take the best candidate for each sample
        x_next = x_next_all.view(batch_size, k, *x_next_all.shape[1:])[:, 0]  # [batch_size, C, H, W]
    elif sampling_method == SamplingMethod.MCTS:
        # Get MCTS parameters
        b = method_params.N
        N = method_params.S
        
        # Run MCTS separately for each batch element
        results = []
        
        # Process in mini-batches for better GPU utilization
        mini_batch_size = min(2, batch_size)  # Process up to 36 samples in parallel
        for mb_start in tqdm.tqdm(range(0, batch_size, mini_batch_size), desc="MCTS sampling", unit="batch"):
            mb_end = min(mb_start + mini_batch_size, batch_size)
            mb_size = mb_end - mb_start
            # print(f"\n==== Processing mini-batch {mb_start//mini_batch_size + 1}/{(batch_size + mini_batch_size - 1)//mini_batch_size} (samples {mb_start+1}-{mb_end}) ====")
            
            # Extract mini-batch elements
            x_batch = x_next[mb_start:mb_end]
            class_labels_batch = None if class_labels is None else class_labels[mb_start:mb_end]
            
            # Initialize per-sample MCTS data structures
            all_children = [{} for _ in range(mb_size)]
            all_parent = [{} for _ in range(mb_size)]
            all_reward = [{} for _ in range(mb_size)]
            all_visit = [{} for _ in range(mb_size)]
            all_roots = []
            all_root_keys = []
            
            # Helper function to get a unique key for tensors
            def get_tensor_key(tensor, idx):
                # Use tensor's data pointer as a unique identifier plus sample index for uniqueness
                return f"{idx}_{str(tensor.data_ptr())}"
            
            # Precompute noise candidates for this mini-batch
            precomputed_noise_mcts = {}
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                # If precomputed noise is provided, use it (broadcasting to all samples in mini-batch)
                if precomputed_noise is not None and i in precomputed_noise:
                    # The provided noise is shape [1, b, C, H, W] - repeat for each sample in mini-batch
                    noise_candidates = precomputed_noise[i].repeat(mb_size, 1, 1, 1, 1)
                else:
                    # Shape: [mb_size, b, channels, height, width]
                    noise_candidates = torch.randn(mb_size, b, *x_batch.shape[1:], device=device)
                precomputed_noise_mcts[i] = noise_candidates
            
            # Initialize root nodes for each sample in mini-batch
            for idx in range(mb_size):
                x_single = x_batch[idx:idx+1]
                root = x_single.clone()
                root_key = get_tensor_key(root, idx)
                all_children[idx][root_key] = []
                all_reward[idx][root_key] = 0
                all_visit[idx][root_key] = 1
                all_roots.append(root)
                all_root_keys.append(root_key)
            
            # Main MCTS loop for this mini-batch
            for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
                # print(f"  Timestep {i+1}/{len(t_steps)-1} (t={t_cur.item():.6f} → {t_next.item():.6f})")
                
                # First expand the root nodes for all samples in the mini-batch at once
                expansion_tensors = []
                expansion_indices = []
                expansion_sample_indices = []
                
                for sample_idx in range(mb_size):
                    x_cur_key = all_root_keys[sample_idx]
                    children = all_children[sample_idx]
                    
                    # Expand root with b noise candidates if not already expanded
                    if not x_cur_key in children or len(children[x_cur_key]) == 0:
                        root = all_roots[sample_idx]
                        for noise_idx in range(b):
                            expansion_tensors.append(root)
                            expansion_indices.append(noise_idx)
                            expansion_sample_indices.append(sample_idx)
                
                # Batch process all expansions at once
                if expansion_tensors:
                    expansion_batch = torch.cat(expansion_tensors, dim=0)
                    noise_batch = torch.cat([
                        precomputed_noise_mcts[i][sample_idx:sample_idx+1, noise_idx] 
                        for sample_idx, noise_idx in zip(expansion_sample_indices, expansion_indices)
                    ], dim=0)
                    
                    # Get class labels for this expansion batch
                    expansion_class_labels = None
                    if class_labels_batch is not None:
                        expansion_class_labels = torch.cat([
                            class_labels_batch[sample_idx:sample_idx+1] 
                            for sample_idx in expansion_sample_indices
                        ], dim=0)
                    
                    # Process expansion in a single batch
                    x_next_candidates, _ = step(expansion_batch, t_cur, t_next, i, noise_batch, expansion_class_labels)
                    
                    # Distribute results back to each sample's tree
                    start_idx = 0
                    for sample_idx, noise_idx in zip(expansion_sample_indices, expansion_indices):
                        root = all_roots[sample_idx]
                        root_key = all_root_keys[sample_idx]
                        x_next_candidate = x_next_candidates[start_idx:start_idx+1]
                        x_next_key = get_tensor_key(x_next_candidate, sample_idx)
                        
                        all_children[sample_idx][root_key].append((x_next_candidate, x_next_key))
                        all_parent[sample_idx][x_next_key] = root_key
                        all_reward[sample_idx][x_next_key] = 0
                        all_visit[sample_idx][x_next_key] = 0
                        all_children[sample_idx][x_next_key] = []
                        
                        start_idx += 1
                
                # Batch simulations for better GPU utilization
                # We'll run multiple simulations simultaneously
                simulation_batch_size = min(16, N * mb_size)  # Process up to 16 simulations at once
                
                for sim_start in range(0, N * mb_size, simulation_batch_size):
                    sim_end = min(sim_start + simulation_batch_size, N * mb_size)
                    
                    # Track simulation data for each sample and rollout
                    all_paths = []
                    all_traversal_data = []
                    
                    # Prepare simulation batch
                    for sim_idx in range(sim_start, sim_end):
                        sample_idx = sim_idx % mb_size  # Which sample this sim belongs to
                        rollout_idx = sim_idx // mb_size  # Which rollout number within the sample
                        
                        # Get sample's MCTS data structures
                        children = all_children[sample_idx]
                        parent = all_parent[sample_idx]
                        reward = all_reward[sample_idx]
                        visit = all_visit[sample_idx]
                        root = all_roots[sample_idx]
                        x_cur_key = all_root_keys[sample_idx]
                        
                        # Selection phase - traverse tree until leaf
                        x_traverse = root.clone()
                        x_traverse_key = x_cur_key
                        t_traverse_cur = t_cur
                        t_traverse_next = t_next
                        i_traverse = i
                        
                        # Selection - traverse tree according to UCB
                        path = [(x_traverse, x_traverse_key, t_traverse_cur, i_traverse)]
                        
                        # Continue selection until node with unexplored children or max depth
                        while children[x_traverse_key]:
                            # Compute UCB for all children
                            ucb_values = []
                            for child, child_key in children[x_traverse_key]:
                                if visit[child_key] == 0:
                                    ucb_values.append(float('inf'))
                                else:
                                    exploitation = reward[child_key] / visit[child_key]
                                    exploration = np.sqrt(2 * np.log(visit[x_traverse_key]) / visit[child_key])
                                    ucb_values.append(exploitation + exploration)
                            
                            # Select child with highest UCB
                            best_child_idx = np.argmax(ucb_values)
                            x_traverse, x_traverse_key = children[x_traverse_key][best_child_idx]
                            
                            # Update time indices for next step
                            i_traverse += 1
                            if i_traverse < len(t_steps) - 1:
                                t_traverse_cur = t_steps[i_traverse]
                                t_traverse_next = t_steps[i_traverse + 1]
                            
                            path.append((x_traverse, x_traverse_key, t_traverse_cur, i_traverse))
                        
                        # Expansion (if not at terminal node)
                        if i_traverse < len(t_steps) - 2:
                            # Generate noise candidates for expansion
                            for noise_idx in range(b):
                                eps_i = precomputed_noise_mcts.get(i_traverse, 
                                        torch.randn(1, *x_batch[sample_idx:sample_idx+1].shape[1:], device=device))[sample_idx, noise_idx:noise_idx+1]
                                
                                x_next_candidate, _ = step(x_traverse, t_traverse_cur, t_traverse_next, 
                                                          i_traverse, eps_i, 
                                                          None if class_labels_batch is None else class_labels_batch[sample_idx:sample_idx+1])
                                x_next_key = get_tensor_key(x_next_candidate, sample_idx)
                                
                                children[x_traverse_key].append((x_next_candidate, x_next_key))
                                parent[x_next_key] = x_traverse_key
                                reward[x_next_key] = 0
                                visit[x_next_key] = 0
                                children[x_next_key] = []
                            
                            # Choose random child for simulation
                            child_idx = np.random.randint(0, len(children[x_traverse_key]))
                            x_traverse, x_traverse_key = children[x_traverse_key][child_idx]
                            
                            # Update time indices
                            i_traverse += 1
                            if i_traverse < len(t_steps) - 1:
                                t_traverse_cur = t_steps[i_traverse]
                                t_traverse_next = t_steps[i_traverse + 1]
                            
                            path.append((x_traverse, x_traverse_key, t_traverse_cur, i_traverse))
                        
                        # Store path for backpropagation
                        all_paths.append(path)
                        # Store current state for simulation
                        all_traversal_data.append((x_traverse.clone(), i_traverse, sample_idx))
                    
                    # Perform simulations in parallel for all samples in this batch
                    if all_traversal_data:
                        # Extract simulation starting points
                        x_sim_batch = torch.cat([data[0] for data in all_traversal_data], dim=0)
                        i_sim_batch = [data[1] for data in all_traversal_data]
                        sample_indices_batch = [data[2] for data in all_traversal_data]
                        
                        # Run deterministic sampling from current timestep to the end
                        temp_x = x_sim_batch.clone()
                        
                        # Process each simulation from its current timestep to the end
                        for sim_idx, i_start in enumerate(i_sim_batch):
                            temp_x_single = temp_x[sim_idx:sim_idx+1]
                            
                            # Get class labels for this simulation
                            sim_class_label = None
                            if class_labels_batch is not None:
                                sample_idx_for_labels = sample_indices_batch[sim_idx]
                                sim_class_label = class_labels_batch[sample_idx_for_labels:sample_idx_for_labels+1]
                            
                            # Sample from i_start to the end
                            for j in range(i_start, len(t_steps) - 1):
                                t_j_cur = t_steps[j]
                                t_j_next = t_steps[j + 1]
                                
                                # Use step function with zero noise for deterministic sampling
                                temp_x_single, _ = step(
                                    temp_x_single, 
                                    t_j_cur, 
                                    t_j_next, 
                                    j, 
                                    torch.zeros_like(temp_x_single),  # Zero noise for deterministic
                                    sim_class_label
                                )
                            
                            # Update the batch with the fully sampled result
                            temp_x[sim_idx:sim_idx+1] = temp_x_single
                        
                        # The final temp_x is our fully denoised result
                        denoised = temp_x
                        
                        # Get current timestep info for each simulation
                        current_t = torch.stack([t_steps[i_sim] for i_sim in i_sim_batch]).to(device)
                        
                        # Get class labels for scoring
                        sim_class_labels = None
                        if class_labels_batch is not None:
                            sim_class_labels = torch.cat([
                                class_labels_batch[idx:idx+1] for idx in sample_indices_batch
                            ], dim=0)
                        
                        # Score the predicted denoised images
                        image_for_scoring = (denoised * 127.5 + 128).clip(0, 255).to(torch.uint8)
                        # Use timesteps=0 for denoised predictions
                        timesteps = torch.zeros(image_for_scoring.shape[0], device=device)
                        sim_rewards = method_params.scorer(image_for_scoring, sim_class_labels, timesteps)
                        
                        # Backpropagation for each simulation
                        for sim_idx, (path, reward_value) in enumerate(zip(all_paths, sim_rewards)):
                            sample_idx = sample_indices_batch[sim_idx]
                            reward = all_reward[sample_idx]
                            visit = all_visit[sample_idx]
                            
                            # Update reward and visit counts for all nodes in path
                            for _, node_key, _, _ in path:
                                # Ensure node exists in dictionaries
                                if node_key not in reward:
                                    reward[node_key] = 0
                                if node_key not in visit:
                                    visit[node_key] = 0
                                
                                reward[node_key] += reward_value.item()
                                visit[node_key] += 1
                
                # Select best child for each sample based on average reward
                for sample_idx in range(mb_size):
                    children = all_children[sample_idx]
                    reward = all_reward[sample_idx]
                    visit = all_visit[sample_idx]
                    x_cur_key = all_root_keys[sample_idx]
                    
                    best_reward = -float('inf')
                    best_child = None
                    
                    for child, child_key in children[x_cur_key]:
                        if visit[child_key] > 0:
                            avg_reward = reward[child_key] / visit[child_key]
                            if avg_reward > best_reward:
                                best_reward = avg_reward
                                best_child = child
                    
                    # Update root for next timestep
                    assert best_child is not None
                    all_roots[sample_idx] = best_child.clone()
                    all_root_keys[sample_idx] = get_tensor_key(best_child, sample_idx)
            
            # Add final result for each sample in mini-batch
            for root in all_roots:
                results.append(root)
            
            # print(f"  Completed mini-batch processing for samples {mb_start+1}-{mb_end}")
        
        # Combine results back into a batch
        x_next = torch.cat(results, dim=0)
        # print(f"MCTS sampling completed for all {batch_size} elements")
    elif sampling_method == SamplingMethod.ZERO_ORDER or sampling_method == SamplingMethod.EPS_GREEDY:
        # Get Zero-Order parameters
        lambda_param = method_params.lambda_param * np.sqrt(3 * 64 * 64)
        N = method_params.N
        K = method_params.K
        eps = method_params.eps
        # 预算/阈值调度参数（extra_budget<=0 或 tau0<=0 时不启用）
        use_budget_scheduler = method_params.extra_budget > 0 and method_params.tau0 > 0
        log_gain = log_gain or (sampling_method == SamplingMethod.EPS_GREEDY)
        B_extra = method_params.extra_budget
        gamma_reserve = method_params.gamma_reserve
        tau0 = method_params.tau0
        alpha = method_params.alpha
        probe_M = method_params.probe_M
        B_used = 0.0
        heavy_cost = float(K * N * 2 + 2)  # 近似每步 heavy search 的上限 NFE（两次 UNet 视为 2）
        log_gain = sampling_method == SamplingMethod.EPS_GREEDY
        gains_per_step = []  # 记录每个 timestep 内各迭代的 gain（对 EPS_GREEDY）
        
        print(f"Zero-Order parameters: lambda={lambda_param / np.sqrt(3 * 64 * 64)}, N={N}, K={K}, eps={eps}")
        
        # Use precomputed pivot noise if provided, otherwise generate a fresh one
        if precomputed_noise is not None and 'pivot' in precomputed_noise:
            pivot_noise = precomputed_noise['pivot']
        else:
            pivot_noise = torch.randn_like(x_next)
        
        # Main denoising loop over timesteps
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'):
            x_cur = x_next
            
            # Initialize pivot noise with a fresh Gaussian sample
            if precomputed_noise is not None and f'pivot_{i}' in precomputed_noise:
                pivot_noise = precomputed_noise[f'pivot_{i}']
            else:
                pivot_noise = torch.randn_like(x_cur)
            
            # baseline：用当前 pivot 做一次 step，仅取 x0 评分（用于阈值/日志）
            need_baseline = use_budget_scheduler or log_gain
            base_scores = None
            if need_baseline:
                with torch.no_grad():
                    _, x0_baseline = step(x_cur, t_cur, t_next, i, pivot_noise, class_labels)
                x_base_for_scoring = (x0_baseline * 127.5 + 128).clip(0, 255).to(torch.uint8)
                base_timesteps = torch.zeros(x_base_for_scoring.shape[0], device=device)
                base_scores = method_params.scorer(x_base_for_scoring, class_labels, base_timesteps)
                base_scores = base_scores.reshape(-1)  # [batch_size]

            iterations_run = 0  # 实际运行的迭代次数
            # -------- 预算 & 难度调度（可选）--------
            if use_budget_scheduler:

                # probe：M 个候选噪声，仅做“一步 x0 + score”
                D_t = 0.0
                if probe_M > 0:
                    candidate_noises = [torch.randn_like(x_cur) for _ in range(probe_M)]
                    all_noises = torch.cat(candidate_noises, dim=0)  # [M*batch, C, H, W]
                    x_cur_expanded_probe = x_cur.repeat(probe_M, 1, 1, 1)
                    class_labels_expanded_probe = None
                    if class_labels is not None:
                        class_labels_expanded_probe = class_labels.repeat(probe_M, 1)
                    with torch.no_grad():
                        _, x0_probe = step(x_cur_expanded_probe, t_cur, t_next, i, all_noises, class_labels_expanded_probe)
                    x_probe_for_scoring = (x0_probe * 127.5 + 128).clip(0, 255).to(torch.uint8)
                    probe_timesteps = torch.zeros(x_probe_for_scoring.shape[0], device=device)
                    probe_scores = method_params.scorer(x_probe_for_scoring, class_labels_expanded_probe, probe_timesteps)
                    probe_scores = probe_scores.reshape(probe_M, -1)  # [M, batch]
                    # 使用 gain 作为难度度量：max - baseline
                    D_t = (probe_scores.max(dim=0).values - base_scores).mean().item()
                    # 注意：不计入 probe 成本，按用户要求忽略探针的 NFE
                B_rem = max(B_extra - B_used, 0.0)
                b_t = B_rem / B_extra  # 剩余预算比例
                tau_t = tau0 * (1 + alpha * (1 - b_t))  # 动态阈值
                steps_left = (len(t_steps) - 1) - (i + 1)
                allow_budget = (B_rem - heavy_cost) >= gamma_reserve * steps_left * heavy_cost and B_rem >= heavy_cost

                # 若难度不足或预算不够，则跳过 heavy search，直接 baseline 一步
                if not (D_t > tau_t and allow_budget):
                    x_next, _ = step(x_cur, t_cur, t_next, i, pivot_noise, class_labels)
                    if log_gain:
                        gains_per_step.append([0.0])
                        print(f"[EPS_GREEDY] step {i}: K_used={iterations_run}")
                    continue

            # For each timestep, store the best noise from each local search iteration
            # Shape: [K, batch_size, C, H, W]
            best_noises_this_timestep = []
            per_iter_gains = []  # 记录当前 timestep 内每次迭代的 gain
            iter_cost_used = 0.0  # 实际使用的 NFE 成本
            prev_best_scores = None
            break_inner = False
            last_best_scores = None
            
            # Run K iterations of local search
            for k in range(K):
                base_noise = pivot_noise
                
                # Generate N candidate noises by adding scaled random vectors
                candidate_noises = []
                for n in range(N):
                    # Decide between perturbed noise or fresh noise
                    if torch.rand(1, device=device) < (1 - eps):
                        # Use precomputed noise if available
                        if precomputed_noise is not None and i in precomputed_noise and k < precomputed_noise[i].shape[1] and n < precomputed_noise[i].shape[2]:
                            # Extract direction from precomputed noise
                            random_direction = precomputed_noise[i][:, k, n]
                            # Debug print
                            print(f"Shapes before reshape - base_noise: {base_noise.shape}, random_direction: {random_direction.shape}")
                            # Ensure shape matches base_noise
                            if random_direction.shape != base_noise.shape:
                                random_direction = random_direction.reshape(base_noise.shape)
                                print(f"Reshaped random_direction to: {random_direction.shape}")
                            # Normalize to get a unit vector
                            dims = tuple(range(1, random_direction.dim()))  # All dims except batch dim
                            random_direction = random_direction / torch.norm(random_direction, p=2, dim=dims, keepdim=True)
                        else:
                            # Generate random unit vector (by normalizing a Gaussian)
                            random_direction = torch.randn_like(base_noise)
                            # Normalize while adapting to tensor dimensions
                            dims = tuple(range(1, random_direction.dim()))  # All dims except batch dim
                            random_direction = random_direction / torch.norm(random_direction, p=2, dim=dims, keepdim=True)
                        
                        # Scale by random factor between 0 and lambda
                        # Use deterministic scaling if seed is fixed
                        if seed is not None:
                            # Derive scale from a hash of the current state
                            scale_seed = hash(f"{i}_{k}_{n}") % 1000 / 1000.0
                            # Create scale tensor with shape appropriate for the tensor dimensions
                            shape = [random_direction.shape[0]] + [1] * (random_direction.dim() - 1)
                            scale = torch.ones(shape, device=device) * scale_seed * lambda_param
                        else:
                            # Create scale tensor with shape appropriate for the tensor dimensions
                            shape = [random_direction.shape[0]] + [1] * (random_direction.dim() - 1)
                            scale = torch.rand(shape, device=device) * lambda_param
                        
                        # Debug print before addition
                        # print(f"Final shapes - base_noise: {base_noise.shape}, random_direction: {random_direction.shape}, scale: {scale.shape}")
                        
                        candidate_noise = base_noise + scale * random_direction
                    else:
                        # Use precomputed fresh noise if available
                        if precomputed_noise is not None and f'fresh_{i}_{k}_{n}' in precomputed_noise:
                            candidate_noise = precomputed_noise[f'fresh_{i}_{k}_{n}']
                        else:
                            # With probability eps, just use a fresh Gaussian sample
                            candidate_noise = torch.randn_like(x_cur)
                    
                    candidate_noises.append(candidate_noise)
                
                # Concatenate all candidate noises
                all_noises = torch.cat(candidate_noises, dim=0)  # [N*batch_size, C, H, W]
                
                # Repeat x_cur and class_labels for each candidate
                x_cur_expanded = x_cur.repeat(N, 1, 1, 1)  # [N*batch_size, C, H, W]
                class_labels_expanded = None
                if class_labels is not None:
                    class_labels_expanded = class_labels.repeat(N, 1)  # [N*batch_size, num_classes]
                
                # Run denoising step for all candidates at once
                x_candidates, x0_candidates = step(x_cur_expanded, t_cur, t_next, i, all_noises, class_labels_expanded)
                
                # Properly reshape based on actual dimensions - don't make assumptions
                total_images = x_candidates.shape[0]
                channels, height, width = x_cur.shape[1:]
                
                # Handle case where the actual shape doesn't match our expected N*batch_size
                effective_batch_size = total_images // N
                
                # Reshape using the effective batch size
                x_candidates = x_candidates.reshape(N, effective_batch_size, channels, height, width)
                x0_candidates = x0_candidates.reshape(N, effective_batch_size, channels, height, width)
                
                # Score candidates
                # Score based on predicted x0 (denoised result)
                # Reshape for scoring [N*batch_size, C, H, W]
                x_for_scoring = x0_candidates.reshape(-1, *x0_candidates.shape[2:])
                # Convert to proper format for scorer
                x_for_scoring = (x_for_scoring * 127.5 + 128).clip(0, 255).to(torch.uint8)
                # Use timesteps=0 for predicted clean images
                timesteps = torch.zeros(x_for_scoring.shape[0], device=device)
                
                # Make sure class labels are properly formatted for scorer
                if class_labels is not None:
                    scorer_class_labels = class_labels.repeat(N, 1)
                else:
                    scorer_class_labels = None
                
                # Get scores for all candidates
                scores = method_params.scorer(x_for_scoring, scorer_class_labels, timesteps)
                scores = scores.reshape(N, batch_size)  # [N, batch_size]
                
                # Find the best noise for each sample in the batch
                best_indices = scores.argmax(dim=0)  # [batch_size]
                iteration_best_scores = scores.max(dim=0).values  # [batch_size]
                last_best_scores = iteration_best_scores
                if log_gain:
                    if prev_best_scores is None:
                        per_iter_gains.append(0.0)
                    else:
                        gain_iter = (iteration_best_scores - prev_best_scores).mean().item()
                        per_iter_gains.append(gain_iter)
                
                # Gather best noise for each batch element
                candidate_noises_batch = all_noises.reshape(N, batch_size, *all_noises.shape[1:])  # [N, batch_size, C, H, W]
                
                # Gather best noise for each batch element
                new_pivot_noise = torch.stack([
                    candidate_noises_batch[best_idx, batch_idx] 
                    for batch_idx, best_idx in enumerate(best_indices)
                ])  # [batch_size, C, H, W]
                
                # Store the best noise from this iteration (clone to avoid reference issues)
                best_noises_this_timestep.append(new_pivot_noise.cpu().clone())
                
                # Update pivot_noise for next iteration
                pivot_noise = new_pivot_noise
                iter_cost_used += float(N * 2)  # 本次迭代成本（约 N*2 次 UNet 调用）
                iterations_run += 1
                
                # -------- K 内动态早停（第一个迭代不判断）--------
                if use_budget_scheduler and prev_best_scores is not None:
                    gain_mean = (iteration_best_scores - prev_best_scores).mean().item()
                    iter_tau = tau_t  # 使用同一动态阈值
                    if gain_mean <= iter_tau:
                        break_inner = True
                        break
                prev_best_scores = iteration_best_scores
            
            # Use the final best noise for this denoising step
            x_next, _ = step(x_cur, t_cur, t_next, i, pivot_noise, class_labels)
            print(f"[EPS_GREEDY_1] step {i}: K_used={iterations_run}")
            if use_budget_scheduler:
                # 实际成本 = 已迭代成本 + 最终一步 step 成本（约 2 次 UNet）
                actual_heavy_cost = iter_cost_used + 2.0
                B_used += actual_heavy_cost
            if log_gain:
                if per_iter_gains:
                    gains_per_step.append(per_iter_gains)
                else:
                    gains_per_step.append([0.0])
                print(f"[EPS_GREEDY] step {i}: K_used={iterations_run}")
        if log_gain:
            print(f"[EPS_GREEDY] Gain per timestep & per-iteration (mean over batch): {gains_per_step}")
    
    elif sampling_method == SamplingMethod.EPS_GREEDY_1:
        # EPS_GREEDY variant with adaptive K:
        # steps 0-1 & last 4 steps use K1, middle steps use K2
        lambda_param = method_params.lambda_param * np.sqrt(3 * 64 * 64)
        N = method_params.N
        eps = method_params.eps
        K1 = method_params.K1
        K2 = method_params.K2
        revert_on_negative = getattr(method_params, "revert_on_negative", False)
        record_eps1_trace = sampling_params.get("record_eps1_trace", False)
        replay_noise_plan = sampling_params.get("replay_noise_plan")
        eps1_best_noises = [] if record_eps1_trace else None
        eps1_final_noises = [] if record_eps1_trace else None
        if log_gain and revert_on_negative:
            print("[EPS_GREEDY_1] revert_on_negative enabled")
        if log_gain:
            gains_per_step = []
        
        num_steps_total = len(t_steps) - 1
        head_count = min(2, num_steps_total)  # 前2步（若存在）
        tail_count = min(4, max(num_steps_total - head_count, 0))  # 后4步（若存在）
        tail_start = num_steps_total - tail_count
        
        print(
            f"EPS_GREEDY_1 parameters: lambda={lambda_param / np.sqrt(3 * 64 * 64)}, "
            f"N={N}, K1={K1} (head {head_count} steps + tail {tail_count} steps), "
            f"K2={K2} (middle steps), eps={eps}"
        )
        
        # Use precomputed pivot noise if provided, otherwise generate a fresh one
        if precomputed_noise is not None and 'pivot' in precomputed_noise:
            pivot_noise = precomputed_noise['pivot']
        else:
            pivot_noise = torch.randn_like(x_next)
        
        # Main denoising loop over timesteps
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'):
            x_cur = x_next
            
            # Determine K based on current step: head/tail use K1, middle use K2
            is_head_tail = (i < head_count) or (i >= tail_start)
            K = K1 if is_head_tail else K2
            iterations_run = 0
            prev_best_scores = None  # 用于跨迭代比较（首迭代不触发负增益回退）
            
            # Initialize pivot noise with a fresh Gaussian sample
            if precomputed_noise is not None and f'pivot_{i}' in precomputed_noise:
                pivot_noise = precomputed_noise[f'pivot_{i}']
            else:
                pivot_noise = torch.randn_like(x_cur)
            
            # For each timestep, store the best noise from each local search iteration
            best_noises_this_timestep = [] if record_eps1_trace else []
            per_iter_gains = [] if log_gain else None
            
            if replay_noise_plan is not None:
                if i >= len(replay_noise_plan):
                    raise ValueError(f"replay_noise_plan length {len(replay_noise_plan)} is shorter than required steps {i+1}")
                pivot_noise = replay_noise_plan[i].to(device=x_cur.device, dtype=x_cur.dtype)
                iterations_run = 1
            else:
                # Run K iterations of local search
                for k in range(K):
                    iterations_run += 1
                    base_noise = pivot_noise
                    
                    # Generate N candidate noises by adding scaled random vectors
                    candidate_noises = []
                    for n in range(N):
                        # Decide between perturbed noise or fresh noise
                        if torch.rand(1, device=device) < (1 - eps):
                            # Use precomputed noise if available
                            if precomputed_noise is not None and i in precomputed_noise and k < precomputed_noise[i].shape[1] and n < precomputed_noise[i].shape[2]:
                                # Extract direction from precomputed noise
                                random_direction = precomputed_noise[i][:, k, n]
                                if random_direction.shape != base_noise.shape:
                                    random_direction = random_direction.reshape(base_noise.shape)
                                # Normalize to get a unit vector
                                dims = tuple(range(1, random_direction.dim()))
                                random_direction = random_direction / torch.norm(random_direction, p=2, dim=dims, keepdim=True)
                            else:
                                # Generate random unit vector (by normalizing a Gaussian)
                                random_direction = torch.randn_like(base_noise)
                                dims = tuple(range(1, random_direction.dim()))
                                random_direction = random_direction / torch.norm(random_direction, p=2, dim=dims, keepdim=True)
                            
                            # Scale by random factor between 0 and lambda
                            if seed is not None:
                                scale_seed = hash(f"{i}_{k}_{n}") % 1000 / 1000.0
                                shape = [random_direction.shape[0]] + [1] * (random_direction.dim() - 1)
                                scale = torch.ones(shape, device=device) * scale_seed * lambda_param
                            else:
                                shape = [random_direction.shape[0]] + [1] * (random_direction.dim() - 1)
                                scale = torch.rand(shape, device=device) * lambda_param
                            
                            candidate_noise = base_noise + scale * random_direction
                        else:
                            # Use precomputed fresh noise if available
                            if precomputed_noise is not None and f'fresh_{i}_{k}_{n}' in precomputed_noise:
                                candidate_noise = precomputed_noise[f'fresh_{i}_{k}_{n}']
                            else:
                                # With probability eps, just use a fresh Gaussian sample
                                candidate_noise = torch.randn_like(x_cur)
                    
                        candidate_noises.append(candidate_noise)
                    
                    # Concatenate all candidate noises
                    all_noises = torch.cat(candidate_noises, dim=0)  # [N*batch_size, C, H, W]
                    
                    # Repeat x_cur and class_labels for each candidate
                    x_cur_expanded = x_cur.repeat(N, 1, 1, 1)  # [N*batch_size, C, H, W]
                    class_labels_expanded = None
                    if class_labels is not None:
                        class_labels_expanded = class_labels.repeat(N, 1)  # [N*batch_size, num_classes]
                    
                    # Run denoising step for all candidates at once
                    x_candidates, x0_candidates = step(x_cur_expanded, t_cur, t_next, i, all_noises, class_labels_expanded)
                    
                    # Properly reshape based on actual dimensions
                    total_images = x_candidates.shape[0]
                    channels, height, width = x_cur.shape[1:]
                    effective_batch_size = total_images // N
                    
                    # Reshape using the effective batch size
                    x_candidates = x_candidates.reshape(N, effective_batch_size, channels, height, width)
                    x0_candidates = x0_candidates.reshape(N, effective_batch_size, channels, height, width)
                    
                    # Score candidates based on predicted x0 (denoised result)
                    x_for_scoring = x0_candidates.reshape(-1, *x0_candidates.shape[2:])
                    x_for_scoring = (x_for_scoring * 127.5 + 128).clip(0, 255).to(torch.uint8)
                    timesteps = torch.zeros(x_for_scoring.shape[0], device=device)
                    
                    # Make sure class labels are properly formatted for scorer
                    if class_labels is not None:
                        scorer_class_labels = class_labels.repeat(N, 1)
                    else:
                        scorer_class_labels = None
                    
                    # Get scores for all candidates
                    scores = method_params.scorer(x_for_scoring, scorer_class_labels, timesteps)
                    scores = scores.reshape(N, batch_size)  # [N, batch_size]
                    
                    # Find the best noise for each sample in the batch
                    best_indices = scores.argmax(dim=0)  # [batch_size]
                    iteration_best_scores = scores.max(dim=0).values  # [batch_size]
                    
                    gain_mean = None
                    if prev_best_scores is not None:
                        gain_mean = (iteration_best_scores - prev_best_scores).mean().item()
                    if log_gain:
                        per_iter_gains.append(0.0 if gain_mean is None else gain_mean)
                    
                    # Gather best noise for each batch element
                    candidate_noises_batch = all_noises.reshape(N, batch_size, *all_noises.shape[1:])  # [N, batch_size, C, H, W]
                    
                    new_pivot_noise = torch.stack([
                        candidate_noises_batch[best_idx, batch_idx] 
                        for batch_idx, best_idx in enumerate(best_indices)
                    ])  # [batch_size, C, H, W]
                    
                    # 负增益防护：若当前迭代平均提升为负，则保持上一轮 pivot、不更新（首迭代不触发）
                    if revert_on_negative and gain_mean is not None and gain_mean < 0:
                        if log_gain:
                            print(f"[EPS_GREEDY_1] step {i}, iter {k}: gain={gain_mean:.6f}<0, revert pivot")
                        continue  # 保持旧 pivot，下一轮仍用旧 pivot 进行探索
                    
                    # Store the best noise from this iteration
                    if record_eps1_trace:
                        best_noises_this_timestep.append(new_pivot_noise.cpu().clone())
                    
                    # Update pivot_noise for next iteration
                    pivot_noise = new_pivot_noise
                    prev_best_scores = iteration_best_scores
            
            # Use the final best noise for this denoising step
            x_next, _ = step(x_cur, t_cur, t_next, i, pivot_noise, class_labels)
            if log_gain:
                gains_per_step.append(per_iter_gains if per_iter_gains else [0.0])
            print(f"[EPS_GREEDY_1] step {i}: K_used={iterations_run}, revert_on_negative={revert_on_negative}")
            if record_eps1_trace:
                eps1_final_noises.append(pivot_noise.cpu().clone())
                eps1_best_noises.append(best_noises_this_timestep if best_noises_this_timestep is not None else [])

        if log_gain:
            print(f"[EPS_GREEDY_1] Gain per timestep & per-iteration (mean over batch): {gains_per_step}")

    elif sampling_method == SamplingMethod.EPS_GREEDY_ONLINE:
        # Online scheduling variant for EDM:
        # - Middle steps are "high" and use K1
        # - Low-value steps are the first 2 and last 4 steps; they share the remaining budget
        # 早停策略与 SD 一致：高价值区默认启用阈值判断，高区前两步不做早停；低价值区不早停。
        lambda_param = method_params.lambda_param * np.sqrt(3 * 64 * 64)
        N = method_params.N
        eps = method_params.eps
        K1 = method_params.K1
        K2_fallback = getattr(method_params, "K2", K1)  # only used for fallback budget if total_budget not provided
        revert_on_negative = getattr(method_params, "revert_on_negative", False)
        high_slack = getattr(method_params, "high_slack", 2)
        thresh_gain_coef = getattr(method_params, "thresh_gain_coef", 1.0)
        thresh_var_coef = getattr(method_params, "thresh_var_coef", 1.0)
        disable_early_stop_global = thresh_gain_coef <= 0 and thresh_var_coef <= 0
        all_historical_gains = []
        all_historical_variances = []
        total_steps = len(t_steps) - 1
        low_head = min(2, total_steps)
        low_tail = min(4, max(total_steps - low_head, 0))
        low_total = low_head + low_tail
        high_count = max(0, total_steps - low_total)
        total_budget = sampling_params.get("total_budget", None)
        if total_budget is None:
            # fallback: use legacy budget estimate
            total_budget = high_count * K1 + low_total * K2_fallback
        remaining_budget = total_budget - high_count * K1
        if remaining_budget <= 0:
            # 如果用户给的 total_budget 只够高价值步，回退为低价值步使用 K2_fallback
            remaining_budget = low_total * K2_fallback
            total_budget = high_count * K1 + remaining_budget
            print(
                f"[EPS_GREEDY_ONLINE][EDM][warn] remaining_budget<=0, "
                f"fallback low budget = {remaining_budget} (K2_fallback={K2_fallback})"
            )
        # build low schedule (front-load fractional budget)
        k_mean = remaining_budget / max(1, low_total)
        k_floor = int(np.floor(k_mean))
        k_floor = max(1, k_floor)
        extra = int(round(remaining_budget - k_floor * low_total))
        extra = max(min(extra, low_total), 0)
        # 动态预算跟踪，低区每步重算（使尾部能补足预算）
        high_used_acc = 0
        low_used_acc = 0
        high_steps_done = 0
        low_steps_done = 0

        print(
            f"[EPS_GREEDY_ONLINE][EDM] params: K1={K1}, total_budget={total_budget}, "
            f"low_head={low_head}, low_tail={low_tail}"
        )

        if log_gain:
            gains_per_step = []

        pivot_noise = torch.randn_like(x_next)

        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'):
            x_cur = x_next

            # determine if this step is low or high
            is_low_head = i < low_head
            tail_start = total_steps - low_tail
            is_low_tail = i >= tail_start
            is_low = is_low_head or is_low_tail
            force_full_k1 = (not is_low) and (i < 2)

            if is_low:
                remaining_low_steps = max(1, low_total - low_steps_done)
                remaining_high_steps = max(0, high_count - high_steps_done)
                # 预留未来高区预算 = 剩余高步数 * K1
                remaining_budget = total_budget - high_used_acc - low_used_acc - remaining_high_steps * K1
                k_mean = remaining_budget / remaining_low_steps
                k_floor = int(np.floor(k_mean))
                k_floor = max(1, k_floor)
                extra = int(round(remaining_budget - k_floor * remaining_low_steps))
                extra = max(min(extra, remaining_low_steps), 0)
                # 前置小数
                low_schedule_dynamic = [k_floor + 1] * extra + [k_floor] * (remaining_low_steps - extra)
                K_cur = low_schedule_dynamic[0]
            else:
                K_cur = K1

            iterations_run = 0
            prev_best_scores = None
            per_iter_gains = [] if log_gain else None
            timestep_scores_flat = []
            recent_gains_window = []

            # 早停窗口
            disable_early_stop = disable_early_stop_global
            watch_start = max(1, K_cur - high_slack)
            max_iter = K_cur + high_slack
            if is_low or force_full_k1 or disable_early_stop:
                watch_start = K_cur + 1  # 不触发早停
                max_iter = K_cur

            while iterations_run < max_iter:
                base_noise = pivot_noise

                candidate_noises = []
                for n in range(N):
                    if torch.rand(1, device=device) < (1 - eps):
                        random_direction = torch.randn_like(base_noise)
                        dims = tuple(range(1, random_direction.dim()))
                        random_direction = random_direction / torch.norm(random_direction, p=2, dim=dims, keepdim=True)
                        if seed is not None:
                            scale_seed = hash(f"{i}_{iterations_run}_{n}") % 1000 / 1000.0
                            shape = [random_direction.shape[0]] + [1] * (random_direction.dim() - 1)
                            scale = torch.ones(shape, device=device) * scale_seed * lambda_param
                        else:
                            shape = [random_direction.shape[0]] + [1] * (random_direction.dim() - 1)
                            scale = torch.rand(shape, device=device) * lambda_param
                        candidate_noise = base_noise + scale * random_direction
                    else:
                        candidate_noise = torch.randn_like(x_cur)
                    candidate_noises.append(candidate_noise)

                all_noises = torch.cat(candidate_noises, dim=0)
                x_cur_expanded = x_cur.repeat(N, 1, 1, 1)
                class_labels_expanded = None
                if class_labels is not None:
                    class_labels_expanded = class_labels.repeat(N, 1)

                x_candidates, x0_candidates = step(x_cur_expanded, t_cur, t_next, i, all_noises, class_labels_expanded)
                total_images = x_candidates.shape[0]
                channels, height, width = x_cur.shape[1:]
                effective_batch_size = total_images // N
                x0_candidates = x0_candidates.reshape(N, effective_batch_size, channels, height, width)
                x_for_scoring = x0_candidates.reshape(-1, *x0_candidates.shape[2:])
                x_for_scoring = (x_for_scoring * 127.5 + 128).clip(0, 255).to(torch.uint8)
                timesteps = torch.zeros(x_for_scoring.shape[0], device=device)

                scorer_class_labels = None
                if class_labels is not None:
                    scorer_class_labels = class_labels.repeat(N, 1)

                scores = method_params.scorer(x_for_scoring, scorer_class_labels, timesteps)
                scores = scores.reshape(N, batch_size)

                best_indices = scores.argmax(dim=0)
                iteration_best_scores = scores.max(dim=0).values

                # 记录方差
                timestep_scores_flat.extend(scores.detach().cpu().numpy().ravel().tolist())
                var_score = np.var(timestep_scores_flat) if len(timestep_scores_flat) > 1 else 0.0

                gain_mean = None
                if prev_best_scores is not None:
                    gain_mean = (iteration_best_scores - prev_best_scores).mean().item()

                if gain_mean is not None:
                    recent_gains_window.append(gain_mean)
                    if len(recent_gains_window) > 2:
                        recent_gains_window.pop(0)
                    gain_cur = float(np.mean(recent_gains_window))
                    all_historical_gains.append(gain_cur)
                else:
                    gain_cur = 0.0

                if log_gain:
                    per_iter_gains.append(gain_cur)

                candidate_noises_batch = all_noises.reshape(N, batch_size, *all_noises.shape[1:])
                new_pivot_noise = torch.stack([
                    candidate_noises_batch[best_idx, batch_idx]
                    for batch_idx, best_idx in enumerate(best_indices)
                ])

                iterations_run += 1

                if revert_on_negative and gain_mean is not None and gain_mean < 0:
                    # 仅回退 pivot，但仍允许计次、继续早停判定
                    if iterations_run >= watch_start:
                        if disable_early_stop_global:
                            pass
                        else:
                            hist_mean_gain = np.mean(all_historical_gains) if len(all_historical_gains) > 0 else 0.0
                            hist_mean_var = np.mean(all_historical_variances) if len(all_historical_variances) > 0 else 0.0
                            gain_thresh = -float("inf") if disable_early_stop_global else hist_mean_gain * thresh_gain_coef
                            var_thresh = -float("inf") if disable_early_stop_global else hist_mean_var * thresh_var_coef
                            if gain_cur < gain_thresh and var_score < var_thresh:
                                print(
                                    f"[EPS_GREEDY_ONLINE][EDM][EARLY_STOP][revert] step {i} iter {iterations_run}: "
                                    f"gain_cur={gain_cur:.6f}, gain_thresh={gain_thresh:.6f}, "
                                    f"var_cur={var_score:.6f}, var_thresh={var_thresh:.6f}, "
                                    f"hist_mean_gain={hist_mean_gain:.6f}, hist_mean_var={hist_mean_var:.6f}"
                                )
                                break
                    if iterations_run >= max_iter:
                        break
                    continue

                pivot_noise = new_pivot_noise
                prev_best_scores = iteration_best_scores

                if iterations_run >= watch_start:
                    hist_mean_gain = np.mean(all_historical_gains) if len(all_historical_gains) > 0 else 0.0
                    hist_mean_var = np.mean(all_historical_variances) if len(all_historical_variances) > 0 else 0.0
                    gain_thresh = -float("inf") if disable_early_stop_global else hist_mean_gain * thresh_gain_coef
                    var_thresh = -float("inf") if disable_early_stop_global else hist_mean_var * thresh_var_coef
                    if not disable_early_stop and gain_cur < gain_thresh and var_score < var_thresh:
                        print(
                            f"[EPS_GREEDY_ONLINE][EDM][EARLY_STOP] step {i} iter {iterations_run}: "
                            f"gain_cur={gain_cur:.6f}, gain_thresh={gain_thresh:.6f}, "
                            f"var_cur={var_score:.6f}, var_thresh={var_thresh:.6f}, "
                            f"hist_mean_gain={hist_mean_gain:.6f}, hist_mean_var={hist_mean_var:.6f}"
                        )
                        break

            # 记录本步方差统计
            var_timestep = np.var(timestep_scores_flat) if len(timestep_scores_flat) > 1 else 0.0
            all_historical_variances.append(var_timestep)

            x_next, _ = step(x_cur, t_cur, t_next, i, pivot_noise, class_labels)
            if log_gain:
                gains_per_step.append(per_iter_gains if per_iter_gains else [0.0])
            print(f"[EPS_GREEDY_ONLINE][EDM] step {i}: K_used={iterations_run}, is_low={is_low}")
            # 记录已用预算
            if is_low:
                low_used_acc += iterations_run
                low_steps_done += 1
            else:
                high_used_acc += iterations_run
                high_steps_done += 1

        if log_gain:
            print(f"[EPS_GREEDY_ONLINE][EDM] Gain per timestep & per-iteration (mean over batch): {gains_per_step}")

    else:  # NAIVE (default)
        for i, (t_cur, t_next) in tqdm.tqdm(list(enumerate(zip(t_steps[:-1], t_steps[1:]))), unit='step'):
            x_cur = x_next
            eps_i = torch.randn_like(x_next)
            x_next, _ = step(x_cur, t_cur, t_next, i, eps_i, class_labels)

    # Process final images for display
    image = (x_next * 127.5 + 128).clip(0, 255).to(torch.uint8)
    
    # Calculate final scores if needed - use the exact format needed by the scorer
    image_for_scoring = image.clone()  # Already in uint8 [0,255] format, which is what the model expects
    # Use timesteps=0 for fully denoised final images
    timesteps = torch.zeros(image_for_scoring.shape[0], device=device)
    scores = method_params.scorer(image_for_scoring, class_labels, timesteps)
    avg_score = scores.mean().item()
    print(f'Average score: {avg_score}')
    
    # Create and save the final grid (only if dest_path is provided)
    if dest_path is not None:
        print(f'Saving image grid to "{dest_path}"...')
        image = image.reshape(gridh, gridw, *image.shape[1:]).permute(0, 3, 1, 4, 2)
        image = image.reshape(gridh * net.img_resolution, gridw * net.img_resolution, net.img_channels)
        image = image.cpu().numpy()
        PIL.Image.fromarray(image, 'RGB').save(dest_path)
    
    print('Done.')
    
    eps1_trace = None
    if sampling_method == SamplingMethod.EPS_GREEDY_1 and record_eps1_trace:
        eps1_trace = {
            "best_noises_per_step": eps1_best_noises,
            "final_noise_per_step": eps1_final_noises,
        }
    
    # Return average score for multiple runs
    if eps1_trace is not None:
        return avg_score, eps1_trace
    return avg_score

#----------------------------------------------------------------------------

def main():
    model_root = 'https://nvlabs-fi-cdn.nvidia.com/edm/pretrained'
    g = 6
    
    latents = torch.randn([g * g, 3, 64, 64])
    class_labels = torch.eye(1000)[torch.randint(1000, size=[g * g])]
    
    # Set device
    device = torch.device('cuda')
    num_steps = 18
    
    # First load network to get info on the model's resolution, etc.
    with dnnlib.util.open_url(f'{model_root}/edm-imagenet-64x64-cond-adm.pkl') as f:
        net = pickle.load(f)['ema'].to(device)
    
    # Import scorers
    from scorers import BrightnessScorer, CompressibilityScorer, ImageNetScorer

    # Set a fixed seed for noise generation to ensure reproducibility
    torch.manual_seed(0)
    
    # List of scorers to use
    scorers = {
        'brightness': BrightnessScorer(dtype=torch.float32),
        'compressibility': CompressibilityScorer(dtype=torch.float32),
        'imagenet': ImageNetScorer(dtype=torch.float32)
    }

    method = SamplingMethod.NAIVE # choose between NAIVE, REJECTION_SAMPLING, BEAM_SEARCH, MCTS, ZERO_ORDER, EPS_GREEDY
    
    # Run experiments with each scorer
    for scorer_name, scorer in scorers.items():
        print(f"\n=== Running Zero-Order experiments with {scorer_name} scorer ===")
        
        output_path = f'naive_{scorer_name}.png'
        print(f"Generating {output_path}...")

        # Create parameters dictionary based on the method
        sampling_params = {'scorer': scorer}
        
        generate_image_grid(
            f'{model_root}/edm-imagenet-64x64-cond-adm.pkl', 
            output_path, 
            latents, 
            class_labels, 
            num_steps=num_steps, 
            S_churn=40, 
            S_min=0.05, 
            S_max=50, 
            S_noise=1.003, 
            sampling_method=method, 
            sampling_params=sampling_params, 
            gridw=g, 
            gridh=g,
        )

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------