#!/usr/bin/env python3
"""E3a post-hoc scoring: sweep archived PNGs with CLIP / ImageReward / Aesthetic.

Usage:
  python posthoc_score.py --day0 results/day0 --out results/posthoc_scores.csv [--metrics clip,imagereward] [--configs cfg1,cfg2]

Prompt recovery: image filenames are <prefix>_<promptIdx>[_rX].png; prompt text is
re-read from the CSV used by the config (prompts.csv for e1/e2/e5/e6, drawbench200.csv for e4).
Run inside conda env diffusion-tts on any GPU (CUDA_VISIBLE_DEVICES).
"""
import argparse, csv, os, re, sys, glob
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

def load_prompts(csv_path):
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    if rows and rows[0] and "prompt" in rows[0][0].lower():
        rows = rows[1:]
    return [r[0] for r in rows if r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day0", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "day0"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "posthoc_scores.csv"))
    ap.add_argument("--metrics", default="clip,imagereward,aesthetic")
    ap.add_argument("--configs", default=None, help="comma list; default = all dirs with PNGs")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    metrics = args.metrics.split(",")
    scorers = {}
    if "clip" in metrics or "aesthetic" in metrics:
        from transformers import CLIPModel, CLIPProcessor
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval()
        proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        scorers["_clip"] = (clip, proc)
    if "aesthetic" in metrics:
        import torch.nn as nn
        head = nn.Linear(768, 1)
        sd = torch.load("/mindopt/caoyuan/weights/sac+logos+ava1-l14-linearMSE.pth", map_location="cpu")
        # sac+logos head is an MLP; fall back gracefully if structure differs
        try:
            from collections import OrderedDict
            class MLP(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(768, 1024), nn.Dropout(0.2), nn.Linear(1024, 128), nn.Dropout(0.2),
                        nn.Linear(128, 64), nn.Dropout(0.1), nn.Linear(64, 16), nn.Linear(16, 1))
                def forward(self, x): return self.layers(x)
            m = MLP(); m.load_state_dict(sd); head = m
        except Exception as e:
            print(f"[aesthetic] load failed: {e}; skipping aesthetic")
            metrics = [x for x in metrics if x != "aesthetic"]
        head = head.to(device).eval()
        scorers["_aes"] = head
    if "imagereward" in metrics:
        import ImageReward as ir
        scorers["_ir"] = ir.load("ImageReward-v1.0", device=device)

    prompts_main = load_prompts(os.path.join(REPO, "prompts.csv"))
    prompts_db = load_prompts(os.path.join(REPO, "drawbench200.csv"))

    cfgs = args.configs.split(",") if args.configs else sorted(
        os.path.basename(d) for d in glob.glob(os.path.join(args.day0, "*"))
        if os.path.isdir(d) and glob.glob(os.path.join(d, "*.png")))
    from PIL import Image
    rows_out = []
    fre = re.compile(r"_(\d+)(?:_r(\d+))?\.png$")
    for cfg in cfgs:
        plist = prompts_db if cfg.startswith("e4_db") else prompts_main
        pngs = sorted(glob.glob(os.path.join(args.day0, cfg, "*.png")))
        print(f"[{cfg}] {len(pngs)} images")
        for p in pngs:
            m = fre.search(os.path.basename(p))
            if not m: continue
            pi = int(m.group(1)); seed = m.group(2) or "0"
            if pi >= len(plist): continue
            prompt = plist[pi]
            img = Image.open(p).convert("RGB")
            rec = {"config": cfg, "prompt_idx": pi, "seed": seed, "file": os.path.basename(p)}
            with torch.no_grad():
                if "_clip" in scorers:
                    clip, proc = scorers["_clip"]
                    ins = proc(images=img, text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
                    ie = clip.get_image_features(pixel_values=ins["pixel_values"])
                    te = clip.get_text_features(input_ids=ins["input_ids"], attention_mask=ins["attention_mask"])
                    ie = ie / ie.norm(dim=-1, keepdim=True); te = te / te.norm(dim=-1, keepdim=True)
                    rec["clip"] = float((ie * te).sum()) * 100  # CLIP score convention x100
                    if "_aes" in scorers:
                        rec["aesthetic"] = float(scorers["_aes"](ie))
                if "_ir" in scorers:
                    rec["imagereward"] = float(scorers["_ir"].score(prompt, img))
            rows_out.append(rec)
    if rows_out:
        keys = ["config", "prompt_idx", "seed", "file"] + [k for k in ("clip", "aesthetic", "imagereward") if k in rows_out[0]]
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows_out)
        print(f"wrote {args.out} ({len(rows_out)} rows)")

if __name__ == "__main__":
    main()
