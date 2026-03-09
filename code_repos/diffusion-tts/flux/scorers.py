"""
Scorers for FLUX backend.
Reuses the same scoring functions as SD (brightness, compressibility).
"""
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io


class Scorer(torch.nn.Module):
    """Base class for all scorers"""
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.dtype = dtype
        self.eval()

    @torch.no_grad()
    def __call__(self, images, prompts, timesteps=None):
        raise NotImplementedError("Subclasses must implement __call__")


class BrightnessScorer(Scorer):
    """Score images based on perceived luminance (higher = brighter)"""
    def __init__(self, dtype=torch.float32):
        super().__init__(dtype)

    @torch.no_grad()
    def __call__(self, images, prompts=None, timesteps=None):
        # Handle list of images
        if isinstance(images, list):
            processed = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    if img.device.type != "cpu":
                        img = img.detach().cpu()
                    if img.dim() == 4:
                        processed.append(img)
                    else:
                        processed.append(img.unsqueeze(0))
                else:
                    tensor_img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0)
                    processed.append(tensor_img)
            images = torch.cat(processed, dim=0)

        # Convert uint8 to float
        if images.dtype == torch.uint8:
            images = images.float() / 255.0

        # Apply perceived luminance formula: 0.2126*R + 0.7152*G + 0.0722*B
        if images.size(1) == 3:
            weights = torch.tensor([0.2126, 0.7152, 0.0722], device=images.device).view(1, 3, 1, 1)
            luminance = (images * weights).sum(dim=1).mean(dim=(1, 2))
        else:
            if images.dim() == 4:
                luminance = images.mean(dim=(1, 2, 3))
            else:
                luminance = images.mean(dim=(1, 2))

        luminance = torch.clamp(luminance, 0.0, 1.0)
        return luminance


class CompressibilityScorer(Scorer):
    """Score images based on JPEG compressibility (higher = more compressible)"""
    def __init__(self, quality=80, min_size=0, max_size=150000, dtype=torch.float32):
        super().__init__(dtype)
        self.quality = quality
        self.min_size = min_size
        self.max_size = max_size

    @torch.no_grad()
    def __call__(self, images, prompts=None, timesteps=None):
        if isinstance(images, torch.Tensor):
            if images.dim() == 4:
                return torch.tensor([self._calculate_score(img.cpu().numpy()) for img in images])
            else:
                return torch.tensor([self._calculate_score(images.cpu().numpy())])
        elif isinstance(images, list):
            scores = []
            for img in images:
                if isinstance(img, torch.Tensor):
                    scores.append(self._calculate_score(img.cpu().numpy()))
                else:
                    scores.append(self._calculate_score(np.array(img)))
            return torch.tensor(scores)
        else:
            return torch.tensor([self._calculate_score(np.array(images))])

    def _calculate_score(self, image):
        # Handle CHW format
        if image.ndim == 3:
            if image.shape[0] == 1 or image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            if image.shape[2] == 1:
                image = image.squeeze(2)

        # Convert to uint8
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        buffer = io.BytesIO()
        img = Image.fromarray(image)
        img.save(buffer, format="JPEG", quality=self.quality)
        compressed_size = len(buffer.getvalue())

        # Normalize: 1.0 = highly compressible (small size)
        normalized_score = 1.0 - min(1.0, max(0.0,
            (compressed_size - self.min_size) / (self.max_size - self.min_size)))
        return normalized_score
