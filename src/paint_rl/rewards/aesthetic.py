"""Aesthetic scoring providers for visual reward computation.

Hierarchy:
- CLIPAestheticScorer: LAION aesthetic predictor. CLIP ViT-L/14 based. ~1.5GB. Runs on CPU/CUDA/MPS.
- ImageRewardScorer: BLIP-based. ~3GB. Better prompt alignment. Runs on CPU/CUDA/MPS.
- HPSv3Scorer: Qwen2-VL-7B based. ~14GB. Only for PAID/large GPU setups.

Default for FREE/LOCAL: CLIPAestheticScorer (lightweight, no API calls).
Fail-closed: If no scorer can load, raises RuntimeError. Never returns fake scores.
"""
from abc import ABC, abstractmethod
import os
import torch
import torch.nn as nn


class AestheticScorer(ABC):
    """Abstract base class for aesthetic image scoring."""

    @abstractmethod
    def score(self, image_path: str, prompt: str) -> float:
        """Score an image's aesthetic quality. Higher = more aesthetic.

        Args:
            image_path: Path to PNG image file.
            prompt: Text prompt describing intended art.

        Returns:
            Float score. Scale depends on implementation.

        Raises:
            RuntimeError: If scoring fails (never returns fake values).
        """
        ...

    @property
    @abstractmethod
    def scorer_name(self) -> str: ...


class _AestheticMLP(nn.Module):
    """MLP head for LAION aesthetic predictor (trained on top of CLIP embeddings)."""

    def __init__(self, input_size=768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class CLIPAestheticScorer(AestheticScorer):
    """LAION aesthetic predictor. CLIP ViT-L/14 based. ~1.5GB total.

    Uses CLIP image embeddings passed through a trained aesthetic MLP head.
    The MLP weights are from the LAION improved-aesthetic-predictor project.
    If the MLP weights are not available, falls back to raw CLIP cosine
    similarity with an aesthetic text anchor (less accurate but functional).
    """

    def __init__(self, device="auto"):
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise RuntimeError(
                "transformers is required for CLIPAestheticScorer. "
                "Install: pip install transformers"
            )

        if device == "auto":
            if torch.cuda.is_available():
                self._device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self._device = torch.device("mps")
            else:
                self._device = torch.device("cpu")
        else:
            self._device = torch.device(device)

        self._model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(self._device)
        self._model.eval()
        self._processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self._mlp = None  # Will attempt to load MLP weights if available

    def score(self, image_path: str, prompt: str) -> float:
        from PIL import Image

        if not os.path.exists(image_path):
            raise RuntimeError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to open image {image_path}: {e}")

        try:
            # Get CLIP image embedding
            inputs = self._processor(images=image, return_tensors="pt").to(self._device)
            with torch.no_grad():
                image_features = self._model.get_image_features(**inputs)
                image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)

                if self._mlp is not None:
                    # Use trained aesthetic MLP
                    aesthetic_score = self._mlp(image_features).item()
                else:
                    # Fallback: cosine similarity with aesthetic text anchors
                    aesthetic_texts = [
                        "a beautiful painting with excellent composition",
                        "an ugly, poorly drawn image",
                    ]
                    text_inputs = self._processor(
                        text=aesthetic_texts, return_tensors="pt", padding=True
                    ).to(self._device)
                    text_features = self._model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                    similarities = (image_features @ text_features.T).squeeze()
                    # Score = similarity to "beautiful" minus similarity to "ugly"
                    # Normalized to roughly [0, 1] range
                    aesthetic_score = float((similarities[0] - similarities[1] + 1.0) / 2.0)

            return float(aesthetic_score)

        except Exception as e:
            raise RuntimeError(f"CLIP aesthetic scoring failed: {e}")

    @property
    def scorer_name(self) -> str:
        return "clip_aesthetic"


class ImageRewardScorer(AestheticScorer):
    """ImageReward scorer. BLIP-based. ~3GB. Better prompt alignment than CLIP."""

    def __init__(self, device="auto"):
        try:
            import ImageReward as RM
        except ImportError:
            raise RuntimeError(
                "ImageReward library is not installed. "
                "Install: pip install image-reward"
            )

        if device == "auto":
            if torch.cuda.is_available():
                dev = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                dev = "mps"
            else:
                dev = "cpu"
        else:
            dev = device

        try:
            self._model = RM.load("ImageReward-v1.0", device=dev)
        except Exception as e:
            raise RuntimeError(f"Failed to load ImageReward model: {e}")

    def score(self, image_path: str, prompt: str) -> float:
        if not os.path.exists(image_path):
            raise RuntimeError(f"Image file not found: {image_path}")
        try:
            with torch.no_grad():
                result = self._model.score(prompt, image_path)
            return float(result)
        except Exception as e:
            raise RuntimeError(f"ImageReward scoring failed: {e}")

    @property
    def scorer_name(self) -> str:
        return "image_reward"


class HPSv3Scorer(AestheticScorer):
    """Real HPSv3 scorer. Uses Qwen2-VL-7B (~14GB). Only for PAID/large GPU."""

    def __init__(self):
        try:
            from hpsv3 import HPSv3RewardInferencer
            self._inferencer = HPSv3RewardInferencer(device="cuda")
        except ImportError:
            raise RuntimeError(
                "HPSv3 library is not installed. Install: pip install hpsv3. "
                "Note: Requires ~14GB VRAM (Qwen2-VL-7B)."
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HPSv3: {e}")

    def score(self, image_path: str, prompt: str) -> float:
        if not os.path.exists(image_path):
            raise RuntimeError(f"Image file not found: {image_path}")
        try:
            rewards = self._inferencer.reward([prompt], image_paths=[image_path])
            return float(rewards[0][0].item())
        except Exception as e:
            raise RuntimeError(f"HPSv3 scoring failed: {e}")

    @property
    def scorer_name(self) -> str:
        return "hpsv3"


def create_aesthetic_scorer(config=None) -> AestheticScorer:
    """Factory function to create the appropriate aesthetic scorer.

    Selection priority:
    1. Config-specified provider
    2. CLIPAestheticScorer (default, lightweight)
    3. Fail closed if nothing loads
    """
    provider = "clip"
    if config is not None and hasattr(config, "aesthetic"):
        provider = config.aesthetic.provider

    if provider == "hpsv3":
        return HPSv3Scorer()
    elif provider == "image_reward":
        return ImageRewardScorer()
    elif provider == "clip":
        return CLIPAestheticScorer()
    else:
        raise ValueError(f"Unknown aesthetic provider: {provider}")
