"""Aesthetic and visual scoring providers for multi-signal reward computation.

Hierarchy:
- CLIPAestheticScorer: LAION aesthetic predictor & prompt-text cosine alignment. CLIP ViT-L/14 based. ~1.5GB. Runs on CPU/CUDA/MPS.
- ImageRewardScorer: BLIP-based. ~3GB. Prompt alignment. Runs on CPU/CUDA/MPS.
- HPSv3Scorer: Qwen2-VL-7B based. ~14GB. Only for PAID/large GPU setups.

Visual Metrics:
- calculate_visual_richness: Pixel standard deviation, canvas area coverage, and color entropy (penalizes blank / single-dot canvases).
- calculate_brush_utilization: Evaluates structured natural-media p5.brush features (washes, fills, bleeds) and detects text-rendering anti-cheats.
"""
from abc import ABC, abstractmethod
import os
import re
import math
import torch
import torch.nn as nn
from typing import Dict, Any, List


class AestheticScorer(ABC):
    """Abstract base class for aesthetic and semantic image scoring."""

    @abstractmethod
    def score(self, image_path: str, prompt: str) -> float:
        """Score an image's overall visual aesthetic quality."""
        ...

    @abstractmethod
    def score_prompt_alignment(self, image_path: str, prompt: str) -> float:
        """Score semantic alignment between the prompt and the rendered image."""
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
    """CLIP ViT-L/14 multimodal scorer. Computes both prompt alignment and visual aesthetic."""

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
        self._mlp = None

    def _get_image_features(self, image_path: str) -> torch.Tensor:
        from PIL import Image
        if not os.path.exists(image_path):
            raise RuntimeError(f"Image file not found: {image_path}")
        image = Image.open(image_path).convert("RGB")
        inputs = self._processor(images=image, return_tensors="pt").to(self._device)
        with torch.no_grad():
            img_feat = self._model.get_image_features(**inputs)
            img_feat = img_feat / img_feat.norm(p=2, dim=-1, keepdim=True)
        return img_feat

    def score(self, image_path: str, prompt: str = "") -> float:
        """Compute general aesthetic quality score."""
        try:
            image_features = self._get_image_features(image_path)
            with torch.no_grad():
                if self._mlp is not None:
                    aesthetic_score = self._mlp(image_features).item()
                else:
                    aesthetic_texts = [
                        "a beautiful masterpiece painting with rich textures and elegant composition",
                        "an empty white canvas or poorly drawn ugly scribbles",
                    ]
                    text_inputs = self._processor(
                        text=aesthetic_texts, return_tensors="pt", padding=True, truncation=True
                    ).to(self._device)
                    text_features = self._model.get_text_features(**text_inputs)
                    text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                    similarities = (image_features @ text_features.T).squeeze()
                    # Scale to roughly [0, 1]
                    aesthetic_score = float((similarities[0] - similarities[1] + 1.0) / 2.0)

            return max(0.0, min(1.0, float(aesthetic_score)))
        except Exception as e:
            raise RuntimeError(f"CLIP aesthetic scoring failed: {e}")

    def score_prompt_alignment(self, image_path: str, prompt: str) -> float:
        """Compute cosine similarity between rendered image and the specific art prompt."""
        if not prompt or not prompt.strip():
            return 0.5

        try:
            image_features = self._get_image_features(image_path)
            # Format prompt with artistic context for high CLIP fidelity
            formatted_prompt = f"a generative watercolor painting of {prompt.strip()}"
            
            with torch.no_grad():
                text_inputs = self._processor(
                    text=[formatted_prompt, "a blank solid white background with no art"],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self._device)
                
                text_features = self._model.get_text_features(**text_inputs)
                text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)

                # Similarity against prompt vs similarity against blank negative
                sim_prompt = (image_features @ text_features[0:1].T).item()
                sim_negative = (image_features @ text_features[1:2].T).item()

                # Calibrate CLIP cosine similarity (typical range 0.15 - 0.32)
                # Differential scoring penalizes blank canvas matches
                differential = sim_prompt - sim_negative
                # differential usually sits around 0.05 for good matches, <= 0 for blank
                calibrated = (differential + 0.05) / 0.20
                score = max(0.0, min(1.0, float(calibrated)))

            return score
        except Exception as e:
            raise RuntimeError(f"CLIP prompt alignment scoring failed: {e}")

    @property
    def scorer_name(self) -> str:
        return "clip_aesthetic"


class ImageRewardScorer(AestheticScorer):
    """ImageReward scorer. BLIP-based. ~3GB."""

    def __init__(self, device="auto"):
        try:
            import ImageReward as RM
        except ImportError:
            raise RuntimeError("ImageReward library is not installed.")

        if device == "auto":
            dev = "cuda" if torch.cuda.is_available() else "mps" if (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()) else "cpu"
        else:
            dev = str(device)
        self._model = RM.load("ImageReward-v1.0", device=dev)

    def score(self, image_path: str, prompt: str = "") -> float:
        if not os.path.exists(image_path):
            raise RuntimeError(f"Image file not found: {image_path}")
        try:
            with torch.no_grad():
                result = self._model.score(prompt or "generative art painting", image_path)
            # Normalize ImageReward (-2 to +2 scale into 0 to 1)
            return max(0.0, min(1.0, (float(result) + 2.0) / 4.0))
        except Exception as e:
            raise RuntimeError(f"ImageReward scoring failed: {e}")

    def score_prompt_alignment(self, image_path: str, prompt: str) -> float:
        return self.score(image_path, prompt)

    @property
    def scorer_name(self) -> str:
        return "image_reward"


class HPSv3Scorer(AestheticScorer):
    """Real HPSv3 scorer. Uses Qwen2-VL-7B (~14GB)."""

    def __init__(self):
        try:
            from hpsv3 import HPSv3RewardInferencer
            self._inferencer = HPSv3RewardInferencer(device="cuda")
        except ImportError:
            raise RuntimeError("HPSv3 library is not installed.")

    def score(self, image_path: str, prompt: str = "") -> float:
        if not os.path.exists(image_path):
            raise RuntimeError(f"Image file not found: {image_path}")
        try:
            rewards = self._inferencer.reward([prompt or "generative art"], image_paths=[image_path])
            return float(rewards[0][0].item())
        except Exception as e:
            raise RuntimeError(f"HPSv3 scoring failed: {e}")

    def score_prompt_alignment(self, image_path: str, prompt: str) -> float:
        return self.score(image_path, prompt)

    @property
    def scorer_name(self) -> str:
        return "hpsv3"


# =====================================================================
# Pixel Space & Anti-Cheat Visual Metrics
# =====================================================================

def calculate_visual_richness(image_path: str) -> Dict[str, Any]:
    """Inspect rendered canvas pixel distribution.
    
    Penalizes:
    - Blank / solid white or solid black canvases (zero standard deviation)
    - Single-dot or tiny stroke hacks (low active canvas coverage)
    - Monochromatic outputs with no palette spread
    
    Returns:
        Dict with keys:
        - richness_score: float in [0.0, 1.0]
        - pixel_std: float
        - coverage_ratio: float
        - color_count: int
        - is_blank: bool
    """
    if not image_path or not os.path.exists(image_path):
        return {
            "richness_score": 0.0,
            "pixel_std": 0.0,
            "coverage_ratio": 0.0,
            "color_count": 0,
            "edge_variance": 0.0,
            "is_blank": True,
            "critique": "[FAIL] Image file not found",
        }

    try:
        from PIL import Image
        import numpy as np

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img, dtype=np.float32)

        # 1. Global Pixel Standard Deviation across channels
        std_per_channel = np.std(arr, axis=(0, 1))
        pixel_std = float(np.mean(std_per_channel))

        # Blank Canvas Threshold: std < 4.0 means nearly monochromatic / empty canvas
        if pixel_std < 4.0:
            return {
                "richness_score": 0.0,
                "pixel_std": pixel_std,
                "coverage_ratio": 0.0,
                "color_count": 1,
                "edge_variance": 0.0,
                "is_blank": True,
                "critique": "[FAIL] Blank or monochromatic canvas detected (pixel std {:.1f} < 4.0)".format(pixel_std),
            }

        # 2. Active Canvas Coverage (percentage of pixels that differ from the background corner)
        y_sample = min(5, max(0, arr.shape[0] - 1))
        x_sample = min(5, max(0, arr.shape[1] - 1))
        bg_color = arr[y_sample, x_sample, :]
        diff = np.sqrt(np.sum((arr - bg_color) ** 2, axis=-1))
        active_pixels = np.sum(diff > 15.0)  # Distinct from background
        total_pixels = arr.shape[0] * arr.shape[1]
        coverage_ratio = float(active_pixels / max(1, total_pixels))

        # Penalty for tiny single-dot (coverage < 1%) or completely filled rectangle (coverage == 0)
        if coverage_ratio < 0.01:
            coverage_score = 0.0
        elif coverage_ratio < 0.08:
            coverage_score = coverage_ratio / 0.08  # Linear ramp for small sketches
        elif coverage_ratio <= 0.85:
            coverage_score = 1.0  # Sweet spot for artwork with balanced negative space
        else:
            coverage_score = max(0.6, 1.0 - (coverage_ratio - 0.85) * 2.0)  # Slight penalty for 100% solid fill

        # 3. Color Palette Diversity (quantize into 4-bit per channel color space)
        quantized = (arr // 32).astype(np.int32)
        color_keys = quantized[:, :, 0] * 64 + quantized[:, :, 1] * 8 + quantized[:, :, 2]
        color_count = len(np.unique(color_keys))
        
        # Color diversity score: 1 color = 0.1, 5+ colors = 0.8, 10+ colors = 1.0
        color_score = min(1.0, max(0.1, color_count / 10.0))

        # 4. Laplacian Edge Variance (measures texture sharpness and structural detail)
        gray = np.mean(arr, axis=-1)
        if gray.shape[0] >= 3 and gray.shape[1] >= 3:
            laplacian = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
                - 4.0 * gray[1:-1, 1:-1]
            )
            edge_variance = float(np.var(laplacian)) if laplacian.size > 0 else 0.0
        else:
            edge_variance = 0.0
        # Normalize: low edge variance (< 50) = flat, high (> 500) = rich texture
        edge_score = min(1.0, max(0.0, (edge_variance - 50.0) / 450.0))

        # Standard deviation score: std 4.0 -> 0.0, std 35.0+ -> 1.0
        std_score = min(1.0, max(0.0, (pixel_std - 4.0) / 30.0))

        # Composite visual richness (now includes edge sharpness)
        richness = 0.30 * coverage_score + 0.25 * std_score + 0.20 * color_score + 0.25 * edge_score
        richness_score = max(0.0, min(1.0, float(richness)))

        # Build diagnostic critique string
        critiques = []
        if coverage_ratio < 0.05:
            critiques.append("[CRITIQUE] Very low canvas coverage ({:.0%}) — most of the canvas is empty".format(coverage_ratio))
        elif coverage_ratio > 0.85:
            critiques.append("[CRITIQUE] Overfilled canvas ({:.0%}) — no negative space or breathing room".format(coverage_ratio))
        else:
            critiques.append("[GOOD] Balanced canvas coverage ({:.0%})".format(coverage_ratio))
        
        if color_count < 3:
            critiques.append("[CRITIQUE] Very limited palette ({} distinct hues)".format(color_count))
        elif color_count >= 8:
            critiques.append("[EXCELLENT] Rich color palette ({} distinct hues)".format(color_count))
        else:
            critiques.append("[GOOD] Adequate palette ({} hues)".format(color_count))
        
        if edge_variance < 100:
            critiques.append("[CRITIQUE] Flat/blurry composition (edge variance {:.0f})".format(edge_variance))
        elif edge_variance > 500:
            critiques.append("[EXCELLENT] Strong texture detail (edge variance {:.0f})".format(edge_variance))
        else:
            critiques.append("[GOOD] Decent structural detail (edge variance {:.0f})".format(edge_variance))

        return {
            "richness_score": round(richness_score, 4),
            "pixel_std": round(pixel_std, 2),
            "coverage_ratio": round(coverage_ratio, 4),
            "color_count": int(color_count),
            "edge_variance": round(edge_variance, 2),
            "is_blank": False,
            "critique": " | ".join(critiques),
        }

    except Exception:
        return {
            "richness_score": 0.0,
            "pixel_std": 0.0,
            "coverage_ratio": 0.0,
            "color_count": 0,
            "edge_variance": 0.0,
            "is_blank": True,
            "critique": "[FAIL] Image analysis error",
        }


def calculate_brush_utilization(code: str) -> Dict[str, Any]:
    """Inspect generated p5.js code to evaluate proper p5.brush and natural-media usage.
    
    Anti-Cheat Protections:
    - Text-in-image hack detection: calls to text(), textFont(), textSize() to fool CLIP -> Zero reward!
    - Trivial code hack: code shorter than 120 chars -> Zero reward!
    """
    if not code or not isinstance(code, str):
        return {"brush_score": 0.0, "features_used": [], "has_cheat": True}

    clean_code = code.strip()

    # 1. Anti-Cheat: Text primitive detection
    text_calls = re.findall(r'\btext\s*\(|\btextFont\s*\(|\btextSize\s*\(', clean_code)
    if text_calls:
        return {
            "brush_score": 0.0,
            "features_used": ["prohibited_text_primitive"],
            "has_cheat": True,
            "cheat_reason": "TEXT_IN_CANVAS_HACK"
        }

    # 2. Length check: Generative art code must have sufficient structure
    if len(clean_code) < 120:
        return {
            "brush_score": 0.0,
            "features_used": ["too_short"],
            "has_cheat": True,
            "cheat_reason": "TRIVIAL_CODE_LENGTH"
        }

    # 3. Feature extraction
    features = []
    score = 0.0

    # Initialization features
    if "brush.scaleBrushes" in clean_code:
        score += 0.20
        features.append("scaleBrushes")
    if "brush.load" in clean_code:
        score += 0.15
        features.append("brush_load")
    if "noLoop" in clean_code:
        score += 0.15
        features.append("noLoop")

    # Natural-media drawing features
    if re.search(r'brush\.(fill|fillBleed|fillTexture|wash)', clean_code):
        score += 0.25
        features.append("watercolor_fill")
    if re.search(r'brush\.(set|pick|stroke|strokeWeight)', clean_code):
        score += 0.15
        features.append("brush_stroke")
    if re.search(r'brush\.(rect|circle|line|polygon|spline|arc|beginShape)', clean_code):
        score += 0.20
        features.append("brush_primitives")
    if re.search(r'\bfor\s*\(|\bwhile\s*\(', clean_code):
        score += 0.10
        features.append("algorithmic_iteration")

    # Color definitions (hex codes or RGB tuples)
    if re.search(r'#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3}|color\s*\(', clean_code):
        score += 0.10
        features.append("color_palette")

    final_score = max(0.0, min(1.0, score))
    
    # Build diagnostic critique
    critiques = []
    if "scaleBrushes" in features:
        critiques.append("[GOOD] brush.scaleBrushes() called for proper brush sizing")
    else:
        critiques.append("[CRITIQUE] Missing brush.scaleBrushes() — brushes may render at microscopic size")
    
    if "watercolor_fill" in features:
        critiques.append("[GOOD] Natural media fills detected (wash/fill/bleed)")
    else:
        critiques.append("[CRITIQUE] No watercolor/wash fills — artwork may look flat")
    
    if "brush_primitives" in features:
        critiques.append("[GOOD] Brush drawing primitives used (shapes/lines/splines)")
    elif "brush_stroke" in features:
        critiques.append("[GOOD] Brush strokes configured")
    else:
        critiques.append("[CRITIQUE] No brush drawing calls — no visible brush strokes")
    
    if "algorithmic_iteration" in features:
        critiques.append("[GOOD] Algorithmic iteration (for/while loops) for generative patterns")
    
    return {
        "brush_score": round(final_score, 4),
        "features_used": features,
        "has_cheat": False,
        "cheat_reason": None,
        "critique": " | ".join(critiques),
    }


def create_aesthetic_scorer(config=None) -> AestheticScorer:
    """Factory function to create the appropriate aesthetic scorer."""
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
