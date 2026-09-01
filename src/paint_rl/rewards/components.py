from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Optional, Dict, Any

from paint_rl.rewards.aesthetic import calculate_visual_richness, calculate_brush_utilization


@dataclass
class RewardResult:
    raw_score: float
    weight: float
    weighted_score: float
    component_name: str
    latency_ms: float
    error_state: Optional[str] = None
    metadata: dict = field(default_factory=dict)


class RewardComponent(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...
    
    @property
    @abstractmethod  
    def version(self) -> str: ...
    
    @abstractmethod
    def compute(self, **kwargs) -> RewardResult: ...


class CompileRewardComponent(RewardComponent):
    """Verifiable execution gate: evaluates whether the JavaScript code rendered a canvas."""
    def __init__(self, weight: float = 0.15):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "compile"
        
    @property
    def version(self) -> str:
        return "2.0"
        
    def compute(self, render_result: dict, **kwargs) -> RewardResult:
        start_t = time.time()
        success = render_result.get("success", False)
        raw_score = 1.0 if success else 0.0
        error_state = render_result.get("error_classification") if not success else None
        
        latency_ms = (time.time() - start_t) * 1000
        
        return RewardResult(
            raw_score=raw_score,
            weight=self._weight,
            weighted_score=raw_score * self._weight,
            component_name=self.name,
            latency_ms=latency_ms,
            error_state=error_state,
            metadata={"render_result": render_result}
        )


class PromptAlignmentRewardComponent(RewardComponent):
    """Semantic alignment reward: computes cosine similarity between rendered image and prompt text."""
    def __init__(self, weight: float = 0.35, scorer=None):
        self._weight = weight
        self.scorer = scorer

    @property
    def name(self) -> str:
        return "prompt_alignment"

    @property
    def version(self) -> str:
        return "2.0"

    def compute(self, image_path: str, prompt: str, **kwargs) -> RewardResult:
        start_t = time.time()
        error_state = None
        raw_score = 0.0

        if not self.scorer:
            error_state = "SCORER_UNINITIALIZED"
        else:
            try:
                raw_score = self.scorer.score_prompt_alignment(image_path, prompt)
            except Exception as e:
                error_state = f"PROMPT_ALIGN_ERROR: {str(e)}"

        latency_ms = (time.time() - start_t) * 1000

        return RewardResult(
            raw_score=raw_score,
            weight=self._weight,
            weighted_score=raw_score * self._weight,
            component_name=self.name,
            latency_ms=latency_ms,
            error_state=error_state,
            metadata={"prompt": prompt}
        )


class VisualRichnessRewardComponent(RewardComponent):
    """Visual geometry & anti-cheat reward: penalizes blank canvases, single dots, and low-entropy pixels."""
    def __init__(self, weight: float = 0.25):
        self._weight = weight

    @property
    def name(self) -> str:
        return "visual_richness"

    @property
    def version(self) -> str:
        return "2.0"

    def compute(self, image_path: str, **kwargs) -> RewardResult:
        start_t = time.time()
        metrics = calculate_visual_richness(image_path)
        raw_score = metrics.get("richness_score", 0.0)
        error_state = "BLANK_CANVAS_DETECTED" if metrics.get("is_blank") else None

        latency_ms = (time.time() - start_t) * 1000

        return RewardResult(
            raw_score=raw_score,
            weight=self._weight,
            weighted_score=raw_score * self._weight,
            component_name=self.name,
            latency_ms=latency_ms,
            error_state=error_state,
            metadata=metrics
        )


class BrushUtilizationRewardComponent(RewardComponent):
    """Structural natural-media reward: rewards p5.brush features (washes/strokes) and stops text-cheats."""
    def __init__(self, weight: float = 0.15):
        self._weight = weight

    @property
    def name(self) -> str:
        return "brush_utilization"

    @property
    def version(self) -> str:
        return "2.0"

    def compute(self, code: str, **kwargs) -> RewardResult:
        start_t = time.time()
        metrics = calculate_brush_utilization(code)
        raw_score = metrics.get("brush_score", 0.0)
        error_state = metrics.get("cheat_reason") if metrics.get("has_cheat") else None

        latency_ms = (time.time() - start_t) * 1000

        return RewardResult(
            raw_score=raw_score,
            weight=self._weight,
            weighted_score=raw_score * self._weight,
            component_name=self.name,
            latency_ms=latency_ms,
            error_state=error_state,
            metadata=metrics
        )


class AestheticRewardComponent(RewardComponent):
    """Global aesthetic harmony reward."""
    def __init__(self, weight: float = 0.10, scorer=None):
        self._weight = weight
        self.scorer = scorer
        
    @property
    def name(self) -> str:
        return "aesthetic"
        
    @property
    def version(self) -> str:
        return "2.0"
        
    def compute(self, image_path: str, prompt: str = "", **kwargs) -> RewardResult:
        start_t = time.time()
        error_state = None
        raw_score = 0.0
        
        if not self.scorer:
            error_state = "SCORER_UNINITIALIZED"
        else:
            try:
                raw_score = self.scorer.score(image_path, prompt)
            except Exception as e:
                error_state = f"AESTHETIC_ERROR: {str(e)}"
            
        latency_ms = (time.time() - start_t) * 1000
        
        return RewardResult(
            raw_score=raw_score,
            weight=self._weight,
            weighted_score=raw_score * self._weight,
            component_name=self.name,
            latency_ms=latency_ms,
            error_state=error_state,
        )


class PairwiseRewardComponent(RewardComponent):
    """Relative visual preference reward via local/cloud VLM."""
    def __init__(self, weight: float = 0.60, judge_provider=None):
        self._weight = weight
        self.judge = judge_provider
        
    @property
    def name(self) -> str:
        return "pairwise"
        
    @property
    def version(self) -> str:
        return "2.0"
        
    def compute(self, candidate_path: str, reference_path: str, prompt: str, **kwargs) -> RewardResult:
        start_t = time.time()
        error_state = None
        raw_score = 0.0
        status = "none"
        
        if not self.judge:
            error_state = "JUDGE_UNINITIALIZED"
        else:
            try:
                pair_result = self.judge.compare(candidate_path=candidate_path, reference_path=reference_path, prompt=prompt)
                raw_score = pair_result.get("score", 0.0)
                status = pair_result.get("status", "none")
                if status in ("VLM_API_ERROR", "VLM_MALFORMED_RESPONSE"):
                    error_state = status
                    raw_score = 0.0
            except Exception as e:
                error_state = f"JUDGE_ERROR: {str(e)}"
            
        latency_ms = (time.time() - start_t) * 1000
        
        return RewardResult(
            raw_score=raw_score,
            weight=self._weight,
            weighted_score=raw_score * self._weight,
            component_name=self.name,
            latency_ms=latency_ms,
            error_state=error_state,
            metadata={"judge_status": status}
        )
