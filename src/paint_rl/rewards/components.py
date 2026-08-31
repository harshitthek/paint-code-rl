from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import time
from typing import Optional

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
    def __init__(self, weight: float):
        self._weight = weight
        
    @property
    def name(self) -> str:
        return "compile"
        
    @property
    def version(self) -> str:
        return "1.0"
        
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

class AestheticRewardComponent(RewardComponent):
    def __init__(self, weight: float, scorer):
        self._weight = weight
        self.scorer = scorer
        
    @property
    def name(self) -> str:
        return "aesthetic"
        
    @property
    def version(self) -> str:
        return "1.0"
        
    def compute(self, image_path: str, prompt: str, **kwargs) -> RewardResult:
        start_t = time.time()
        error_state = None
        raw_score = 0.0
        
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
    def __init__(self, weight: float, judge_provider):
        self._weight = weight
        self.judge = judge_provider
        
    @property
    def name(self) -> str:
        return "pairwise"
        
    @property
    def version(self) -> str:
        return "1.0"
        
    def compute(self, candidate_path: str, reference_path: str, prompt: str, **kwargs) -> RewardResult:
        start_t = time.time()
        error_state = None
        raw_score = 0.0
        status = "none"
        
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
