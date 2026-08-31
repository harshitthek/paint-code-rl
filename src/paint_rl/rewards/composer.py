"""RewardComposer — aggregates multiple RewardComponents into a total reward.

Each component receives the kwargs it needs. The composer:
1. Runs compile reward (needs render_result)
2. Runs aesthetic reward (needs image_path, prompt) — only if render succeeded
3. Runs pairwise reward (needs candidate image, reference image, prompt) — only if render succeeded
4. Validates total (no NaN, no Inf)
"""
from typing import List, Dict, Any
import math

from .components import RewardComponent, RewardResult, CompileRewardComponent, AestheticRewardComponent, PairwiseRewardComponent


class RewardComposer:
    def __init__(self, components: List[RewardComponent]):
        self.components = components

    def compute(self, render_result: dict, image_path: str = None,
                prompt: str = "", reference_path: str = None, **kwargs) -> Dict[str, Any]:
        """Compute all reward components and aggregate.

        Args:
            render_result: Dict from renderer with success/error info.
            image_path: Path to rendered PNG (if render succeeded).
            prompt: Art prompt text.
            reference_path: Path to reference image for pairwise comparison.

        Returns:
            Dict with total, error_class, and list of per-component RewardResults.
        """
        results: List[RewardResult] = []
        total_score = 0.0
        error_class = None
        render_success = render_result.get("success", False)

        for comp in self.components:
            try:
                if isinstance(comp, CompileRewardComponent):
                    res = comp.compute(render_result=render_result)
                elif isinstance(comp, AestheticRewardComponent):
                    if render_success and image_path:
                        res = comp.compute(image_path=image_path, prompt=prompt)
                    else:
                        res = RewardResult(
                            raw_score=0.0, weight=comp._weight, weighted_score=0.0,
                            component_name=comp.name, latency_ms=0.0,
                            error_state="SKIPPED_RENDER_FAILED"
                        )
                elif isinstance(comp, PairwiseRewardComponent):
                    if render_success and image_path and reference_path:
                        res = comp.compute(
                            candidate_path=image_path,
                            reference_path=reference_path,
                            prompt=prompt
                        )
                    else:
                        res = RewardResult(
                            raw_score=0.0, weight=comp._weight, weighted_score=0.0,
                            component_name=comp.name, latency_ms=0.0,
                            error_state="SKIPPED_RENDER_FAILED" if not render_success else "SKIPPED_NO_REFERENCE"
                        )
                else:
                    # Generic component — pass all kwargs
                    res = comp.compute(
                        render_result=render_result,
                        image_path=image_path,
                        prompt=prompt,
                        reference_path=reference_path,
                        **kwargs
                    )
            except Exception as e:
                res = RewardResult(
                    raw_score=0.0, weight=0.0, weighted_score=0.0,
                    component_name=comp.name, latency_ms=0.0,
                    error_state=f"COMPONENT_ERROR: {str(e)}"
                )

            results.append(res)

            # Only add to total if score is finite
            if math.isfinite(res.weighted_score):
                total_score += res.weighted_score
            else:
                error_class = error_class or f"NON_FINITE_{res.component_name.upper()}"

            if res.error_state and error_class is None:
                error_class = res.error_state

        # Validate total
        if not math.isfinite(total_score):
            total_score = 0.0
            error_class = error_class or "NON_FINITE_TOTAL"

        return {
            "total": total_score,
            "error_class": error_class,
            "components": results,
        }
