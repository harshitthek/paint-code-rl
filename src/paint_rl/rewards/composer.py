"""RewardComposer — aggregates multiple RewardComponents into a total reward.

Execution Pipeline:
1. Runs Compile reward (verifiable execution gate)
2. Runs Visual Richness (pixel space entropy / anti-blank check)
3. Runs Prompt Alignment (semantic text-image CLIP similarity)
4. Runs Brush Utilization (natural media structure & anti-cheat check)
5. Runs Pairwise / Aesthetic rewards (if configured)
6. Validates total bounds and returns full component metadata
"""
from typing import List, Dict, Any
import math

from .components import (
    RewardComponent,
    RewardResult,
    CompileRewardComponent,
    PromptAlignmentRewardComponent,
    VisualRichnessRewardComponent,
    BrushUtilizationRewardComponent,
    AestheticRewardComponent,
    PairwiseRewardComponent,
)


class RewardComposer:
    def __init__(self, components: List[Any]):
        unpacked = []
        for c in components:
            if isinstance(c, tuple):
                comp, weight = c
                if hasattr(comp, "_weight"):
                    comp._weight = weight
                unpacked.append(comp)
            else:
                unpacked.append(c)
        self.components = unpacked

    def compute(self, render_result: dict, image_path: str = None,
                prompt: str = "", code: str = "", reference_path: str = None, **kwargs) -> Dict[str, Any]:
        """Compute all reward components and aggregate.

        Args:
            render_result: Dict from renderer with success/error info.
            image_path: Path to rendered PNG (if render succeeded).
            prompt: Art prompt text.
            code: Generated p5.js code string.
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
                elif isinstance(comp, VisualRichnessRewardComponent):
                    if render_success and image_path:
                        res = comp.compute(image_path=image_path)
                    else:
                        res = RewardResult(
                            raw_score=0.0, weight=comp._weight, weighted_score=0.0,
                            component_name=comp.name, latency_ms=0.0,
                            error_state="SKIPPED_RENDER_FAILED"
                        )
                elif isinstance(comp, PromptAlignmentRewardComponent):
                    if render_success and image_path:
                        res = comp.compute(image_path=image_path, prompt=prompt)
                    else:
                        res = RewardResult(
                            raw_score=0.0, weight=comp._weight, weighted_score=0.0,
                            component_name=comp.name, latency_ms=0.0,
                            error_state="SKIPPED_RENDER_FAILED"
                        )
                elif isinstance(comp, BrushUtilizationRewardComponent):
                    res = comp.compute(code=code)
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
                        code=code,
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
            "total": round(total_score, 4),
            "error_class": error_class,
            "components": results,
        }

    def generate_scorecard(self, compute_result: Dict[str, Any], prompt: str = "") -> str:
        """Generate a human-readable diagnostic scorecard from compute results.
        
        Args:
            compute_result: Output from self.compute()
            prompt: The art prompt for context
            
        Returns:
            Multi-line string with structured diagnostic report.
        """
        lines = []
        lines.append("=" * 60)
        lines.append("  DIAGNOSTIC SCORECARD")
        if prompt:
            lines.append(f"  Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        lines.append("=" * 60)
        
        total = compute_result.get("total", 0.0)
        error_class = compute_result.get("error_class")
        
        for result in compute_result.get("components", []):
            name = result.component_name.upper()
            raw = result.raw_score
            weighted = result.weighted_score
            
            # Determine quality tier
            if result.error_state:
                tier = "[FAIL]"
            elif raw >= 0.7:
                tier = "[GOOD]"
            elif raw >= 0.4:
                tier = "[OK]  "
            else:
                tier = "[POOR]"
            
            lines.append(f"  {tier} {name:<20s} raw={raw:.3f}  weighted={weighted:.3f}")
            
            # Add component-specific critique from metadata
            critique = result.metadata.get("critique", "") if result.metadata else ""
            if critique:
                lines.append(f"         {critique}")
            
            if result.error_state:
                lines.append(f"         Error: {result.error_state}")
        
        lines.append("-" * 60)
        
        # Overall verdict
        if total >= 0.7:
            verdict = "[EXCELLENT]"
        elif total >= 0.5:
            verdict = "[GOOD]     "
        elif total >= 0.3:
            verdict = "[MEDIOCRE] "
        else:
            verdict = "[POOR]     "
        
        lines.append(f"  {verdict} TOTAL REWARD: {total:.4f}")
        if error_class:
            lines.append(f"  Error Class: {error_class}")
        lines.append("=" * 60)
        
        return "\n".join(lines)
