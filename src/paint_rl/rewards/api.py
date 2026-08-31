"""Reward API — lazy factory pattern, config-driven, no eager singletons.

This module provides get_rewards() as the main entrypoint for the reward pipeline.
All providers are lazily instantiated on first use and cached.
Weights come from ACTIVE_CONFIG.reward.weights — never hardcoded.
"""
import uuid
import time
from typing import Optional

from .validation import validate_reward_bundle, RewardValidationError
from .components import (
    RewardResult,
    CompileRewardComponent,
    AestheticRewardComponent,
    PairwiseRewardComponent,
)
from .composer import RewardComposer

# Lazy singletons — NOT instantiated at import time
_composer: Optional[RewardComposer] = None
_renderer = None


def _get_config():
    """Safely get config, returning None if not loaded."""
    try:
        from paint_rl.config.core import ACTIVE_CONFIG
        return ACTIVE_CONFIG
    except Exception:
        return None


def _get_renderer():
    """Lazily create renderer client."""
    global _renderer
    if _renderer is not None:
        return _renderer

    config = _get_config()
    if config is None:
        return None

    from paint_rl.renderer.manager import RendererService
    _renderer = RendererService(
        host=config.renderer.host,
        port=config.renderer.port,
        timeout=config.renderer.timeout_ms // 1000,
    )
    return _renderer


def _get_composer() -> RewardComposer:
    """Lazily create the reward composer with config-driven weights."""
    global _composer
    if _composer is not None:
        return _composer

    config = _get_config()

    # Get weights from config
    if config is not None:
        w_compile = config.reward.weights.compile
        w_aesthetic = config.reward.weights.hpsv3  # Named hpsv3 in config for backward compat
        w_pairwise = config.reward.weights.pairwise
    else:
        # Defaults matching base.yaml
        w_compile = 0.10
        w_aesthetic = 0.30
        w_pairwise = 0.60

    # Create components
    compile_component = CompileRewardComponent(weight=w_compile)

    # Aesthetic scorer — lazy, fail-closed
    aesthetic_component = None
    try:
        from .aesthetic import create_aesthetic_scorer
        scorer = create_aesthetic_scorer(config)
        aesthetic_component = AestheticRewardComponent(weight=w_aesthetic, scorer=scorer)
    except Exception as e:
        print(f"[WARN] Aesthetic scorer unavailable: {e}. Aesthetic reward will be 0.")

    # Pairwise judge — lazy, mode-aware
    pairwise_component = None
    try:
        from paint_rl.judges.providers import create_judge_provider
        judge = create_judge_provider(config)
        pairwise_component = PairwiseRewardComponent(weight=w_pairwise, judge_provider=judge)
    except Exception as e:
        print(f"[WARN] Judge provider unavailable: {e}. Pairwise reward will be 0.")

    components = [compile_component]
    if aesthetic_component:
        components.append(aesthetic_component)
    if pairwise_component:
        components.append(pairwise_component)

    _composer = RewardComposer(components=components)
    return _composer


def get_rewards(prompt: str, code: str, reference_path: str, seed: int) -> dict:
    """Main reward entrypoint. Renders code, computes all reward components.

    Returns:
        dict with keys: total, compile, hps, pair, len, error_class, img_path,
        judge_status, components (list of per-component details), reward_version.
    """
    run_id = f"job_{uuid.uuid4().hex[:8]}"
    start_t = time.time()

    # 1. Render the code
    renderer = _get_renderer()
    if renderer is None:
        return {
            "total": 0.0, "compile": 0.0, "hps": 0.0, "pair": 0.0,
            "len": len(code), "error_class": "RENDERER_UNAVAILABLE",
            "img_path": None, "judge_status": "none", "reward_version": "2.0"
        }

    render_result = renderer.render(code, seed=seed, prompt=prompt)

    # 2. Get composer and compute rewards
    composer = _get_composer()
    bundle = composer.compute(
        render_result=render_result,
        image_path=render_result.get("image_path"),
        prompt=prompt,
        reference_path=reference_path,
    )

    # 3. Build backward-compatible output dict
    component_map = {r.component_name: r for r in bundle.get("components", [])}

    output = {
        "total": bundle["total"],
        "compile": component_map.get("compile", RewardResult(0, 0, 0, "compile", 0)).raw_score,
        "hps": component_map.get("aesthetic", RewardResult(0, 0, 0, "aesthetic", 0)).raw_score,
        "pair": component_map.get("pairwise", RewardResult(0, 0, 0, "pairwise", 0)).raw_score,
        "len": len(code),
        "error_class": bundle.get("error_class"),
        "img_path": render_result.get("image_path"),
        "judge_status": component_map.get("pairwise", RewardResult(0, 0, 0, "pairwise", 0, metadata={"judge_status": "none"})).metadata.get("judge_status", "none"),
        "reward_version": "2.0",
        "latency_ms": (time.time() - start_t) * 1000,
        "components": [
            {
                "name": r.component_name,
                "raw_score": r.raw_score,
                "weight": r.weight,
                "weighted_score": r.weighted_score,
                "latency_ms": r.latency_ms,
                "error_state": r.error_state,
            }
            for r in bundle.get("components", [])
        ],
    }

    try:
        return validate_reward_bundle(output)
    except RewardValidationError:
        output["total"] = 0.0
        output["error_class"] = "REWARD_VALIDATION_ERROR"
        return output
