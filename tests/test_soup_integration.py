import pytest
import math
from paint_rl.models.registry import ModelRegistry, CapabilityEvaluator
from paint_rl.rewards.validation import validate_reward_bundle, RewardValidationError
from paint_rl.trainer.async_rollout import RolloutEngine
from paint_rl.trainer.checkpoint_validator import CheckpointValidator

def test_preflight():
    caps = {"cuda_available": True, "gpu_count": 2, "vram_gb": [16.0, 16.0]}
    sel = ModelRegistry.select_models("FREE", caps, allow_paid_api=False)
    assert sel["policy_device"] == "cuda:0"
    
def test_reward_validation():
    # Valid
    validate_reward_bundle({"total": 1.5})
    
    # NaN
    with pytest.raises(RewardValidationError, match="returned NaN"):
        validate_reward_bundle({"total": math.nan})

def test_async_rollout():
    engine = RolloutEngine(max_workers=2)
    def dummy_reward(p, c, r, s):
        if s == 0:
            raise ValueError("Intentional crash")
        return {"total": 1.0}
        
    results = engine.generate_and_evaluate(
        prompts=["a", "b"], codes=["c", "d"], reference_paths=["x", "y"], seeds=[0, 1],
        reward_fn=dummy_reward
    )
    
    assert results[0]["error_class"] == "WORKER_CRASH"
    assert results[1]["total"] == 1.0
