import math
from typing import Any, Dict

class RewardValidationError(ValueError):
    pass

def validate_reward_score(score: Any, component_name: str, min_val: float = None, max_val: float = None) -> float:
    if score is None:
        raise RewardValidationError(f"Reward component '{component_name}' returned None.")
    if not isinstance(score, (int, float)):
        raise RewardValidationError(f"Reward component '{component_name}' returned non-numeric type: {type(score)}")
    if math.isnan(score):
        raise RewardValidationError(f"Reward component '{component_name}' returned NaN.")
    if math.isinf(score):
        raise RewardValidationError(f"Reward component '{component_name}' returned Infinity.")
        
    score_float = float(score)
    if min_val is not None and score_float < min_val:
        raise RewardValidationError(f"Reward component '{component_name}' ({score_float}) below minimum ({min_val}).")
    if max_val is not None and score_float > max_val:
        raise RewardValidationError(f"Reward component '{component_name}' ({score_float}) above maximum ({max_val}).")
        
    return score_float

def validate_reward_bundle(bundle: Dict[str, Any]) -> Dict[str, Any]:
    # Ensure critical keys exist and are valid
    if "total" not in bundle:
        raise RewardValidationError("Reward bundle missing 'total' key.")
        
    bundle["total"] = validate_reward_score(bundle["total"], "total")
    return bundle
