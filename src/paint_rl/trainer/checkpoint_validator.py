import os
import json
from safetensors import safe_open
from paint_rl.config.core import CONFIG_HASH

class CheckpointValidator:
    @classmethod
    def validate_safetensors(cls, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint {filepath} not found.")
            
        try:
            with safe_open(filepath, framework="pt") as f:
                keys = f.keys()
                # 1. Soup Bug Defense: Ensure no '.inner.' corruption in keys
                for key in keys:
                    if ".inner." in key:
                        raise ValueError(f"CRITICAL: Found corrupted adapter key '{key}' (contains '.inner.'). "
                                         "This indicates an inert adapter save.")
                
                # Ensure we have adapter keys (assuming LoRA)
                lora_keys = [k for k in keys if "lora_" in k]
                if not lora_keys:
                    raise ValueError("CRITICAL: No 'lora_' keys found in checkpoint. Adapter is empty.")
                    
        except Exception as e:
            raise ValueError(f"Safetensors validation failed for {filepath}: {str(e)}")

    @classmethod
    def validate_experiment_state(cls, state_path: str):
        if not os.path.exists(state_path):
            raise FileNotFoundError("Experiment state not found.")
            
        with open(state_path, "r") as f:
            state = json.load(f)
            
        if state.get("config_hash") != CONFIG_HASH:
            raise ValueError(f"CRITICAL: Checkpoint config hash mismatch! "
                             f"Saved: {state.get('config_hash')} != Current: {CONFIG_HASH}")
                             
        return state
