import os
import yaml
import json
import hashlib
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any

class RunConfig(BaseModel):
    experiment_name: str
    seed: int = 42
    log_level: str = "INFO"

class ModelConfig(BaseModel):
    id: str
    revision: str
    device_map: str = "auto"

class TrainingConfig(BaseModel):
    batch_size: int
    group_size: int
    max_steps: int
    checkpoint_freq: int

class GenerationConfig(BaseModel):
    max_new_tokens: int
    temperature: float
    top_p: float

class RendererConfig(BaseModel):
    host: str
    port: int
    timeout_ms: int
    max_inflight_renders: int

class RewardWeights(BaseModel):
    compile: float
    hpsv3: float
    pairwise: float

class RewardConfig(BaseModel):
    weights: RewardWeights

class JudgeConfig(BaseModel):
    provider: str
    model: str

class StorageConfig(BaseModel):
    type: str
    base_path: str
    datasets_path: str

class ProjectConfig(BaseModel):
    run: RunConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    renderer: RendererConfig
    reward: RewardConfig
    judge: JudgeConfig
    storage: StorageConfig

def deep_merge(dict1: dict, dict2: dict) -> dict:
    for k, v in dict2.items():
        if isinstance(v, dict) and k in dict1 and isinstance(dict1[k], dict):
            dict1[k] = deep_merge(dict1[k], v)
        else:
            dict1[k] = v
    return dict1

def find_configs_dir():
    if "CONFIG_ROOT" in os.environ and os.path.isdir(os.environ["CONFIG_ROOT"]):
        return os.path.abspath(os.environ["CONFIG_ROOT"])
    curr = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        cand = os.path.join(curr, "configs")
        if os.path.isdir(cand) and os.path.exists(os.path.join(cand, "base.yaml")):
            return cand
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    if os.path.isdir("configs") and os.path.exists(os.path.join("configs", "base.yaml")):
        return os.path.abspath("configs")
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "configs"))

def load_config(env: str = "local") -> tuple[ProjectConfig, str, str]:
    import sys
    cfg_dir = find_configs_dir()
    base_path = os.path.join(cfg_dir, "base.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Configuration base file not found at {base_path}")
        
    with open(base_path, "r") as f:
        merged = yaml.safe_load(f)

    env_path = os.path.join(cfg_dir, f"{env}.yaml")
    if not os.path.exists(env_path):
        # Also check configs/modes/{env}.yaml or configs/providers/{env}.yaml
        cand_mode = os.path.join(cfg_dir, "modes", f"{env}.yaml")
        cand_prov = os.path.join(cfg_dir, "providers", f"{env}.yaml")
        if os.path.exists(cand_mode):
            env_path = cand_mode
        elif os.path.exists(cand_prov):
            env_path = cand_prov

    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_config = yaml.safe_load(f)
            if env_config:
                merged = deep_merge(merged, env_config)

    # Validate
    try:
        config = ProjectConfig(**merged)
    except Exception as e:
        print(f"Configuration Error: {e}")
        sys.exit(1)

    # Hash
    config_json = config.model_dump_json(indent=2)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    
    return config, config_json, config_hash

# Singleton pattern for the loaded config
ACTIVE_CONFIG, CONFIG_JSON, CONFIG_HASH = load_config(os.environ.get("ENV", "local"))
