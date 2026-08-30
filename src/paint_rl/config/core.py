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

def load_config(env: str = "local") -> tuple[ProjectConfig, str, str]:
    import sys
    base_path = os.path.join(os.path.dirname(__file__), "configs", "base.yaml")
    with open(base_path, "r") as f:
        merged = yaml.safe_load(f)

    env_path = os.path.join(os.path.dirname(__file__), "configs", f"{env}.yaml")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            env_config = yaml.safe_load(f)
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
