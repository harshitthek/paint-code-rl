import os
import yaml
import json
import hashlib
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class SafetyConfig(BaseModel):
    allow_external_apis: bool = False

class AestheticConfig(BaseModel):
    provider: str = "clip"
    model: str = "openai/clip-vit-large-patch14"
    device: str = "auto"

class DeviceConfig(BaseModel):
    """Auto-detected compute device configuration."""
    type: str = "cpu"          # "cuda", "mps", "cpu"
    precision: str = "float32" # "float16", "bfloat16", "float32"
    mps_fallback: bool = False # Whether PYTORCH_ENABLE_MPS_FALLBACK is set

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
    repetition_penalty: float = 1.12
    no_repeat_ngram_size: int = 6

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
    model_config = {"extra": "ignore"}

    run: RunConfig
    model: ModelConfig
    training: TrainingConfig
    generation: GenerationConfig
    renderer: RendererConfig
    reward: RewardConfig
    judge: JudgeConfig
    storage: StorageConfig
    safety: SafetyConfig = SafetyConfig()
    aesthetic: AestheticConfig = AestheticConfig()
    device: DeviceConfig = DeviceConfig()

def deep_merge(dict1: dict, dict2: dict) -> dict:
    """Recursively merge dict2 into dict1. dict2 values take precedence."""
    result = dict1.copy()
    for k, v in dict2.items():
        if isinstance(v, dict) and k in result and isinstance(result[k], dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def find_configs_dir():
    """Locate the configs/ directory by walking up from this file or using env var."""
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


def detect_compute_device() -> dict:
    """Detect available compute device and return device config dict.
    
    Returns dict suitable for merging into config with device type,
    precision, and MPS fallback status.
    """
    device_info = {"type": "cpu", "precision": "float32", "mps_fallback": False}
    try:
        import torch
        if torch.cuda.is_available():
            device_info["type"] = "cuda"
            device_info["precision"] = "float16"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_info["type"] = "mps"
            # Use float32 for MPS training stability (FP16 causes grad instabilities
            # in GRPO backward pass on MPS)
            device_info["precision"] = "float32"
            device_info["mps_fallback"] = True
    except ImportError:
        pass
    return device_info


def _apply_mps_env():
    """Set MPS fallback environment variable for unsupported ops."""
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def load_config(env: str = "local") -> tuple:
    """Load and merge configuration from YAML files.
    
    Resolution order:
    1. base.yaml (always loaded)
    2. Environment overlay ({env}.yaml or modes/{env}.yaml or providers/{env}.yaml)
    3. Device info injected into config
    
    Returns:
        Tuple of (ProjectConfig, config_json_str, config_hash_hex)
    """
    import sys
    cfg_dir = find_configs_dir()
    base_path = os.path.join(cfg_dir, "base.yaml")
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Configuration base file not found at {base_path}")
        
    with open(base_path, "r") as f:
        merged = yaml.safe_load(f)

    # Load environment overlay
    env_path = os.path.join(cfg_dir, f"{env}.yaml")
    if not os.path.exists(env_path):
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

    # Auto-detect device and inject device info
    device_info = detect_compute_device()
    if device_info["type"] == "mps":
        _apply_mps_env()

    # Inject device info
    merged["device"] = device_info

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

# Lazy singleton initialization
ACTIVE_CONFIG, CONFIG_JSON, CONFIG_HASH = None, None, None
try:
    ACTIVE_CONFIG, CONFIG_JSON, CONFIG_HASH = load_config(os.environ.get("ENV", "local"))
except Exception as e:
    pass
