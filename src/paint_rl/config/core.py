import os
import yaml
import json
import hashlib
from pydantic import BaseModel, Field
from typing import Dict, Optional, Any


class ConfigurationError(ValueError):
    """Raised when configuration validation or mode invariants fail."""
    pass


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
    model_config = {"extra": "allow"}

    compile: float = 0.15
    prompt_alignment: Optional[float] = 0.35
    visual_richness: Optional[float] = 0.25
    brush_utilization: Optional[float] = 0.15
    aesthetic: Optional[float] = 0.10
    hpsv3: Optional[float] = 0.30
    pairwise: Optional[float] = 0.60

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
            device_info["mps_fallback"] = (os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "1")
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
        raise ConfigurationError(f"Configuration Error: {e}") from e

    # Hash
    config_json = config.model_dump_json(indent=2)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    
    return config, config_json, config_hash


def apply_max_hardware_config(config):
    """Probe hardware and scale config for maximum throughput.
    
    Called when --max CLI flag is set. Probes available RAM, CPU cores,
    and GPU VRAM to maximize group_size, max_new_tokens, and thread pools.
    
    Args:
        config: ProjectConfig instance (mutated in-place).
        
    Returns:
        Dict of adjustments made for logging.
    """
    import torch
    adjustments = {}
    
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / 1e9
        avail_gb = psutil.virtual_memory().available / 1e9
    except ImportError:
        ram_gb = 8.0
        avail_gb = 4.0
    
    cpu_count = os.cpu_count() or 4
    
    # Set torch threads to use all physical cores
    torch.set_num_threads(cpu_count)
    adjustments["torch_threads"] = cpu_count
    
    device_type = config.device.type if config else "cpu"
    
    if device_type == "mps":
        # Apple Silicon unified memory scaling:
        # 1.5B float32 model alone requires ~6.16GB.
        # Forward/backward tensors for B=2 take ~2GB.
        # Safe memory budget: ~11GB total peak footprint inside 16GB unified RAM.
        if ram_gb >= 64:
            config.training.group_size = 6
            config.training.batch_size = 6
            config.generation.max_new_tokens = 448
        elif ram_gb >= 32:
            config.training.group_size = 4
            config.training.batch_size = 4
            config.generation.max_new_tokens = 384
        elif ram_gb >= 16:
            config.training.group_size = 2
            config.training.batch_size = 2
            config.generation.max_new_tokens = 320
        else:
            config.training.group_size = 2
            config.training.batch_size = 2
            config.generation.max_new_tokens = 256
        adjustments["mps_group_size"] = config.training.group_size
        adjustments["mps_max_new_tokens"] = config.generation.max_new_tokens
        
    elif device_type == "cuda":
        # CUDA: scale by VRAM
        try:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        except Exception:
            vram_gb = 8.0
        
        if vram_gb >= 40:  # A100
            config.training.group_size = 8
            config.training.batch_size = 8
            config.generation.max_new_tokens = 512
        elif vram_gb >= 15:  # T4/V100
            config.training.group_size = 6
            config.training.batch_size = 6
            config.generation.max_new_tokens = 450
        else:
            config.training.group_size = 4
            config.training.batch_size = 4
            config.generation.max_new_tokens = 384
        adjustments["cuda_vram_gb"] = round(vram_gb, 1)
        adjustments["cuda_group_size"] = config.training.group_size
        
    else:  # CPU
        # Maximize threads, keep small group_size (CPU generation is slow)
        config.training.group_size = 2
        config.training.batch_size = 2
        config.generation.max_new_tokens = min(256, config.generation.max_new_tokens)
        adjustments["cpu_cores"] = cpu_count
    
    adjustments["ram_total_gb"] = round(ram_gb, 1)
    adjustments["ram_available_gb"] = round(avail_gb, 1)
    
    return adjustments


# Lazy singleton initialization
ACTIVE_CONFIG, CONFIG_JSON, CONFIG_HASH = None, None, None
try:
    ACTIVE_CONFIG, CONFIG_JSON, CONFIG_HASH = load_config(os.environ.get("ENV", "local"))
except Exception as e:
    pass
