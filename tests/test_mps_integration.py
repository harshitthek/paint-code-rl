"""Tests for MPS/device integration — config resolution, model selection, memory safety.

These tests verify that the MPS config overlay is applied correctly and that
the trainer selects memory-safe parameters for Apple Silicon.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Set MPS fallback for test environment
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


class TestDeviceDetection:
    """Verify device detection returns valid device info."""

    def test_detect_returns_dict(self):
        from paint_rl.config.core import detect_compute_device
        info = detect_compute_device()
        assert isinstance(info, dict)
        assert "type" in info
        assert "precision" in info
        assert "mps_fallback" in info
        assert info["type"] in ("cuda", "mps", "cpu")

    def test_detect_precision_matches_device(self):
        from paint_rl.config.core import detect_compute_device
        info = detect_compute_device()
        if info["type"] == "mps":
            # MPS should use float32 for training stability
            assert info["precision"] == "float32"
        elif info["type"] == "cuda":
            assert info["precision"] in ("float16", "bfloat16")
        else:
            assert info["precision"] == "float32"


class TestMPSConfigOverlay:
    """Verify MPS config overlay is applied when device is MPS."""

    def test_mps_config_loads(self):
        """MPS provider config file should be valid YAML."""
        import yaml
        mps_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'providers', 'mps.yaml')
        assert os.path.exists(mps_path), f"MPS config not found at {mps_path}"
        with open(mps_path) as f:
            config = yaml.safe_load(f)
        assert config is not None
        assert "model" in config
        assert "training" in config
        assert config["model"]["id"] == "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        assert config["training"]["batch_size"] <= 4

    def test_mps_config_batch_size_safe(self):
        """MPS config must have memory-safe batch sizes."""
        import yaml
        mps_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'providers', 'mps.yaml')
        with open(mps_path) as f:
            config = yaml.safe_load(f)
        assert config["training"]["batch_size"] <= 4, \
            f"MPS batch_size={config['training']['batch_size']} exceeds safe limit for 16GB"
        assert config["training"]["group_size"] <= 4, \
            f"MPS group_size={config['training']['group_size']} exceeds safe limit"

    def test_mps_config_model_is_small(self):
        """MPS config must not use 7B model."""
        import yaml
        mps_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'providers', 'mps.yaml')
        with open(mps_path) as f:
            config = yaml.safe_load(f)
        model_id = config["model"]["id"]
        assert "7B" not in model_id, f"MPS config uses 7B model: {model_id}"
        assert "1.5B" in model_id or "0.5B" in model_id


class TestDeviceConfig:
    """Verify DeviceConfig is part of ProjectConfig schema."""

    def test_device_in_project_config(self):
        from paint_rl.config.core import load_config
        config, _, _ = load_config("local")
        assert hasattr(config, "device")
        assert config.device.type in ("cuda", "mps", "cpu")

    def test_device_config_defaults(self):
        from paint_rl.config.core import DeviceConfig
        dc = DeviceConfig()
        assert dc.type == "cpu"
        assert dc.precision == "float32"
        assert dc.mps_fallback is False


class TestTrainerDeviceSelection:
    """Verify PaintGRPOTrainer selects correct model per device."""

    def test_select_model_id_returns_string(self):
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        model_id = trainer.select_model_id()
        assert isinstance(model_id, str)
        assert "Qwen" in model_id

    def test_resolve_model_overrides_config_on_constrained_device(self):
        """On MPS/CPU, trainer must override config model to memory-safe variant."""
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        
        if trainer.device.type in ("mps", "cpu"):
            resolved = trainer._resolve_model_id()
            assert "7B" not in resolved, \
                f"Trainer resolved to {resolved} on {trainer.device.type} — should be ≤1.5B"

    def test_safe_batch_params_bounded(self):
        """Batch params must be bounded on constrained devices."""
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        batch_size, group_size = trainer._get_safe_batch_params()
        
        if trainer.device.type == "mps":
            assert batch_size <= 2, f"MPS batch_size={batch_size}, expected <=2"
            assert group_size <= 2, f"MPS group_size={group_size}, expected <=2"
        elif trainer.device.type == "cpu":
            assert batch_size <= 2
            assert group_size <= 2

    def test_dtype_safe_for_device(self):
        """MPS must use float32, CUDA can use float16/bfloat16."""
        import torch
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        dtype = trainer._get_dtype()
        
        if trainer.device.type == "mps":
            assert dtype == torch.float32
        elif trainer.device.type == "cpu":
            assert dtype == torch.float32


class TestModelRegistryMPS:
    """Verify ModelRegistry uses correct judge model IDs."""

    def test_mps_local_judge_model(self):
        from paint_rl.models.registry import ModelRegistry
        caps = {"cuda_available": False, "mps_available": True, "ram_gb": 16.0, "gpu_count": 0, "vram_gb": []}
        sel = ModelRegistry.select_models("LOCAL", caps, allow_paid_api=False)
        assert sel["policy_device"] == "mps"
        # Must use actual existing model, not the nonexistent Qwen2.5-VL-7B
        assert "Qwen2-VL-2B" in sel["judge_model"], \
            f"Judge model should be Qwen2-VL-2B-Instruct, got: {sel['judge_model']}"

    def test_mps_selects_small_policy(self):
        from paint_rl.models.registry import ModelRegistry
        caps = {"cuda_available": False, "mps_available": True, "ram_gb": 16.0, "gpu_count": 0, "vram_gb": []}
        sel = ModelRegistry.select_models("LOCAL", caps, allow_paid_api=False)
        assert "7B" not in sel["policy_model"], \
            f"MPS 16GB should not select 7B model: {sel['policy_model']}"

    def test_cpu_fallback_uses_smallest_model(self):
        from paint_rl.models.registry import ModelRegistry
        caps = {"cuda_available": False, "mps_available": False, "ram_gb": 8.0, "gpu_count": 0, "vram_gb": []}
        sel = ModelRegistry.select_models("LOCAL", caps, allow_paid_api=False)
        assert "0.5B" in sel["policy_model"]
        assert sel["policy_device"] == "cpu"
