"""Centralized, dynamic hardware & environment detection and optimization engine.

Auto-detects runtime environments (Kaggle, Colab, Apple Silicon, Linux/Windows CUDA, CPU),
probes available compute/VRAM/RAM resources, and dynamically configures:
- Safe batch sizes and group sizes
- Gradient accumulation steps
- Precision (bf16, fp16, fp32)
- Memory ceilings and gradient checkpointing
- Headless WebGL Chromium flags per OS platform
"""
import os
import sys
import platform
import psutil
import torch
from dataclasses import dataclass, field
from typing import Dict, Any, List


@dataclass
class HardwareProfile:
    env_name: str                  # "kaggle", "colab", "macos_apple_silicon", "cuda_linux", "cuda_windows", "cpu"
    device: torch.device
    device_name: str
    vram_gb: float
    ram_total_gb: float
    ram_available_gb: float
    cpu_cores: int
    torch_dtype: torch.dtype
    policy_model: str
    train_batch_size: int
    group_size: int
    gradient_accumulation_steps: int
    max_new_tokens: int
    gradient_checkpointing: bool
    attn_implementation: str
    renderer_args: List[str] = field(default_factory=list)


class AutoHardwareOptimizer:
    """Dynamically adapts training, rollout, and rendering settings to any host environment."""

    @staticmethod
    def is_kaggle() -> bool:
        return os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    @staticmethod
    def is_colab() -> bool:
        return "COLAB_GPU" in os.environ or os.path.exists("/content")

    @staticmethod
    def is_apple_silicon() -> bool:
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    @classmethod
    def get_profile(cls) -> HardwareProfile:
        ram = psutil.virtual_memory()
        ram_total_gb = round(ram.total / 1e9, 2)
        ram_available_gb = round(ram.available / 1e9, 2)
        cpu_cores = psutil.cpu_count(logical=True) or 4

        # 1. CUDA Acceleration (Kaggle GPU, Colab, Local CUDA)
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
            device = torch.device("cuda:0")

            if cls.is_kaggle():
                env_name = "kaggle"
            elif cls.is_colab():
                env_name = "colab"
            elif platform.system() == "Windows":
                env_name = "cuda_windows"
            else:
                env_name = "cuda_linux"

            # Precision selection: bfloat16 for Ampere/Ada/Hopper, float16 for Volta/Turing/Tesla T4
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16

            # Model & batch scaling based on VRAM
            if vram_gb >= 22.0:
                policy_model = "Qwen/Qwen2.5-Coder-7B-Instruct"
                batch_size = 8
                group_size = 4
                max_tokens = 550
            elif vram_gb >= 14.0:
                policy_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
                batch_size = 6
                group_size = 4
                max_tokens = 512
            else:
                policy_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
                batch_size = 4
                group_size = 2
                max_tokens = 450

            grad_accum = max(1, batch_size // group_size)
            renderer_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--enable-webgl",
                "--use-gl=angle",
                "--use-angle=swiftshader-webgl"
            ]

            return HardwareProfile(
                env_name=env_name,
                device=device,
                device_name=gpu_name,
                vram_gb=vram_gb,
                ram_total_gb=ram_total_gb,
                ram_available_gb=ram_available_gb,
                cpu_cores=cpu_cores,
                torch_dtype=torch_dtype,
                policy_model=policy_model,
                train_batch_size=batch_size,
                group_size=group_size,
                gradient_accumulation_steps=grad_accum,
                max_new_tokens=max_tokens,
                gradient_checkpointing=True,
                attn_implementation="sdpa",
                renderer_args=renderer_args,
            )

        # 2. Apple Silicon MPS Acceleration (MacBook Air / Pro M1/M2/M3/M4)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
            device_name = f"Apple Silicon ({platform.processor() or 'M-Series'})"
            env_name = "macos_apple_silicon"

            # Safe parameters for unified memory
            policy_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
            batch_size = 4
            group_size = 2
            grad_accum = 2
            max_tokens = 480
            torch_dtype = torch.float32

            renderer_args = [
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
                "--enable-webgl",
                "--use-gl=angle",
                "--use-angle=metal",
                "--disable-gpu-sandbox"
            ]

            return HardwareProfile(
                env_name=env_name,
                device=device,
                device_name=device_name,
                vram_gb=0.0,
                ram_total_gb=ram_total_gb,
                ram_available_gb=ram_available_gb,
                cpu_cores=cpu_cores,
                torch_dtype=torch_dtype,
                policy_model=policy_model,
                train_batch_size=batch_size,
                group_size=group_size,
                gradient_accumulation_steps=grad_accum,
                max_new_tokens=max_tokens,
                gradient_checkpointing=True,
                attn_implementation="sdpa",
                renderer_args=renderer_args,
            )

        # 3. CPU Fallback (Graceful minimal execution)
        device = torch.device("cpu")
        device_name = f"CPU ({platform.machine()})"
        env_name = "cpu"

        policy_model = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
        batch_size = 2
        group_size = 2
        grad_accum = 1
        max_tokens = 320
        torch_dtype = torch.float32

        renderer_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--enable-webgl",
            "--use-gl=angle",
            "--use-angle=swiftshader-webgl"
        ]

        return HardwareProfile(
            env_name=env_name,
            device=device,
            device_name=device_name,
            vram_gb=0.0,
            ram_total_gb=ram_total_gb,
            ram_available_gb=ram_available_gb,
            cpu_cores=cpu_cores,
            torch_dtype=torch_dtype,
            policy_model=policy_model,
            train_batch_size=batch_size,
            group_size=group_size,
            gradient_accumulation_steps=grad_accum,
            max_new_tokens=max_tokens,
            gradient_checkpointing=False,
            attn_implementation="sdpa",
            renderer_args=renderer_args,
        )

    @classmethod
    def apply_runtime_optimizations(cls) -> HardwareProfile:
        """Configures environment flags, threads, and memory allocators."""
        profile = cls.get_profile()

        # Optimize thread pool (accelerators need fewer CPU threads, CPU-only training needs more)
        optimal_threads = min(profile.cpu_cores, 4 if profile.device.type != "cpu" else 10)
        torch.set_num_threads(optimal_threads)

        # MPS allocator flags
        if profile.device.type == "mps":
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.8")

        # Disable tokenizer parallelism warnings
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        return profile
