import json
import torch
import psutil
import platform

class CapabilityEvaluator:
    @staticmethod
    def estimate_memory_gb(params_b: float, precision: str, has_optimizer: bool = False, has_kv: bool = False) -> float:
        """
        THEORETICAL ESTIMATE ONLY.
        This provides a rough memory footprint calculation including gradients, 
        optimizer implementation, master weights, activations, KV cache, and framework overhead.
        """
        bytes_per_param = 2.0 if precision in ("float16", "bfloat16") else 1.0 
        
        # Base weights
        mem = params_b * bytes_per_param
        
        if has_optimizer:
            # Gradients
            mem += params_b * bytes_per_param
            # Optimizer state (e.g., AdamW needs 2x FP32 master states = 8 bytes/param)
            mem += params_b * 8.0
            # Activations (highly variable, rough estimate for Phase-0 batch sizes)
            mem += 2.0 
            
        if has_kv:
            # KV cache for inference/rollout
            mem += 1.5 
            
        # OS / Framework / Runtime overhead
        mem += 1.5
            
        return mem

class ModelRegistry:
    """Model selection based on hardware capabilities and execution mode.
    
    Selects policy model, judge model, and device placement based on
    detected hardware, execution mode, and memory constraints.
    """

    # Actual local VLM judge model — lightweight, fits on 16GB Mac and T4
    LOCAL_JUDGE_MODEL = "Qwen/Qwen2-VL-2B-Instruct"

    @classmethod
    def select_models(cls, mode: str, caps: dict, allow_paid_api: bool = False):
        cuda_avail = caps.get("cuda_available", False)
        mps_avail = caps.get("mps_available", False)
        ram_gb = caps.get("ram_gb", 0)
        gpu_count = caps.get("gpu_count", 0)
        
        vram_list = caps.get("vram_gb", [])
        total_vram_gb = sum(vram_list) if vram_list else 0
        
        # In Free/Local, Paid API is strictly forbidden unless explicitly overridden
        if mode in ("FREE", "LOCAL") and not allow_paid_api:
            judge_model = cls.LOCAL_JUDGE_MODEL
            judge_device = "mps" if mps_avail else "cuda:1" if gpu_count > 1 else "cpu"
        else:
            judge_model = "openai"
            judge_device = "cpu"

        if mps_avail and mode == "LOCAL":
            # Memory budget: unified RAM minus ~4GB for macOS
            usable_ram = max(0, ram_gb - 4.0)
            
            # 1.5B policy (FP32 training): ~6.8GB
            # 2B VLM judge (FP16 sequential): ~4.5GB (loaded/unloaded between phases)
            policy_1_5b_train_mem = 6.8
            
            if usable_ram >= policy_1_5b_train_mem:
                return {
                    "policy_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    "policy_device": "mps",
                    "judge_model": judge_model,
                    "judge_device": "mps" if judge_model != "openai" else "cpu"
                }
            else:
                # Extremely constrained: fall back to 0.5B
                return {
                    "policy_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    "policy_device": "mps",
                    "judge_model": judge_model,
                    "judge_device": "mps" if judge_model != "openai" else "cpu"
                }

        # CUDA Local or Free Backend
        if cuda_avail:
            if mode == "FREE" and gpu_count == 2 and total_vram_gb >= 30.0:
                return {
                    "policy_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "policy_quant": "int4",
                    "policy_device": "cuda:0",
                    "judge_model": cls.LOCAL_JUDGE_MODEL,
                    "judge_quant": "int4",
                    "judge_device": "cuda:1"
                }
            elif mode == "LOCAL" and total_vram_gb <= 12.0:
                return {
                    "policy_model": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
                    "policy_quant": "int4",
                    "policy_device": "cuda:0",
                    "judge_model": "openai" if allow_paid_api else judge_model, 
                    "judge_quant": "none",
                    "judge_device": "cpu" if allow_paid_api else "cuda:0"
                }
            else:
                return {
                    "policy_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
                    "policy_quant": "bfloat16",
                    "policy_device": "cuda:0",
                    "judge_model": "openai" if allow_paid_api else judge_model,
                    "judge_quant": "none",
                    "judge_device": "cpu" if allow_paid_api else "cuda:0"
                }

        # CPU Fallback
        return {
            "policy_model": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
            "policy_device": "cpu",
            "judge_model": judge_model,
            "judge_device": "cpu"
        }
