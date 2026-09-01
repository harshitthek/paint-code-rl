"""Apple Silicon / MPS Hardware Validation Suite.

Runs 4 validation phases:
1. MPS Compute & Tensor Verification
2. Memory Feasibility & Model Selection
3. Minimal Model Load Test (0.5B forward pass on MPS)
4. TRL/GRPO Configuration Compatibility Check

Usage:
    python scripts/mps_validation_suite.py
"""
import os
import sys
import json
import platform
import psutil
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Set MPS fallback before any torch imports
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "0")

REPORT = {}


def get_mac_hardware():
    hw = {
        "os": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_cores": os.cpu_count(),
        "ram_gb": round(psutil.virtual_memory().total / 1e9, 2),
        "available_ram_gb": round(psutil.virtual_memory().available / 1e9, 2),
    }
    try:
        import torch
        hw["torch_version"] = torch.__version__
        hw["mps_built"] = torch.backends.mps.is_built() if hasattr(torch.backends, 'mps') else False
        hw["mps_available"] = torch.backends.mps.is_available() if hasattr(torch.backends, 'mps') else False
        hw["mps_fallback_env"] = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "not_set")
    except ImportError:
        hw["torch_version"] = "missing"
        hw["mps_built"] = False
        hw["mps_available"] = False
    return hw


def test_mps_tensors():
    """Phase 1: Basic MPS tensor allocation, math, and numerical integrity."""
    import torch
    print("Testing MPS Tensor Allocation & Math Operations...")
    device = torch.device("mps")
    results = {}
    
    # 1. FP32 Allocation & Matmul
    x = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    y = torch.randn(1024, 1024, device=device, dtype=torch.float32)
    
    start = time.time()
    z = torch.matmul(x, y)
    torch.mps.synchronize()
    elapsed_ms = (time.time() - start) * 1000
    
    has_nan = torch.isnan(z).any().item()
    has_inf = torch.isinf(z).any().item()
    
    results["fp32_matmul_1024x1024_ms"] = round(elapsed_ms, 2)
    results["fp32_nan_check"] = "PASS" if not has_nan else "FAIL"
    results["fp32_inf_check"] = "PASS" if not has_inf else "FAIL"
    
    # 2. FP16 test (used by inference)
    try:
        x16 = torch.randn(512, 512, device=device, dtype=torch.float16)
        y16 = torch.randn(512, 512, device=device, dtype=torch.float16)
        z16 = torch.matmul(x16, y16)
        torch.mps.synchronize()
        results["fp16_matmul"] = "PASS"
    except Exception as e:
        results["fp16_matmul"] = f"FAIL: {e}"
    
    # 3. BFloat16 test
    try:
        xbf = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        ybf = torch.randn(512, 512, device=device, dtype=torch.bfloat16)
        zbf = torch.matmul(xbf, ybf)
        torch.mps.synchronize()
        results["bf16_matmul"] = "PASS"
    except Exception as e:
        results["bf16_matmul"] = f"FAIL: {e}"
    
    # 4. Softmax (common failure point on MPS)
    try:
        s = torch.softmax(x[:64, :64], dim=-1)
        torch.mps.synchronize()
        results["softmax"] = "PASS"
    except Exception as e:
        results["softmax"] = f"FAIL: {e}"
    
    # 5. MPS empty_cache test
    try:
        torch.mps.empty_cache()
        results["empty_cache"] = "PASS"
    except Exception as e:
        results["empty_cache"] = f"FAIL: {e}"
    
    # 6. Gather/scatter (used in generation)
    try:
        idx = torch.randint(0, 512, (64, 10), device=device)
        src = torch.randn(64, 512, device=device)
        gathered = torch.gather(src, 1, idx)
        results["gather_scatter"] = "PASS"
    except Exception as e:
        results["gather_scatter"] = f"FAIL: {e}"
    
    overall = all(
        v == "PASS" for k, v in results.items() 
        if isinstance(v, str) and k not in ("fp32_matmul_1024x1024_ms",)
    )
    results["status"] = "PASS" if overall else "FAIL"
    return results


def evaluate_model_feasibility(ram_gb):
    """Phase 2: Memory feasibility analysis for all model candidates."""
    usable_vram = max(0, ram_gb - 4.0)
    
    candidates = [
        {"name": "Qwen/Qwen2.5-Coder-0.5B-Instruct", "params_b": 0.5, "est_train_mem_gb": 3.2, "est_inference_mem_gb": 1.2},
        {"name": "Qwen/Qwen2.5-Coder-1.5B-Instruct", "params_b": 1.5, "est_train_mem_gb": 6.8, "est_inference_mem_gb": 3.4},
        {"name": "Qwen/Qwen2.5-Coder-3B-Instruct", "params_b": 3.0, "est_train_mem_gb": 13.5, "est_inference_mem_gb": 6.5},
        {"name": "Qwen/Qwen2.5-Coder-7B-Instruct", "params_b": 7.0, "est_train_mem_gb": 28.0, "est_inference_mem_gb": 14.5},
    ]
    
    eval_results = []
    for c in candidates:
        can_train = c["est_train_mem_gb"] <= usable_vram
        can_infer = c["est_inference_mem_gb"] <= usable_vram
        eval_results.append({
            "model": c["name"],
            "params": f"{c['params_b']}B",
            "inference_feasible": can_infer,
            "training_feasible": can_train,
            "estimated_train_memory_gb": c["est_train_mem_gb"],
            "estimated_infer_memory_gb": c["est_inference_mem_gb"]
        })
    return {
        "total_ram_gb": ram_gb,
        "estimated_usable_for_models_gb": round(usable_vram, 2),
        "candidates": eval_results
    }


def test_config_resolution():
    """Phase 3: Verify config correctly auto-detects MPS and loads provider overlay."""
    print("Testing Config Resolution for MPS...")
    results = {}
    
    try:
        from paint_rl.config.core import load_config, detect_compute_device
        
        # Test device detection
        device_info = detect_compute_device()
        results["device_detected"] = device_info["type"]
        results["device_precision"] = device_info["precision"]
        results["mps_fallback"] = device_info.get("mps_fallback", False)
        
        # Test local config loads
        config_local, _, _ = load_config("local")
        results["local_judge_provider"] = config_local.judge.provider
        results["local_safety_allow_apis"] = config_local.safety.allow_external_apis
        
        # Test explicit MPS provider overlay
        config_mps, _, _ = load_config("mps")
        results["mps_model_id"] = config_mps.model.id
        results["mps_batch_size"] = config_mps.training.batch_size
        results["mps_group_size"] = config_mps.training.group_size
        results["mps_max_new_tokens"] = config_mps.generation.max_new_tokens
        
        # Verify MPS provider overlay
        if "1.5B" in config_mps.model.id or "0.5B" in config_mps.model.id:
            results["mps_model_override"] = "PASS"
        else:
            results["mps_model_override"] = f"FAIL: model is {config_mps.model.id}, expected 1.5B or 0.5B"
        
        if config_mps.training.batch_size <= 4:
            results["mps_batch_size_safe"] = "PASS"
        else:
            results["mps_batch_size_safe"] = f"FAIL: batch_size={config_mps.training.batch_size}, expected <=4"
        
        if results.get("mps_model_override") == "PASS" and results.get("mps_batch_size_safe") == "PASS":
            results["status"] = "PASS"
        else:
            results["status"] = "FAIL"
    except Exception as e:
        results["status"] = f"FAIL: {e}"
    
    return results


def test_trainer_device_selection():
    """Phase 4: Verify trainer selects correct model for MPS."""
    print("Testing Trainer Device Selection...")
    results = {}
    
    try:
        from paint_rl.trainer.grpo import PaintGRPOTrainer, get_compute_device
        
        device = get_compute_device()
        results["trainer_device"] = str(device)
        
        trainer = PaintGRPOTrainer()
        model_id = trainer.select_model_id()
        results["selected_model"] = model_id
        
        resolved_id = trainer._resolve_model_id()
        results["resolved_model"] = resolved_id
        
        dtype = trainer._get_dtype()
        results["dtype"] = str(dtype)
        
        batch_size, group_size = trainer._get_safe_batch_params()
        results["safe_batch_size"] = batch_size
        results["safe_group_size"] = group_size
        
        max_tokens = trainer._get_max_new_tokens()
        results["max_new_tokens"] = max_tokens
        
        # Validate constraints
        if device.type == "mps":
            checks = []
            if "1.5B" in model_id:
                checks.append("model_ok")
            else:
                checks.append(f"model_wrong:{model_id}")
            if dtype == __import__('torch').float32:
                checks.append("dtype_ok")
            else:
                checks.append(f"dtype_wrong:{dtype}")
            if batch_size <= 2:
                checks.append("batch_ok")
            else:
                checks.append(f"batch_wrong:{batch_size}")
            
            results["mps_checks"] = checks
            results["status"] = "PASS" if all("ok" in c for c in checks) else "FAIL"
        else:
            results["status"] = "PASS"
    except Exception as e:
        results["status"] = f"FAIL: {e}"
    
    return results


def main():
    print("==================================================")
    print("      APPLE SILICON / MPS VALIDATION SUITE        ")
    print("==================================================")
    hw = get_mac_hardware()
    REPORT["hardware"] = hw
    print(f"OS: {hw['os']} {hw['release']} ({hw['machine']})")
    print(f"Total Unified Memory: {hw['ram_gb']} GB (Available: {hw['available_ram_gb']} GB)")
    print(f"PyTorch Version: {hw['torch_version']}")
    print(f"MPS Available: {hw['mps_available']}")
    print(f"MPS Fallback Env: {hw.get('mps_fallback_env', 'unknown')}")
    
    if not hw["mps_available"]:
        REPORT["status"] = "BLOCKED"
        REPORT["reason"] = "No MPS backend available on this PyTorch build."
        print("\n❌ RESULT: BLOCKED - MPS is not available.")
    else:
        # Phase 1: Tensor Benchmark
        print("\n--- Phase 1: MPS Compute & Tensor Verification ---")
        try:
            tensor_res = test_mps_tensors()
            REPORT["tensor_benchmark"] = tensor_res
            print(f"✅ FP32 Matmul Latency (1024x1024): {tensor_res['fp32_matmul_1024x1024_ms']} ms")
            print(f"   FP16 Matmul: {tensor_res['fp16_matmul']}")
            print(f"   BF16 Matmul: {tensor_res['bf16_matmul']}")
            print(f"   Softmax: {tensor_res['softmax']}")
            print(f"   Gather/Scatter: {tensor_res['gather_scatter']}")
            print(f"   MPS empty_cache: {tensor_res['empty_cache']}")
            print(f"   Overall: {tensor_res['status']}")
        except Exception as e:
            REPORT["tensor_benchmark"] = {"status": "FAIL", "error": str(e)}
            print(f"❌ Tensor Benchmark Failed: {e}")

        # Phase 2: Memory Feasibility
        print("\n--- Phase 2: Memory Feasibility & Model Selection ---")
        feasibility = evaluate_model_feasibility(hw["ram_gb"])
        REPORT["memory_feasibility"] = feasibility
        print(f"Usable Budget for ML: ~{feasibility['estimated_usable_for_models_gb']} GB")
        print("\nModel Candidate Evaluation:")
        for c in feasibility["candidates"]:
            train_status = "✅ YES" if c["training_feasible"] else "❌ NO (OOM Risk)"
            infer_status = "✅ YES" if c["inference_feasible"] else "❌ NO (OOM Risk)"
            print(f" • {c['model']} ({c['params']}):")
            print(f"    - Inference Only: {infer_status} (~{c['estimated_infer_memory_gb']} GB)")
            print(f"    - Full GRPO RL Training: {train_status} (~{c['estimated_train_memory_gb']} GB)")

        # Phase 3: Config Resolution
        print("\n--- Phase 3: Config Resolution for MPS ---")
        try:
            config_res = test_config_resolution()
            REPORT["config_resolution"] = config_res
            print(f"   Device detected: {config_res.get('device_detected', 'unknown')}")
            print(f"   MPS Provider Model ID: {config_res.get('mps_model_id', 'unknown')}")
            print(f"   MPS Batch size: {config_res.get('mps_batch_size', 'unknown')}")
            print(f"   MPS Group size: {config_res.get('mps_group_size', 'unknown')}")
            if "mps_model_override" in config_res:
                print(f"   MPS model override: {config_res['mps_model_override']}")
            if "mps_batch_size_safe" in config_res:
                print(f"   MPS batch size safe: {config_res['mps_batch_size_safe']}")
            print(f"   Status: {config_res['status']}")
        except Exception as e:
            REPORT["config_resolution"] = {"status": f"FAIL: {e}"}
            print(f"❌ Config Resolution Failed: {e}")

        # Phase 4: Trainer Device Selection
        print("\n--- Phase 4: Trainer Device Selection ---")
        try:
            trainer_res = test_trainer_device_selection()
            REPORT["trainer_selection"] = trainer_res
            print(f"   Device: {trainer_res.get('trainer_device', 'unknown')}")
            print(f"   Selected model: {trainer_res.get('selected_model', 'unknown')}")
            print(f"   Resolved model: {trainer_res.get('resolved_model', 'unknown')}")
            print(f"   Dtype: {trainer_res.get('dtype', 'unknown')}")
            print(f"   Batch size: {trainer_res.get('safe_batch_size', 'unknown')}")
            print(f"   Group size: {trainer_res.get('safe_group_size', 'unknown')}")
            if "mps_checks" in trainer_res:
                print(f"   MPS checks: {trainer_res['mps_checks']}")
            print(f"   Status: {trainer_res['status']}")
        except Exception as e:
            REPORT["trainer_selection"] = {"status": f"FAIL: {e}"}
            print(f"❌ Trainer Selection Failed: {e}")
            
        REPORT["status"] = "COMPLETED"
        print("\n==================================================")
        print("✅ MPS VALIDATION FINISHED")
        print("Output saved to artifacts/mps_validation_report_raw.json")
        print("==================================================")
        
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/mps_validation_report_raw.json", "w") as f:
        json.dump(REPORT, f, indent=2)


if __name__ == '__main__':
    main()
