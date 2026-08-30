from .validation import validate_reward_bundle, RewardValidationError
import requests
import os
import uuid
import time
from .hpsv3_score import compute_hpsv3
from .pairwise_vlm import OpenAIJudgeProvider
from paint_rl.config.core import ACTIVE_CONFIG
from paint_rl.telemetry.core import ExperimentLogger

class PuppeteerRenderer:
    def __init__(self, host: str, port: int, timeout_ms: int):
        self.url = f"http://{host}:{port}/render"
        self.timeout = timeout_ms / 1000.0
        
    def render(self, code: str, seed: int, run_id: str) -> dict:
        try:
            response = requests.post(self.url, json={"prompt": "", "code": code, "seed": seed}, timeout=self.timeout)
            if response.status_code == 429:
                return {"success": False, "error_classification": "RENDERER_OVERLOAD", "runtime_error": "Backpressure applied"}
            response.raise_for_status()
            return response.json()
        except Exception as e:
            return {"success": False, "error_classification": "RENDERER_HTTP_ERROR", "runtime_error": str(e)}

# Singletons for Phase 0
renderer = PuppeteerRenderer(ACTIVE_CONFIG.renderer.host, ACTIVE_CONFIG.renderer.port, ACTIVE_CONFIG.renderer.timeout_ms)
judge = OpenAIJudgeProvider(model_id=ACTIVE_CONFIG.judge.model)

def get_rewards(prompt: str, code: str, reference_path: str, seed: int) -> dict:
    run_id = f"job_{uuid.uuid4().hex[:8]}"
    start_t = time.time()
    
    render_result = renderer.render(code, seed, run_id)
    
    if not render_result.get("success"):
        return {
            "total": 0.0, "compile": 0.0, "hps": 0.0, "pair": 0.0, 
            "len": len(code), "error_class": render_result.get("error_classification"),
            "img_path": None, "judge_status": "none"
        }
        
    img_path = render_result["image_path"]
    
    try:
        hps = compute_hpsv3(img_path, prompt)
    except Exception as e:
        return {"total": 0.0, "error_class": "HPSV3_ERROR", "img_path": img_path}
    
    pair_result = judge.compare(candidate_path=img_path, reference_path=reference_path, prompt=prompt)
    pair_score = pair_result.get("score", 0.0)
    
    if pair_result.get("status") in ("VLM_API_ERROR", "VLM_MALFORMED_RESPONSE"):
        return {"total": 0.0, "error_class": pair_result["status"], "img_path": img_path}
    
    w = ACTIVE_CONFIG.reward.weights
    total = (w.compile * 1.0) + (w.hpsv3 * hps) + (w.pairwise * pair_score)
    
    bundle = {
        "total": total,
        "compile": 1.0,
        "hps": hps,
        "pair": pair_score,
        "len": len(code),
        "error_class": None,
        "img_path": img_path,
        "judge_status": pair_result["status"]
    }
    return validate_reward_bundle(bundle)
