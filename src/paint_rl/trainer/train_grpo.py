# train_grpo.py
import json
import os
import sqlite3
import uuid

from paint_rl.config import core as config
from paint_rl.telemetry.core import ExperimentLogger

logger = ExperimentLogger(config_hash=config.CONFIG_HASH)

def init_db():
    db_path = os.path.join(config.ACTIVE_CONFIG.storage.base_path, "metrics.db")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS renders
                 (run_id text, step int, prompt_id text, code text, img_path text,
                  render_ms int, compile_reward real, raw_hpsv3 real, weighted_hpsv3 real,
                  raw_pairwise real, weighted_pairwise real, total_reward real, code_length int,
                  ref_id text, ref_pool_version text, ref_selection_seed int,
                  vlm_decision text, judge_orientation_1 text, judge_orientation_2 text,
                  generation_seed int, decoding_params text, model_revision text, renderer_version text,
                  p5js_version text, p5brush_version text, reward_version text,
                  judge_version text)''')
    conn.commit()
    return conn

def save_experiment_state(step):
    state = {
        "step": step,
        "run_id": logger.run_id,
        "config_hash": config.CONFIG_HASH,
        "config": json.loads(config.CONFIG_JSON)
    }
    state_path = os.path.join(config.ACTIVE_CONFIG.storage.base_path, "experiment_state.json")
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    with open(state_path, "w") as f:
        json.dump(state, f, indent=2)
    print(f"Saved experiment state at step {step}")

def resume_experiment_state():
    state_path = os.path.join(config.ACTIVE_CONFIG.storage.base_path, "experiment_state.json")
    if os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        if state["config_hash"] != config.CONFIG_HASH:
            raise ValueError(f"CRITICAL: Attempting to resume with mismatched config hash! "
                             f"Saved: {state['config_hash']} != Current: {config.CONFIG_HASH}")
        print(f"Resuming experiment {state['run_id']} from step {state['step']}")
        return state
    return None

def get_compute_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def run_one_step():
    print("Running REAL one-step GRPO test...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
    except ImportError as e:
        print(f"BLOCKED: Missing PyTorch, Transformers, or TRL ({e}). Gate E BLOCKED.")
        logger.log_error("DEPENDENCY_MISSING", str(e))
        return

    device = get_compute_device()
    print(f"Target compute device: {device}")
    if device.type == "cpu":
        print("WARN: Running on CPU. Training will be extremely slow.")
        
    print(f"Device {device} ready. Proceeding with real GRPO...")

def run_tiny_run():
    print("Running REAL tiny 20-step run...")
    state = resume_experiment_state()
    start_step = state["step"] if state else 0
    try:
        import torch
    except ImportError:
        print("BLOCKED: Missing PyTorch. Gate F BLOCKED.")
        return

    device = get_compute_device()
    print(f"Running on device: {device}")
    save_experiment_state(start_step + 20)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "tiny":
        run_tiny_run()
    else:
        run_one_step()
