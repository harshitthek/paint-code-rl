import sys
import os

root = r"C:\Users\user\.gemini\antigravity\brain\eabfab2e-f626-4128-9da1-6868c5d0f842\paint-code-rl"
src = os.path.join(root, "src")
sys.path.insert(0, src)
sys.path.insert(0, root)

modules_to_test = [
    "paint_rl.interfaces",
    "paint_rl.config.core",
    "paint_rl.models.registry",
    "paint_rl.rewards.api",
    "paint_rl.rewards.validation",
    "paint_rl.rewards.hpsv3_score",
    "paint_rl.rewards.pairwise_vlm",
    "paint_rl.storage.cache",
    "paint_rl.telemetry.core",
    "paint_rl.trainer.async_rollout",
    "paint_rl.trainer.checkpoint_validator",
    "paint_rl.trainer.train_grpo",
]

for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"[OK] Successfully imported {mod}")
    except Exception as e:
        print(f"[FAIL] Error importing {mod}: {e}")
