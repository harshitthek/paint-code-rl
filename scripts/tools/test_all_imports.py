import sys
import os
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent.parent)
src = os.path.join(root, "src")
sys.path.insert(0, src)
sys.path.insert(0, root)

modules_to_test = [
    "paint_rl.interfaces",
    "paint_rl.config.core",
    "paint_rl.models.registry",
    "paint_rl.rewards.api",
    "paint_rl.rewards.validation",
    "paint_rl.rewards.aesthetic",
    "paint_rl.rewards.composer",
    "paint_rl.rewards.components",
    "paint_rl.rewards.pairwise_vlm",
    "paint_rl.storage.cache",
    "paint_rl.telemetry.core",
    "paint_rl.telemetry.dashboard",
    "paint_rl.trainer.async_rollout",
    "paint_rl.trainer.checkpoint_validator",
    "paint_rl.trainer.grpo",
]

failed = False
for mod in modules_to_test:
    try:
        __import__(mod)
        print(f"[OK] Successfully imported {mod}")
    except Exception as e:
        print(f"[FAIL] Error importing {mod}: {e}")
        failed = True

if failed:
    sys.exit(1)
else:
    sys.exit(0)
