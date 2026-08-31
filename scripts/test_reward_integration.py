# scripts/test_reward_integration.py
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from paint_rl.rewards.api import get_rewards

def test_full_pipeline():
    prompt = "Watercolor mountains"
    code = "function setup() { createCanvas(100, 100); background(255); }"
    ref = "nonexistent.png"
    
    res = get_rewards(prompt, code, ref, seed=42)
    print("Test Reward Output:", res)

if __name__ == "__main__":
    test_full_pipeline()
