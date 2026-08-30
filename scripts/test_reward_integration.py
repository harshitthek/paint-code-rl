# scripts/test_reward_integration.py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paint_rl.rewards.api import get_rewards

def run():
    print("Running Reward Integration Test (Gates C1, C2, D1)...")
    try:
        res = get_rewards(
            prompt="draw a red circle", 
            code="function setup() { createCanvas(400,400,WEBGL); background(255); window.signalRenderComplete(); }", 
            reference_path="mock_ref.png", 
            seed=42
        )
        print("Reward bundle returned:")
        import pprint
        pprint.pprint(res)
    except Exception as e:
        print(f"Integration failed: {e}")

if __name__ == '__main__':
    run()
