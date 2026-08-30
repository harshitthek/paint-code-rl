# scripts/generate_baseline.py
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paint_rl.rewards.api import get_rewards

def generate_baseline():
    print("Generating REAL baseline sample set from SFT model...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("BLOCKED: PyTorch/Transformers not installed. Cannot run real baseline.")
        return

    if not torch.cuda.is_available():
        print("BLOCKED: No GPU available. Cannot run real 7B baseline.")
        return

    print("Loading Qwen2.5-Coder-7B-Instruct...")
    model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    with open("datasets/prompts_train.json", "r") as f:
        prompts = json.load(f)
        
    with open("datasets/reference_pool/references.json", "r") as f:
        refs = json.load(f)
        
    for p in prompts:
        print(f"Generating for prompt: {p}")
        inputs = tokenizer(p, return_tensors="pt").to(model.device)
        outputs = model.generate(**inputs, max_new_tokens=512, do_sample=True, top_p=0.9, temperature=0.7)
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        ref_path = refs[0]["path"]
        reward = get_rewards(prompt=p, code=code, reference_path=ref_path, seed=42)
        print(f"Total Reward: {reward['total']:.2f}")

if __name__ == '__main__':
    generate_baseline()
