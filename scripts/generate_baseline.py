# scripts/generate_baseline.py
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from paint_rl.rewards.api import get_rewards

def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def generate_baseline():
    print("Generating baseline sample set...")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("BLOCKED: PyTorch/Transformers not installed. Cannot run real baseline.")
        return

    device = get_device()
    print(f"Target compute device: {device}")

    # Select model based on device/memory
    if device.type == "mps":
        # Safe default for 16GB Apple Silicon
        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    elif device.type == "cuda":
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    print(f"Loading candidate model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32
    ).to(device)
    
    prompts = [
        "Create a generative watercolor painting of a mountain landscape with soft mist using p5.js and p5.brush.",
        "Draw an abstract geometric composition with vibrant brush textures and flowing ink lines."
    ]
    
    print("\nStarting generation test...")
    for i, p in enumerate(prompts):
        print(f"\n--- Prompt {i+1}: {p} ---")
        inputs = tokenizer(p, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        code = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("Generated code preview (first 120 chars):")
        print(code[:120] + "...")

if __name__ == '__main__':
    generate_baseline()
