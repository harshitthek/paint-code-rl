# scripts/generate_baseline.py
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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
        print(f"\n==================================================")
        print(f"Prompt {i+1}: {p}")
        print(f"==================================================")
        
        messages = [
            {"role": "system", "content": "You are an expert generative artist who writes executable p5.js code using the p5.brush library."},
            {"role": "user", "content": f"{p}\nWrite only executable p5.js code inside a `javascript block."}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        # Strip input tokens from output
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        code = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        print("Generated Code Output:\n")
        print(code)

if __name__ == '__main__':
    generate_baseline()
