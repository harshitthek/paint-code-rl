#!/usr/bin/env python3
"""Publish trained Paint-Code-RL LoRA adapters to Kaggle Models and Hugging Face Hub.

Usage:
    # Upload to Kaggle Models (via KaggleHub)
    python scripts/upload_model.py --destination kaggle --handle your-username/paint-code/pyTorch/v2

    # Upload to Hugging Face Hub
    python scripts/upload_model.py --destination hf --repo-id your-username/paint-code-rl-lora --token hf_...

    # Upload to both platforms simultaneously
    python scripts/upload_model.py --destination both --handle your-username/paint-code/pyTorch/v2 --repo-id your-username/paint-code-rl-lora
"""
import os
import sys
import json
import argparse

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def generate_model_card(checkpoint_dir: str, base_model: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct") -> str:
    """Generate a clean model card markdown for Kaggle and Hugging Face."""
    card = f"""---
license: apache-2.0
base_model: {base_model}
tags:
  - reinforcement-learning
  - grpo
  - generative-art
  - p5.js
  - p5.brush
  - creative-coding
pipeline_tag: text-generation
---

# 🎨 Paint-Code-RL: Generative Art Coding Policy

This repository contains a trained **LoRA adapter** for `{base_model}`, trained via **Group Relative Policy Optimization (GRPO)** to write executable, high-aesthetic [p5.js](https://p5js.org/) and [p5.brush](https://github.com/acamposuribe/p5.brush) code in a WebGL sandbox.

## 🚀 Model Details
- **Base Architecture:** `{base_model}`
- **Adapter Type:** LoRA (Low-Rank Adaptation)
- **RL Framework:** TRL GRPOTrainer + Cyclic Curriculum
- **Reward Components:**
  1. `structural_syntax_reward`: Strict JavaScript syntactic correctness, canvas initialization, and natural-media brush primitives.
  2. `render_and_visual_richness_reward`: Headless WebGL canvas rasterization, color palette entropy, edge variance, and visual coverage.

## 💻 How to Use

### 1. In Paint-Code-RL Pipeline:
```bash
# Render artwork directly using this checkpoint:
python scripts/generate_and_render.py \\
  --checkpoint {checkpoint_dir} \\
  --output-dir artifacts/renders \\
  --temperature 0.4 \\
  --max-new-tokens 550
```

### 2. In Python with Transformers & PEFT:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "{base_model}"
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float16, device_map="auto")
model = PeftModel.from_pretrained(base_model, "{checkpoint_dir}")

prompt = "Create a generative watercolor painting of a serene mountain landscape using p5.js and p5.brush."
messages = [
    {{"role": "system", "content": "You are an expert generative artist who writes pure p5.js with p5.brush."}},
    {{"role": "user", "content": f"Create generative art in p5.js: {{prompt}}"}}
]
inputs = tokenizer(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=550, temperature=0.4, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
"""
    return card


def upload_to_kaggle(checkpoint_dir: str, handle: str, version_notes: str = "Trained with GRPO visual RL"):
    """Upload trained checkpoint directory to Kaggle Models via kagglehub."""
    print(f"\n[KaggleHub] Uploading checkpoint to Kaggle Models: {handle}...")
    try:
        import kagglehub
    except ImportError:
        print("[ERROR] kagglehub is not installed. Run: pip install kagglehub")
        return False

    try:
        kagglehub.model_upload(
            handle=handle,
            local_model_dir=checkpoint_dir,
            license_name="Apache 2.0",
            version_notes=version_notes
        )
        print(f"✅ Successfully published model to Kaggle: https://www.kaggle.com/models/{handle}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload to Kaggle: {e}")
        return False


def upload_to_huggingface(checkpoint_dir: str, repo_id: str, token: str = None):
    """Upload trained checkpoint directory to Hugging Face Hub."""
    print(f"\n[HuggingFace] Uploading checkpoint to Hugging Face Hub: {repo_id}...")
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("[ERROR] huggingface_hub is not installed. Run: pip install huggingface_hub")
        return False

    resolved_token = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    api = HfApi(token=resolved_token)

    # Auto-resolve username from token if placeholder is used
    if "YOUR_HF_USERNAME" in repo_id or "/" not in repo_id:
        try:
            username = api.whoami().get("name")
            if username:
                model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
                repo_id = f"{username}/{model_name}"
                print(f"[INFO] Auto-resolved Hugging Face repository to: {repo_id}")
        except Exception:
            pass

    try:
        create_repo(repo_id=repo_id, token=resolved_token, repo_type="model", exist_ok=True)
        api.upload_folder(
            folder_path=checkpoint_dir,
            repo_id=repo_id,
            repo_type="model",
            commit_message="Upload Paint-Code-RL LoRA checkpoint"
        )
        print(f"✅ Successfully published model to Hugging Face: https://huggingface.co/{repo_id}")
        return True
    except Exception as e:
        print(f"❌ Failed to upload to Hugging Face: {e}")
        if not resolved_token:
            print("Tip: Provide your Hugging Face write token with --token or set the HF_TOKEN environment variable.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload Paint-Code-RL LoRA adapters to Kaggle Models and Hugging Face Hub")
    parser.add_argument("--destination", choices=["kaggle", "hf", "both"], default="kaggle",
                        help="Target platform: 'kaggle', 'hf', or 'both'")
    parser.add_argument("--checkpoint-dir", type=str, default="artifacts/checkpoints",
                        help="Path to local trained LoRA checkpoint directory (default: artifacts/checkpoints)")
    parser.add_argument("--handle", type=str, default=None,
                        help="Kaggle model handle, e.g. 'username/paint-code/pyTorch/default' or 'username/paint-code/pyTorch/v2'")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="Hugging Face repo ID, e.g. 'username/paint-code-rl-lora'")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face API Write Token (or set HF_TOKEN env var)")
    parser.add_argument("--version-notes", type=str, default="Trained with GRPO cyclic reinforcement learning",
                        help="Version changelog notes for Kaggle Models")
    args = parser.parse_args()

    checkpoint_dir = os.path.abspath(args.checkpoint_dir)
    if not os.path.exists(checkpoint_dir):
        # Check if user has a packaged zip archive in /kaggle/working
        import glob, zipfile
        zips = sorted(glob.glob("/kaggle/working/paint_rl_artifacts_*.zip") + glob.glob("paint_rl_artifacts_*.zip"), reverse=True)
        if zips:
            print(f"[INFO] Found packaged artifacts archive: {zips[0]}. Automatically extracting checkpoints...")
            try:
                with zipfile.ZipFile(zips[0], "r") as zf:
                    zf.extractall(REPO_ROOT)
                print("✅ Checkpoint extracted from archive successfully!")
            except Exception as ze:
                print(f"[WARN] Could not extract archive: {ze}")

    if not os.path.exists(checkpoint_dir):
        print(f"[ERROR] Checkpoint directory not found at: {checkpoint_dir}")
        sys.exit(1)

    # Check that required adapter weights exist (search recursively)
    candidate_dirs = []
    if os.path.exists(os.path.join(checkpoint_dir, "adapter_config.json")) and (
        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.safetensors")) or
        os.path.exists(os.path.join(checkpoint_dir, "adapter_model.bin"))
    ):
        candidate_dirs.append(checkpoint_dir)
    
    for root, dirs, files in os.walk(checkpoint_dir):
        if "adapter_config.json" in files and ("adapter_model.safetensors" in files or "adapter_model.bin" in files):
            if root not in candidate_dirs:
                candidate_dirs.append(root)

    resolved_adapter_dir = None
    if candidate_dirs:
        final_matches = [d for d in candidate_dirs if "final_adapter" in d]
        if final_matches:
            resolved_adapter_dir = final_matches[0]
        else:
            candidate_dirs.sort(key=lambda d: os.path.getmtime(d), reverse=True)
            resolved_adapter_dir = candidate_dirs[0]

    if not resolved_adapter_dir:
        print(f"[ERROR] No valid adapter directory containing adapter_config.json and weights found in: {checkpoint_dir}")
        sys.exit(1)

    checkpoint_dir = resolved_adapter_dir
    adapter_config_path = os.path.join(checkpoint_dir, "adapter_config.json")
    adapter_weights_path = os.path.join(checkpoint_dir, "adapter_model.safetensors")
    print(f"[INFO] Using resolved adapter directory: {checkpoint_dir}")

    print("=" * 80)
    print(" PAINT-CODE-RL: MODEL PUBLISHING ASSISTANT")
    print("=" * 80)
    print(f"Source Checkpoint Directory: {checkpoint_dir}")

    # Inspect adapter config for base model
    base_model = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    if os.path.exists(adapter_config_path):
        try:
            with open(adapter_config_path, "r") as f:
                cfg = json.load(f)
                base_model = cfg.get("base_model_name_or_path", base_model)
        except Exception:
            pass

    # Ensure README.md / model card exists in checkpoint dir
    readme_path = os.path.join(checkpoint_dir, "README.md")
    if not os.path.exists(readme_path):
        print("Generating comprehensive model card README.md...")
        card_content = generate_model_card(checkpoint_dir, base_model)
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_content)

    success = True
    if args.destination in ["kaggle", "both"]:
        if not args.handle:
            print("[ERROR] --handle is required for Kaggle upload (e.g. --handle your-username/paint-code/pyTorch/v2)")
            success = False
        else:
            k_ok = upload_to_kaggle(checkpoint_dir, args.handle, args.version_notes)
            if not k_ok:
                success = False

    if args.destination in ["hf", "both"]:
        if not args.repo_id:
            print("[ERROR] --repo-id is required for Hugging Face upload (e.g. --repo-id your-username/paint-code-rl-lora)")
            success = False
        else:
            hf_ok = upload_to_huggingface(checkpoint_dir, args.repo_id, args.token)
            if not hf_ok:
                success = False

    if success:
        print("\n🎉 Publishing process finished successfully!")
    else:
        print("\n⚠️ Publishing process encountered errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
