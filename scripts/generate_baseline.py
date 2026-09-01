#!/usr/bin/env python3
"""Generate baseline samples: real model → real code → real render → real rewards → metadata.

Usage:
    python scripts/generate_baseline.py [--num-samples 5] [--output-dir artifacts/baseline]
"""
import argparse
import json
import os
import sys
import time
import uuid
import shutil
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Suppress tokenizer parallelism warning before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from paint_rl.config.prompts import SYSTEM_PROMPT
from paint_rl.utils.code_extractor import robust_extract_js_code


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def select_model(device):
    if device.type == "mps":
        return "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    elif device.type == "cuda":
        return "Qwen/Qwen2.5-Coder-7B-Instruct"
    return "Qwen/Qwen2.5-Coder-0.5B-Instruct"


def load_prompts(dataset_path="datasets/prompts_v1.jsonl", limit=None):
    prompts = []
    if os.path.exists(dataset_path):
        with open(dataset_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    prompts.append(data)
    if limit:
        prompts = prompts[:limit]
    return prompts


def main():
    parser = argparse.ArgumentParser(description="Generate baseline samples")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of prompts to generate for")
    parser.add_argument("--output-dir", type=str, default="artifacts/baseline",
                        help="Output directory for renders and metadata")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_id = f"baseline_{uuid.uuid4().hex[:8]}"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("   PAINT-CODE-RL: BASELINE GENERATION")
    print("=" * 60)

    device = get_device()
    print(f"Device: {device}")

    # Load model
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = select_model(device)
    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if device.type in ["mps", "cpu"] else torch.float16
    ).to(device)

    # Start renderer
    from paint_rl.renderer.manager import RendererService
    renderer = RendererService(port=3000)
    print("Starting renderer...")
    if not renderer.ensure_started(max_wait_sec=20):
        print("[ERROR] Renderer failed to start. Run: cd renderer && npm install && node server.js")
        sys.exit(1)

    # Load prompts
    prompts = load_prompts(limit=args.num_samples)
    if not prompts:
        print("[ERROR] No prompts found in datasets/prompts_v1.jsonl")
        sys.exit(1)

    print(f"Generating {len(prompts)} samples...\n")

    results = []
    git_commit = get_git_commit()

    for i, prompt_data in enumerate(prompts):
        prompt_text = prompt_data["prompt"]
        prompt_id = prompt_data.get("prompt_id", f"prompt_{i}")
        seed = args.seed + i

        print(f"\n--- [{i+1}/{len(prompts)}] {prompt_id}: {prompt_text[:60]}... ---")

        # Generate code
        gen_start = time.time()
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create generative art in p5.js: {prompt_text}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=450,
                do_sample=True,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.12,
                no_repeat_ngram_size=6,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code = robust_extract_js_code(gen_text)
        gen_latency_ms = (time.time() - gen_start) * 1000

        print(f"  Generated {len(code)} chars ({gen_latency_ms:.0f}ms)")

        # Render
        render_start = time.time()
        render_res = renderer.render(code, seed=seed, prompt=prompt_text)
        render_latency_ms = (time.time() - render_start) * 1000

        # Save outputs
        code_path = os.path.join(args.output_dir, f"{prompt_id}_code.js")
        with open(code_path, "w") as f:
            f.write(code)

        img_path = None
        if render_res.get("success") and render_res.get("image_path"):
            img_path = os.path.join(args.output_dir, f"{prompt_id}.png")
            try:
                shutil.copy(render_res["image_path"], img_path)
                print(f"  ✅ RENDERED: {img_path} ({render_latency_ms:.0f}ms)")
            except Exception as e:
                print(f"  ⚠️  File copy failed: {e}")
                img_path = None
        else:
            status = render_res.get("error_classification", "UNKNOWN")
            err_msg = render_res.get("runtime_error", "")
            print(f"  ❌ RENDER FAILED: {status} | {err_msg} ({render_latency_ms:.0f}ms)")

        # Build metadata
        metadata = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "seed": seed,
            "model_id": model_name,
            "model_revision": "main",
            "decoding": {
                "max_new_tokens": 450,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.12,
                "no_repeat_ngram_size": 6
            },
            "renderer_version": "2.0",
            "reward_version": "2.0",
            "git_commit": git_commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "generation_latency_ms": round(gen_latency_ms, 1),
            "render_latency_ms": round(render_latency_ms, 1),
            "code_length": len(code),
            "render_success": render_res.get("success", False),
            "render_error_classification": render_res.get("error_classification"),
            "render_error": render_res.get("runtime_error"),
            "image_path": img_path,
            "code_path": code_path,
        }
        results.append(metadata)

    # Save run manifest
    manifest_path = os.path.join(args.output_dir, f"{run_id}_manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    total = len(results)
    success = sum(1 for r in results if r["render_success"])
    print(f"\n{'=' * 60}")
    print(f"BASELINE COMPLETE: {success}/{total} rendered successfully")
    print(f"Run ID: {run_id}")
    print(f"Manifest: {manifest_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
