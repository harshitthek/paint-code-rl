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
import subprocess

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# Suppress tokenizer parallelism warning before importing transformers
os.environ["TOKENIZERS_PARALLELISM"] = "false"


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


def extract_js_code(raw_text):
    import re
    match = re.search(r'```(?:javascript|js)?\s*\n([\s\S]*?)```', raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


SYSTEM_PROMPT = (
    "You are an expert generative artist writing p5.js code using the p5.brush library.\n\n"
    "Reference working template:\n"
    "```javascript\n"
    "function setup() {\n"
    "    createCanvas(600, 600, WEBGL);\n"
    "    background(245, 243, 238);\n"
    "    brush.load();\n"
    "    brush.scaleBrushes(3);\n"
    "    noLoop();\n"
    "}\n\n"
    "function draw() {\n"
    "    translate(-width/2, -height/2);\n"
    "    // Stroke brushes: 'HB', '2B', '2H', 'cpencil', 'pen', 'rotring', 'spray',\n"
    "    //                  'marker', 'marker2', 'charcoal', 'hatch_brush'\n"
    "    brush.set('charcoal', '#3a6073', 2);\n"
    "    brush.line(50, 50, 550, 550);\n"
    "    // Watercolor fill (not a stroke brush!):\n"
    "    brush.fill('#1a759f', 160);\n"
    "    brush.fillBleed(0.3, 'out');\n"
    "    brush.rect(100, 100, 400, 400);\n"
    "}\n"
    "```\n\n"
    "Rules:\n"
    "1. In setup(), call createCanvas(600, 600, WEBGL), background(...), brush.load(), brush.scaleBrushes(3), and noLoop().\n"
    "2. In WEBGL mode, origin is center. Use translate(-width/2, -height/2) in draw().\n"
    "3. Valid stroke brushes: HB, 2B, 2H, cpencil, pen, rotring, spray, marker, marker2, charcoal.\n"
    "4. For watercolor effects use brush.fill(color, opacity) + brush.fillBleed(). 'watercolor' is NOT a brush name.\n"
    "5. Respond with ONLY executable p5.js code inside a ```javascript block."
)


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
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32
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
            {"role": "user", "content": f"Create generative art: {prompt_text}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=550,
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_js_code(gen_text)
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
            import shutil
            img_path = os.path.join(args.output_dir, f"{prompt_id}.png")
            try:
                shutil.copy(render_res["image_path"], img_path)
                print(f"  ✅ RENDERED: {img_path} ({render_latency_ms:.0f}ms)")
            except Exception as e:
                print(f"  ⚠️  File copy failed: {e}")
                img_path = None
        else:
            status = render_res.get("error_classification", "UNKNOWN")
            print(f"  ❌ RENDER FAILED: {status} ({render_latency_ms:.0f}ms)")

        # Build metadata
        metadata = {
            "run_id": run_id,
            "prompt_id": prompt_id,
            "prompt": prompt_text,
            "seed": seed,
            "model_id": model_name,
            "model_revision": "main",
            "decoding": {"max_new_tokens": 550, "temperature": 0.7, "top_p": 0.9},
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
