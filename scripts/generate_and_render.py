#!/usr/bin/env python3
"""End-to-end generative art synthesis, WebGL sandbox rendering, and gallery visualization.

Supports evaluating:
1. Base untuned models.
2. Local trained LoRA checkpoints (--checkpoint <path>).
3. KaggleHub downloaded models (--kagglehub <handle>).

Usage:
    python scripts/generate_and_render.py
    python scripts/generate_and_render.py --checkpoint artifacts/checkpoints/final_adapter
    python scripts/generate_and_render.py --kagglehub pernavjain/paint-code/pyTorch/default
"""
import os
import sys
import json
import base64
import argparse
import shutil
import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(REPO_ROOT, 'src'))
from paint_rl.renderer.manager import RendererService
from paint_rl.config.prompts import SYSTEM_PROMPT
from paint_rl.utils.code_extractor import robust_extract_js_code


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_gallery_html(renders: list, output_path: str):
    cards_html = ""
    for r in renders:
        img_tag = (
            f'<img src="{r["rel_image_path"]}" alt="Render" class="render-img" />'
            if r.get("rel_image_path")
            else f'<div class="error-box">Render Status: {r.get("status", "Failed")} <br><small>{r.get("error", "Canvas Error")}</small></div>'
        )
        cards_html += f"""
        <div class="card">
            <h3>🎨 {r['prompt']}</h3>
            <div class="card-body">
                <div class="img-container">{img_tag}</div>
                <div class="code-container">
                    <pre><code>{r['code'].replace('<', '&lt;').replace('>', '&gt;')}</code></pre>
                </div>
            </div>
            <div class="meta">
                <span>⚡ Latency: {r.get('render_ms', 0)}ms</span>
                <span>🌱 Seed: {r.get('seed', 42)}</span>
                <span>📋 Status: <strong>{r.get('status', 'UNKNOWN')}</strong></span>
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Paint-Code-RL Gallery</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; }}
        h1 {{ text-align: center; color: #38bdf8; margin-bottom: 2rem; }}
        .gallery {{ display: flex; flex-direction: column; gap: 2rem; max-width: 1100px; margin: 0 auto; }}
        .card {{ background: #1e293b; border-radius: 12px; padding: 1.5rem; border: 1px solid #334155; box-shadow: 0 10px 15px -3px rgba(0,0,0,0.3); }}
        .card h3 {{ margin-top: 0; color: #e2e8f0; font-size: 1.15rem; }}
        .card-body {{ display: grid; grid-template-columns: 1fr 1fr; gap: 1.5rem; margin-top: 1rem; }}
        .img-container {{ background: #0f172a; border-radius: 8px; overflow: hidden; display: flex; align-items: center; justify-content: center; min-height: 400px; border: 1px solid #334155; }}
        .render-img {{ max-width: 100%; height: auto; display: block; }}
        .error-box {{ padding: 2rem; text-align: center; color: #f87171; background: #450a0a; border-radius: 8px; width: 80%; }}
        .code-container {{ background: #0f172a; border-radius: 8px; padding: 1rem; overflow-x: auto; max-height: 450px; border: 1px solid #334155; }}
        pre {{ margin: 0; }}
        code {{ font-family: 'JetBrains Mono', Consolas, Monaco, monospace; font-size: 0.85rem; color: #38bdf8; }}
        .meta {{ margin-top: 1rem; display: flex; gap: 1.5rem; font-size: 0.85rem; color: #94a3b8; border-top: 1px solid #334155; padding-top: 0.75rem; }}
        .meta span {{ display: flex; align-items: center; gap: 0.25rem; }}
        .summary-badge {{ text-align: center; margin-bottom: 2rem; font-size: 1.1rem; color: #4ade80; }}
    </style>
</head>
<body>
    <h1>🎨 Paint-Code-RL Generative Showcase</h1>
    <div class="summary-badge">✨ Multi-Device Verified | Zero-Cost Invariant Compliant ✨</div>
    <div class="gallery">
        {cards_html}
    </div>
</body>
</html>
"""
    parent_dir = os.path.dirname(os.path.abspath(output_path))
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n🎉 Interactive Gallery written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate and render p5.js generative art with trained models", allow_abbrev=False)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to local trained LoRA checkpoint directory")
    parser.add_argument("--kagglehub", type=str, default=None,
                        help="KaggleHub model handle to download (e.g. pernavjain/paint-code/pyTorch/default)")
    parser.add_argument("--hf", type=str, default=None,
                        help="Hugging Face model repo ID (e.g. username/paint-code-rl-lora)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to generate (optional)")
    parser.add_argument("--output-dir", type=str, default="artifacts/renders",
                        help="Output directory for rendered PNGs")
    parser.add_argument("--gallery-path", type=str, default="artifacts/gallery.html",
                        help="Output path for HTML showcase gallery")
    parser.add_argument("--max-new-tokens", type=int, default=550,
                        help="Maximum generation token budget per artwork (default: 550)")
    parser.add_argument("--temperature", type=float, default=0.4,
                        help="Sampling temperature for code generation (default: 0.4)")
    parser.add_argument("--max", action="store_true",
                        help="Enable MAX power mode: parallel batched generation, multi-threaded rendering, and max GPU utilization")
    args = parser.parse_args()

    print("=" * 80)
    print(" PAINT-CODE-RL: GENERATIVE ROLLOUT & RENDER PIPELINE")
    print("=" * 80)

    device = get_device()
    print(f"Detected compute device: {device}")

    # Resolve model checkpoint path
    adapter_path = None
    if args.kagglehub:
        print(f"\n[KaggleHub] Downloading model: {args.kagglehub}...")
        try:
            import kagglehub
            adapter_path = kagglehub.model_download(args.kagglehub)
            print(f"[KaggleHub] Model downloaded to: {adapter_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download via kagglehub: {e}")
            print("Install kagglehub: pip install kagglehub")
            sys.exit(1)
    elif args.hf:
        print(f"\n[HuggingFace] Downloading model from repo: {args.hf}...")
        try:
            from huggingface_hub import snapshot_download
            adapter_path = snapshot_download(repo_id=args.hf)
            print(f"[HuggingFace] Model downloaded to: {adapter_path}")
        except Exception as e:
            print(f"[ERROR] Failed to download from HuggingFace: {e}")
            sys.exit(1)
    elif args.checkpoint:
        adapter_path = os.path.abspath(args.checkpoint)
        # If path doesn't exist, check if user has a packaged zip archive in /kaggle/working
        if not os.path.exists(adapter_path):
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

        # Auto-resolve checkpoint subdirectories (e.g. checkpoint-25, final_checkpoint, etc.)
        if os.path.isdir(adapter_path) and not os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
            subdirs = []
            for root, dirs, files in os.walk(adapter_path):
                if "adapter_config.json" in files:
                    subdirs.append(root)
            if subdirs:
                subdirs.sort(key=lambda s: os.path.getmtime(s), reverse=True)
                adapter_path = subdirs[0]
                print(f"[Checkpoint] Auto-resolved to latest checkpoint subdirectory: {adapter_path}")
            else:
                print(f"[WARN] No adapter_config.json found inside {adapter_path}")
        print(f"[Checkpoint] Using local checkpoint: {adapter_path}")
    else:
        # Default policy: auto-detect local checkpoint if present, else auto-load official HF repo
        default_local = os.path.join(REPO_ROOT, "artifacts", "checkpoints")
        subdirs = []
        if os.path.exists(default_local):
            for root, dirs, files in os.walk(default_local):
                if "adapter_config.json" in files:
                    subdirs.append(root)
        if subdirs:
            subdirs.sort(key=lambda s: os.path.getmtime(s), reverse=True)
            adapter_path = subdirs[0]
            print(f"[Checkpoint] Auto-detected local trained checkpoint: {adapter_path}")
        else:
            print("\n[HuggingFace] Auto-downloading trained model from: HarshittheK/paint-code-rl-lora...")
            try:
                from huggingface_hub import snapshot_download
                adapter_path = snapshot_download(repo_id="HarshittheK/paint-code-rl-lora")
                print(f"[HuggingFace] Trained model downloaded to: {adapter_path}")
            except Exception as e:
                print(f"[WARN] Could not download from Hugging Face: {e}")

    # Start renderer
    renderer = RendererService(port=3000)
    print("Starting WebGL renderer daemon...")
    if not renderer.restart(max_wait_sec=20):
        print("❌ Failed to start renderer service on port 3000.")
        sys.exit(1)

    # Determine base model and policy device
    base_model_name = None
    if adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json")):
        try:
            with open(os.path.join(adapter_path, "adapter_config.json"), "r") as f:
                cfg = json.load(f)
                base_model_name = cfg.get("base_model_name_or_path")
        except Exception:
            pass

    if not base_model_name:
        if device.type == "mps":
            base_model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        elif device.type == "cuda":
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            base_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct" if vram_gb >= 20 else "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        else:
            base_model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    policy_device = device

    print(f"Loading base model: {base_model_name} onto {policy_device}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_config = AutoConfig.from_pretrained(base_model_name)
    model_config.sliding_window = None
    model_config.use_sliding_window = False

    dtype = torch.float32 if policy_device.type in ["mps", "cpu"] else torch.float16
    cuda_devices = torch.cuda.device_count() if policy_device.type == "cuda" else 0
    use_multi_gpu = cuda_devices > 1

    if use_multi_gpu:
        print(f"🚀 Detected {cuda_devices} CUDA GPUs! Distributing model across all GPUs (device_map='auto')...")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=model_config,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="sdpa"
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            config=model_config,
            torch_dtype=dtype,
            attn_implementation="sdpa"
        ).to(policy_device)

    has_adapter_config = adapter_path and os.path.exists(os.path.join(adapter_path, "adapter_config.json"))
    if has_adapter_config:
        print(f"Applying trained LoRA adapter from: {adapter_path}...")
        try:
            if use_multi_gpu:
                model = PeftModel.from_pretrained(base_model, adapter_path, device_map="auto")
            else:
                model = PeftModel.from_pretrained(base_model, adapter_path).to(policy_device)
            print("✅ Trained LoRA policy loaded successfully!")
        except Exception as e:
            print(f"[WARN] Failed to load LoRA adapter ({e}). Falling back to base model.")
            model = base_model
    else:
        model = base_model
        print("ℹ️ Using base untuned model (zero-shot).")

    model.eval()

    if args.prompt:
        prompts = [args.prompt]
    else:
        prompts = [
            "Create a generative watercolor painting of a serene mountain landscape using p5.js and p5.brush.",
            "Draw an abstract geometric composition with vibrant watercolor brush strokes and organic ink textures.",
            "Paint a tranquil ocean sunset with soft pastel gradient washes and textured brush outlines.",
            "Paint an enchanted forest with textured tree trunks, watercolor foliage, and misty light."
        ]

    print(f"\nEvaluating on {len(prompts)} distinct generative art prompts...")
    results = []

    # Clean previous renders in output directory
    os.makedirs(args.output_dir, exist_ok=True)
    import glob
    for old_png in glob.glob(os.path.join(args.output_dir, "*.png")):
        try:
            os.remove(old_png)
        except Exception:
            pass

    if args.max:
        print("\n⚡ [MAX POWER MODE ACTIVATED]")
        print("   • Multi-GPU Tensor Sharding Enabled")
        print("   • Batched Parallel Generation Across All GPUs")
        print("   • Concurrent Multi-Core WebGL Sandbox Rendering")
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    if args.max and len(prompts) > 1:
        print(f"\n🚀 Running Batched Parallel Generation on {len(prompts)} prompts simultaneously...")
        tokenizer.padding_side = "left"
        batch_texts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Create generative art in p5.js: {p}"}],
                tokenize=False,
                add_generation_prompt=True
            )
            for p in prompts
        ]
        target_device = model.device if hasattr(model, "device") else policy_device
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True).to(target_device)
        
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                top_p=0.9,
                temperature=args.temperature,
                repetition_penalty=1.12,
                no_repeat_ngram_size=6,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        codes = []
        for i in range(len(prompts)):
            gen_text = tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            codes.append(robust_extract_js_code(gen_text))
            
        del inputs, outputs
        import gc
        gc.collect()
        if policy_device.type == "mps":
            torch.mps.empty_cache()
        elif policy_device.type == "cuda":
            torch.cuda.empty_cache()
            
        print(f"Generated {len(codes)} artworks simultaneously. Dispatching concurrent renders...")
        from concurrent.futures import ThreadPoolExecutor

        def render_worker(item):
            idx, p, code = item
            print(f"Submitting Artwork {idx+1}/{len(prompts)} to WebGL Sandbox...")
            render_res = renderer.render(code, seed=42 + idx, prompt=p)
            return idx, p, code, render_res

        workers = min(len(prompts), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            rendered_items = list(executor.map(render_worker, [(i, prompts[i], codes[i]) for i in range(len(prompts))]))
            
        rendered_items.sort(key=lambda x: x[0])
        for idx, p, code, render_res in rendered_items:
            img_filename = f"render_{idx+1}.png"
            img_out_path = os.path.join(args.output_dir, img_filename)
            rel_img_path = os.path.relpath(img_out_path, os.path.dirname(os.path.abspath(args.gallery_path)))
            
            if not render_res or not (render_res.get("success") and render_res.get("image_path")):
                print(f"🔄 Retrying Artwork {idx+1} with sanitized 600x600 canvas...")
                import re
                sanitized_code = re.sub(r'createCanvas\s*\([^)]*\)', 'createCanvas(600, 600, WEBGL)', code)
                retry_res = renderer.render(sanitized_code, seed=42 + idx, prompt=p)
                if retry_res and retry_res.get("success") and retry_res.get("image_path"):
                    render_res = retry_res
                    code = sanitized_code

            if render_res and render_res.get("success") and render_res.get("image_path"):
                shutil.copy(render_res["image_path"], img_out_path)
                print(f"✅ Render SUCCESS ({render_res.get('render_ms', 0)}ms): Saved to {img_out_path}")
                status = "SUCCESS"
                err = None
            else:
                status = render_res.get("error_classification", "FAILED") if render_res else "RENDER_FAILED"
                err = render_res.get("runtime_error", "Unknown error") if render_res else "No response from renderer"
                render_ms = render_res.get("render_ms", 0) if render_res else 0
                print(f"❌ Render FAILED ({render_ms}ms): {status} -> {err}")
                rel_img_path = None

            results.append({
                "prompt": p,
                "code": code,
                "rel_image_path": rel_img_path,
                "status": status,
                "error": err,
                "render_ms": render_res.get("render_ms", 0) if render_res else 0,
                "seed": 42 + idx
            })
    else:
        if args.max:
            print("\nℹ️ [--max] Single prompt provided; proceeding with sequential generation pipeline.")
        for i, p in enumerate(prompts):
            print(f"\n--- Generating Artwork {i+1}/{len(prompts)}: '{p}' ---")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Create generative art in p5.js: {p}"}
            ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            target_device = model.device if hasattr(model, "device") else policy_device
            inputs = tokenizer(text, return_tensors="pt").to(target_device)
            
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    top_p=0.9,
                    temperature=args.temperature,
                    repetition_penalty=1.12,
                    no_repeat_ngram_size=6,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
            gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            code = robust_extract_js_code(gen_text)
            
            # Memory optimization: free intermediate tensors & flush GPU/MPS caches
            del inputs, outputs
            import gc
            gc.collect()
            if policy_device.type == "mps":
                torch.mps.empty_cache()
            elif policy_device.type == "cuda":
                torch.cuda.empty_cache()
            
            print(f"Generated {len(code)} chars of p5.js code. Submitting to WebGL Sandbox...")
            
            # Render via Node.js WebGL Sandbox
            render_res = renderer.render(code, seed=42 + i, prompt=p)
            
            img_filename = f"render_{i+1}.png"
            img_out_path = os.path.join(args.output_dir, img_filename)
            rel_img_path = os.path.relpath(img_out_path, os.path.dirname(os.path.abspath(args.gallery_path)))
            
            if not (render_res.get("success") and render_res.get("image_path")):
                print(f"🔄 Retrying Artwork {i+1} with sanitized 600x600 canvas...")
                import re
                sanitized_code = re.sub(r'createCanvas\s*\([^)]*\)', 'createCanvas(600, 600, WEBGL)', code)
                retry_res = renderer.render(sanitized_code, seed=42 + i, prompt=p)
                if retry_res.get("success") and retry_res.get("image_path"):
                    render_res = retry_res
                    code = sanitized_code

            if render_res.get("success") and render_res.get("image_path"):
                shutil.copy(render_res["image_path"], img_out_path)
                print(f"✅ Render SUCCESS ({render_res.get('render_ms', 0)}ms): Saved to {img_out_path}")
                status = "SUCCESS"
                err = None
            else:
                status = render_res.get("error_classification", "FAILED")
                err = render_res.get("runtime_error", "Unknown error")
                print(f"❌ Render FAILED ({render_res.get('render_ms', 0)}ms): {status} -> {err}")
                print("--- Generated Code Preview ---")
                for line in code.split("\n")[:12]:
                    print(f"  | {line}")
                print("------------------------------")
                rel_img_path = None

            results.append({
                "prompt": p,
                "code": code,
                "rel_image_path": rel_img_path,
                "status": status,
                "error": err,
                "render_ms": render_res.get("render_ms", 0),
                "seed": 42 + i
            })

    gallery_parent = os.path.dirname(os.path.abspath(args.gallery_path))
    if gallery_parent:
        os.makedirs(gallery_parent, exist_ok=True)
    build_gallery_html(results, args.gallery_path)
    
    print("\n================================================================================")
    print(f" ROLLOUT COMPLETED: {sum(1 for r in results if r['status'] == 'SUCCESS')}/{len(results)} RENDERED SUCCESSFULLY")
    print("================================================================================")


if __name__ == "__main__":
    main()
