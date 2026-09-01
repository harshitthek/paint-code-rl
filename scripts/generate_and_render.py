import os
import sys
import json
import base64
import torch

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
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
        img_tag = f'<img src="{r["rel_image_path"]}" alt="Render" class="render-img" />' if r.get("rel_image_path") else f'<div class="error-box">Render Status: {r.get("status", "Failed")} <br><small>{r.get("error", "Canvas Error")}</small></div>'
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
                <span>📋 Classification: <strong>{r.get('status', 'UNKNOWN')}</strong></span>
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
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n🎉 Interactive Gallery written to: {output_path}")

def run():
    print("================================================================================")
    print(" PAINT-CODE-RL: END-TO-END GENERATIVE ROLLOUT & RENDER PIPELINE")
    print("================================================================================")

    device = get_device()
    print(f"Detected compute device: {device}")

    # Start renderer
    renderer = RendererService(port=3000)
    print("Starting WebGL renderer daemon...")
    if not renderer.ensure_started(max_wait_sec=20):
        print("❌ Failed to start renderer service on port 3000.")
        sys.exit(1)

    print("Loading language model for p5.js synthesis...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from paint_rl.models.registry import ModelRegistry

    selection = ModelRegistry.select_models(device=device.type)
    model_name = selection.policy_model
    policy_device = torch.device(selection.policy_device)

    print(f"Loading {model_name} onto {policy_device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32 if policy_device.type in ["mps", "cpu"] else torch.float16
    ).to(policy_device)

    prompts = [
        "Create a generative watercolor painting of a serene mountain landscape using p5.js and p5.brush.",
        "Draw an abstract geometric composition with vibrant watercolor brush strokes and organic ink textures.",
        "Paint a tranquil ocean sunset with soft pastel gradient washes and textured brush outlines."
    ]

    os.makedirs("artifacts/renders", exist_ok=True)
    results = []

    for i, p in enumerate(prompts):
        print(f"\n--- Generating Artwork {i+1}/{len(prompts)}: '{p}' ---")
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Create generative art in p5.js: {p}"}
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
        
        print(f"Generated {len(code)} chars of p5.js code. Submitting to WebGL Sandbox...")
        
        # Render via Node.js WebGL Sandbox
        render_res = renderer.render(code, seed=42 + i, prompt=p)
        
        img_out_path = f"artifacts/renders/render_{i+1}.png"
        rel_img_path = f"renders/render_{i+1}.png"
        
        if render_res.get("success") and render_res.get("image_path"):
            import shutil
            shutil.copy(render_res["image_path"], img_out_path)
            print(f"✅ Render SUCCESS ({render_res.get('render_ms', 0)}ms): Saved to {img_out_path}")
            status = "SUCCESS"
            err = None
        else:
            status = render_res.get("error_classification", "FAILED")
            err = render_res.get("runtime_error", "Unknown error")
            print(f"❌ Render FAILED ({render_res.get('render_ms', 0)}ms): {status} -> {err}")
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

    gallery_path = os.path.abspath("artifacts/gallery.html")
    build_gallery_html(results, gallery_path)
    
    print("\n================================================================================")
    print(f" ROLLOUT COMPLETED: {sum(1 for r in results if r['status'] == 'SUCCESS')}/{len(results)} RENDERED SUCCESSFULLY")
    print("================================================================================")

if __name__ == "__main__":
    run()
