import os
import sys
import json
import re
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from paint_rl.renderer.manager import RendererService

def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def extract_js_code(raw_text: str) -> str:
    match = re.search(r'```(?:javascript|js)?\s*\n([\s\S]*?)```', raw_text)
    if match:
        return match.group(1).strip()
    # Fallback: clean out non-code headers if present
    return raw_text.strip()

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
        .img-container {{ display: flex; align-items: center; justify-content: center; background: #020617; border-radius: 8px; overflow: hidden; min-height: 300px; padding: 1rem; text-align: center; }}
        .render-img {{ max-width: 100%; max-height: 400px; object-fit: contain; border-radius: 4px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.5); }}
        .code-container {{ background: #090d16; border-radius: 8px; padding: 1rem; overflow-x: auto; max-height: 400px; }}
        pre {{ margin: 0; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem; color: #a5f3fc; line-height: 1.4; }}
        .meta {{ display: flex; gap: 2rem; margin-top: 1rem; font-size: 0.85rem; color: #94a3b8; border-top: 1px solid #334155; padding-top: 0.75rem; }}
        .error-box {{ color: #f87171; font-family: monospace; font-size: 0.85rem; }}
    </style>
</head>
<body>
    <h1>🎨 Paint-Code-RL: Live Generative Canvas Gallery</h1>
    <div class="gallery">
        {cards_html}
    </div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n✨ Interactive Gallery generated at: {output_path}")

def main():
    print("==================================================")
    print("   PAINT-CODE-RL: END-TO-END GENERATE & RENDER    ")
    print("==================================================")
    
    # 1. Start Renderer Daemon First
    renderer = RendererService(port=3000)
    print("Initializing WebGL Sandbox Daemon...")
    if not renderer.ensure_started(max_wait_sec=15):
        print("[ERROR] Failed to start WebGL renderer daemon.")
        return

    device = get_device()
    print(f"Target compute device: {device}")
    
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("[ERROR] PyTorch or Transformers missing.")
        return

    # Select model
    if device.type == "mps":
        model_name = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
    elif device.type == "cuda":
        model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
    else:
        model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

    print(f"Loading {model_name} onto {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32
    ).to(device)

    prompts = [
        "Create a generative watercolor painting of a serene mountain landscape using p5.js and p5.brush.",
        "Draw an abstract geometric composition with vibrant watercolor brush strokes and organic ink textures.",
        "Paint a tranquil ocean sunset with soft pastel gradient washes and textured brush outlines."
    ]

    os.makedirs("artifacts/renders", exist_ok=True)
    results = []

    # Accurate p5.brush few-shot grammar
    system_prompt = (
        "You are an expert generative artist writing p5.js code using the p5.brush library.\n\n"
        "Reference working template:\n"
        "```javascript\n"
        "function setup() {\n"
        "    createCanvas(600, 600, WEBGL);\n"
        "    background(245, 243, 238);\n"
        "    brush.load();\n"
        "    noLoop();\n"
        "}\n\n"
        "function draw() {\n"
        "    translate(-width/2, -height/2);\n"
        "    // Use brush.set(brush_name, color_hex_or_rgb, weight)\n"
        "    // Valid brush names: 'watercolor', 'charcoal', 'spray', 'marker', 'rotring', 'cpencil'\n"
        "    brush.set('watercolor', '#3a6073', 2);\n"
        "    brush.rect(50, 50, 500, 500);\n"
        "    brush.set('charcoal', '#1a1a1a', 1.5);\n"
        "    brush.circle(300, 300, 150);\n"
        "}\n"
        "```\n\n"
        "Rules:\n"
        "1. In setup(), call createCanvas(600, 600, WEBGL), background(...), brush.load(), and noLoop().\n"
        "2. In WEBGL mode, the origin (0,0) is in the center. Use translate(-width/2, -height/2) in draw() if using 0-600 coordinates.\n"
        "3. Only use valid brush functions: brush.set(), brush.rect(), brush.circle(), brush.line(), brush.bleed().\n"
        "4. Respond with ONLY executable p5.js code inside a ```javascript block."
    )

    for i, p in enumerate(prompts):
        print(f"\n--- Generating Artwork {i+1}/{len(prompts)}: '{p}' ---")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create generative art matching this prompt: {p}"}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=550,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_js_code(gen_text)
        
        print(f"Generated {len(code)} chars of p5.js code. Submitting to WebGL Sandbox...")
        
        # Render via Node.js WebGL Sandbox
        render_res = renderer.render(code, seed=42 + i, prompt=p)
        
        img_out_path = f"artifacts/renders/render_{i+1}.png"
        rel_img_path = f"renders/render_{i+1}.png"
        
        if render_res.get("success") and render_res.get("image_path"):
            try:
                import shutil
                shutil.copy(render_res["image_path"], img_out_path)
                print(f"✅ RENDER SUCCESS: Saved to {img_out_path} ({render_res.get('render_ms', 0)}ms)")
                status = "SUCCESS"
            except Exception as e:
                status = f"FILE_SAVE_ERROR: {e}"
                rel_img_path = None
        else:
            status = render_res.get("error_classification", "FAIL")
            print(f"❌ RENDER NOTICE: [{status}] {render_res.get('runtime_error')}")
            rel_img_path = None

        results.append({
            "prompt": p,
            "code": code,
            "rel_image_path": rel_img_path,
            "render_ms": render_res.get("render_ms", 0),
            "seed": 42 + i,
            "status": status,
            "error": render_res.get("runtime_error")
        })

    # Generate visual gallery HTML
    build_gallery_html(results, "artifacts/gallery.html")
    print("\n🎉 Pipeline run complete! View results with: open artifacts/gallery.html")

if __name__ == "__main__":
    main()
