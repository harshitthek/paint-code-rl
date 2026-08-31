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
    # Look for `javascript or `js blocks
    match = re.search(r'`(?:javascript|js)?\s*\n([\s\S]*?)`', raw_text)
    if match:
        return match.group(1).strip()
    return raw_text.strip()

def build_gallery_html(renders: list, output_path: str):
    cards_html = ""
    for r in renders:
        img_tag = f'<img src=\"{r["rel_image_path"]}\" alt=\"Render\" class=\"render-img\" />' if r.get("rel_image_path") else f'<div class=\"error-box\">Render Failed: {r.get("error", "Unknown error")}</div>'
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
        .img-container {{ display: flex; align-items: center; justify-content: center; background: #020617; border-radius: 8px; overflow: hidden; min-height: 300px; }}
        .render-img {{ max-width: 100%; max-height: 400px; object-fit: contain; }}
        .code-container {{ background: #090d16; border-radius: 8px; padding: 1rem; overflow-x: auto; max-height: 400px; }}
        pre {{ margin: 0; font-family: 'JetBrains Mono', 'Fira Code', monospace; font-size: 0.85rem; color: #a5f3fc; }}
        .meta {{ display: flex; gap: 2rem; margin-top: 1rem; font-size: 0.85rem; color: #94a3b8; border-top: 1px solid #334155; padding-top: 0.75rem; }}
        .error-box {{ color: #f87171; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>🎨 Paint-Code-RL: Live Generative Canvas</h1>
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
    
    device = get_device()
    print(f"Device: {device}")
    
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

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device.type in ["cuda", "mps"] else torch.float32
    ).to(device)

    renderer = RendererService(port=3000)
    print("Connecting to WebGL Renderer...")
    if not renderer.ensure_started():
        print("[WARN] Could not automatically start renderer. Visual renders will be skipped.")

    prompts = [
        "Create a generative watercolor painting of a serene misty pine forest with organic brush strokes in p5.js and p5.brush.",
        "Draw a vibrant abstract floral arrangement using expressive calligraphy ink washes."
    ]

    os.makedirs("artifacts/renders", exist_ok=True)
    results = []

    for i, p in enumerate(prompts):
        print(f"\n--- Generating Artwork {i+1}/{len(prompts)}: '{p}' ---")
        messages = [
            {"role": "system", "content": "You are a master generative artist who creates beautiful digital paintings in p5.js using the p5.brush library."},
            {"role": "user", "content": f"{p}\nRespond only with executable p5.js code inside a `javascript block. Call brush.load() and setup canvas."}
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(device)
        
        outputs = model.generate(
            **inputs,
            max_new_tokens=600,
            do_sample=True,
            top_p=0.9,
            temperature=0.7
        )
        gen_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        code = extract_js_code(gen_text)
        
        print(f"Generated {len(code)} characters of p5.js code. Rendering canvas...")
        
        # Render via Node.js WebGL Sandbox
        render_res = renderer.render(code, seed=42 + i, prompt=p)
        
        img_out_path = f"artifacts/renders/render_{i+1}.png"
        rel_img_path = f"renders/render_{i+1}.png"
        
        if render_res.get("success") and render_res.get("image_path"):
            # Move or save image
            try:
                import shutil
                shutil.copy(render_res["image_path"], img_out_path)
                print(f"✅ Rendered successfully: {img_out_path} ({render_res.get('render_ms', 0)}ms)")
            except Exception:
                rel_img_path = None
        else:
            print(f"❌ Render failed: {render_res.get('error_classification')} - {render_res.get('runtime_error')}")
            rel_img_path = None

        results.append({
            "prompt": p,
            "code": code,
            "rel_image_path": rel_img_path,
            "render_ms": render_res.get("render_ms", 0),
            "seed": 42 + i,
            "error": render_res.get("runtime_error")
        })

    # Generate visual gallery HTML
    build_gallery_html(results, "artifacts/gallery.html")
    print("\n🎉 Complete pipeline execution finished!")

if __name__ == "__main__":
    main()
