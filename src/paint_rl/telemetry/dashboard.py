"""Live HTML Visual Dashboard for Paint-Code-RL training monitoring."""
import os
import json
import time
import html
from typing import List, Dict, Any, Optional


class DashboardWriter:
    """Generates and continuously updates a standalone HTML dashboard."""
    
    def __init__(self, output_path: str = "artifacts/dashboard.html"):
        self.output_path = output_path
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        self._history: List[Dict[str, Any]] = []
        self._samples: List[Dict[str, Any]] = []
        
    def add_sample(self, prompt: str, code: str, image_path: Optional[str] = None, 
                   scorecard: Optional[str] = None, reward: float = 0.0, step: int = 0):
        """Add a rendered artwork sample to the gallery."""
        self._samples.insert(0, {
            "prompt": prompt,
            "code": code,
            "image_path": image_path,
            "scorecard": scorecard,
            "reward": round(reward, 4),
            "step": step,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        })
        # Keep latest 20 samples
        self._samples = self._samples[:20]

    def update(self, metrics_history: List[Dict[str, Any]]):
        """Update dashboard with new cycle/step metrics."""
        self._history = metrics_history
        self._render_html()

    def _render_html(self):
        """Generate static standalone HTML page."""
        cycles = [m.get("cycle", idx + 1) for idx, m in enumerate(self._history)]
        losses = [m.get("loss", 0.0) for m in self._history]
        rewards = [m.get("reward", 0.0) for m in self._history]
        grad_norms = [m.get("grad_norm", 0.0) for m in self._history]
        temps = [m.get("temperature", 0.7) for m in self._history]
        steps = [m.get("steps_done", 0) for m in self._history]
        
        labels_json = json.dumps([f"Cycle {c} (Step {s})" for c, s in zip(cycles, steps)]).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        losses_json = json.dumps(losses).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        rewards_json = json.dumps(rewards).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        grad_norms_json = json.dumps(grad_norms).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        temps_json = json.dumps(temps).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
        
        gallery_items = []
        for s in self._samples:
            img_tag = ""
            if s.get("image_path") and os.path.exists(s["image_path"]):
                # Use relative path if possible
                rel_path = os.path.relpath(s["image_path"], os.path.dirname(os.path.abspath(self.output_path)))
                img_tag = f'<img src="{rel_path}" alt="Rendered Art" class="art-img" />'
            else:
                img_tag = '<div class="no-img">No Image Available</div>'
                
            scorecard_html = ""
            if s.get("scorecard"):
                escaped_card = html.escape(str(s["scorecard"]))
                scorecard_html = f'<pre class="scorecard">{escaped_card}</pre>'
                
            raw_code = s.get("code", "")
            code_snippet = (raw_code[:400] + "...") if len(raw_code) > 400 else raw_code
            escaped_code = html.escape(code_snippet)
            escaped_prompt = html.escape(str(s.get('prompt', '')))
            
            gallery_items.append(f"""
            <div class="card">
                <div class="card-header">
                    <strong>Step {s.get('step', 0)}</strong> | Reward: <span class="badge">{s.get('reward', 0.0):.3f}</span>
                    <div class="prompt-text">"{escaped_prompt}"</div>
                </div>
                <div class="card-body">
                    {img_tag}
                    {scorecard_html}
                    <details>
                        <summary>View p5.js Code</summary>
                        <pre class="code-block"><code>{escaped_code}</code></pre>
                    </details>
                </div>
            </div>
            """)

        gallery_html = "".join(gallery_items) if gallery_items else '<p class="empty-state">No artwork samples generated yet.</p>'

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="10">
    <title>Paint-Code-RL Live Training Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg-color: #0d1117;
            --card-bg: #161b22;
            --border-color: #30363d;
            --text-color: #c9d1d9;
            --accent-color: #58a6ff;
            --success-color: #238636;
            --font-mono: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace;
        }}
        body {{
            background-color: var(--bg-color);
            color: var(--text-color);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
            margin: 0;
            padding: 24px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        h1 {{
            margin: 0;
            font-size: 24px;
            color: #f0f6fc;
        }}
        .status-badge {{
            background-color: var(--success-color);
            color: white;
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 14px;
            font-weight: bold;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .metric-card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 16px;
        }}
        .metric-title {{
            font-size: 14px;
            color: #8b949e;
            margin-bottom: 8px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #f0f6fc;
        }}
        .charts-container {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .gallery-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 20px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}
        .card-header {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border-color);
            font-size: 14px;
        }}
        .prompt-text {{
            color: #8b949e;
            font-style: italic;
            margin-top: 4px;
        }}
        .badge {{
            background: #1f6feb;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
        }}
        .card-body {{
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        .art-img {{
            width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border-color);
            background: #000;
        }}
        .no-img {{
            height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            background: #21262d;
            color: #8b949e;
            border-radius: 6px;
        }}
        pre.scorecard {{
            background: #0d1117;
            padding: 8px;
            border-radius: 6px;
            font-size: 11px;
            font-family: var(--font-mono);
            white-space: pre-wrap;
            margin: 0;
            border: 1px solid #21262d;
            color: #7ee787;
        }}
        details {{
            font-size: 13px;
        }}
        summary {{
            cursor: pointer;
            color: var(--accent-color);
        }}
        pre.code-block {{
            background: #0d1117;
            padding: 8px;
            border-radius: 6px;
            font-size: 11px;
            font-family: var(--font-mono);
            overflow-x: auto;
            margin-top: 8px;
        }}
        .empty-state {{
            color: #8b949e;
            grid-column: 1 / -1;
            text-align: center;
            padding: 40px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🎨 Paint-Code-RL Training Dashboard</h1>
                <div style="font-size: 13px; color: #8b949e; margin-top: 4px;">
                    Auto-refreshing every 10s • Last update: {time.strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
            <span class="status-badge">ACTIVE</span>
        </header>

        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-title">Completed Cycles</div>
                <div class="metric-value">{len(self._history)}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Total Steps Done</div>
                <div class="metric-value">{steps[-1] if steps else 0}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Mean Reward</div>
                <div class="metric-value" style="color: #3fb950;">{rewards[-1] if rewards else 0.0:.3f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Gradient Norm</div>
                <div class="metric-value" style="color: #bc8cff;">{grad_norms[-1] if grad_norms else 0.0:.4f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-title">Current Temperature</div>
                <div class="metric-value" style="color: #f78166;">{temps[-1] if temps else '0.850'}</div>
            </div>
        </div>

        <div class="charts-container">
            <h2 style="margin-top: 0; font-size: 18px;">RL Reward, Gradient & Temperature Trajectory</h2>
            <canvas id="metricsChart" height="90"></canvas>
        </div>

        <h2>Generated Artwork Gallery</h2>
        <div class="gallery-grid">
            {gallery_html}
        </div>
    </div>

    <script>
        const ctx = document.getElementById('metricsChart').getContext('2d');
        const chart = new Chart(ctx, {{
            type: 'line',
            data: {{
                labels: {labels_json},
                datasets: [
                    {{
                        label: 'Mean Reward',
                        data: {rewards_json},
                        borderColor: '#3fb950',
                        backgroundColor: 'rgba(63, 185, 80, 0.1)',
                        tension: 0.2,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Gradient Norm',
                        data: {grad_norms_json},
                        borderColor: '#bc8cff',
                        backgroundColor: 'rgba(188, 140, 255, 0.1)',
                        tension: 0.2,
                        yAxisID: 'y'
                    }},
                    {{
                        label: 'Temperature',
                        data: {temps_json},
                        borderColor: '#f78166',
                        borderDash: [5, 5],
                        tension: 0.2,
                        yAxisID: 'y1'
                    }}
                ]
            }},
            options: {{
                responsive: true,
                interaction: {{
                    mode: 'index',
                    intersect: false,
                }},
                scales: {{
                    x: {{
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e' }}
                    }},
                    y: {{
                        type: 'linear',
                        display: true,
                        position: 'left',
                        grid: {{ color: '#21262d' }},
                        ticks: {{ color: '#8b949e' }},
                        title: {{ display: true, text: 'Reward / Grad Norm', color: '#8b949e' }}
                    }},
                    y1: {{
                        type: 'linear',
                        display: true,
                        position: 'right',
                        grid: {{ drawOnChartArea: false }},
                        ticks: {{ color: '#f78166' }},
                        title: {{ display: true, text: 'Temperature', color: '#f78166' }},
                        min: 0.4,
                        max: 1.0
                    }}
                }},
                plugins: {{
                    legend: {{
                        labels: {{ color: '#c9d1d9' }}
                    }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""
        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
