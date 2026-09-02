"""Unit tests for Phase 1-9 Master Plan upgrades:
- Laplacian edge variance & diagnostic critiques
- Small/corrupted image resilience
- Diagnostic scorecard formatting & error boundaries
- Hardware resource saturation (--max) across simulated devices
- Temperature annealing schedule
- Live HTML dashboard generator & XSS escaping
- Batch rendering client
"""
import os
import tempfile
import pytest
import numpy as np
from PIL import Image

from paint_rl.rewards.aesthetic import calculate_visual_richness, calculate_brush_utilization
from paint_rl.rewards.composer import RewardComposer
from paint_rl.rewards.components import (
    CompileRewardComponent,
    VisualRichnessRewardComponent,
    BrushUtilizationRewardComponent,
)
from paint_rl.config.core import ProjectConfig, load_config, apply_max_hardware_config
from paint_rl.trainer.grpo import PaintGRPOTrainer
from paint_rl.telemetry.dashboard import DashboardWriter
from paint_rl.renderer.manager import RendererService


class TestLaplacianAndCritiques:
    """Test Laplacian edge variance and diagnostic critique generation."""

    def test_blank_canvas_critique(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            img = Image.new("RGB", (100, 100), color=(255, 255, 255))
            img.save(path)
            res = calculate_visual_richness(path)
            assert res["is_blank"] is True
            assert res["richness_score"] == 0.0
            assert "Blank or monochromatic canvas" in res["critique"]
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_tiny_images_do_not_crash(self):
        """Verify 1x1, 2x2, and 3x3 images don't raise IndexError/NaN."""
        for size in [(1, 1), (2, 2), (3, 3), (4, 4)]:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                path = f.name
            try:
                img = Image.new("RGB", size, color=(100, 150, 200))
                img.save(path)
                res = calculate_visual_richness(path)
                assert isinstance(res, dict)
                assert "richness_score" in res
                assert "edge_variance" in res
                assert res["richness_score"] >= 0.0
            finally:
                if os.path.exists(path):
                    os.remove(path)

    def test_nonexistent_image_fails_gracefully(self):
        res = calculate_visual_richness("/nonexistent/file/path.png")
        assert res["is_blank"] is True
        assert res["richness_score"] == 0.0
        assert "[FAIL]" in res["critique"]

    def test_textured_canvas_has_edge_variance_and_good_critique(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            arr[10:90, 10:90] = [200, 100, 50]
            arr[20:40, 20:80] = [50, 150, 220]
            arr[50:80, 30:60] = [220, 20, 120]
            noise = np.random.randint(0, 30, (100, 100, 3), dtype=np.uint8)
            arr = np.clip(arr + noise, 0, 255)
            
            img = Image.fromarray(arr)
            img.save(path)
            
            res = calculate_visual_richness(path)
            assert res["is_blank"] is False
            assert res["richness_score"] > 0.4
            assert res["edge_variance"] > 0.0
            assert "[GOOD]" in res["critique"] or "[EXCELLENT]" in res["critique"]
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_brush_utilization_critique_good(self):
        code = """
        function setup() {
            createCanvas(600, 600, WEBGL);
            brush.scaleBrushes(3);
            brush.load();
            noLoop();
            brush.fill("#3a5a40", 80);
            brush.rect(-100, -100, 200, 200);
            for (let i = 0; i < 10; i++) {
                brush.set("HB", "#588157", 2);
                brush.line(i * 10, -50, i * 10, 50);
            }
        }
        """
        res = calculate_brush_utilization(code)
        assert res["has_cheat"] is False
        assert res["brush_score"] >= 0.7
        assert "[GOOD] brush.scaleBrushes()" in res["critique"]
        assert "[GOOD] Natural media fills" in res["critique"]

    def test_brush_utilization_critique_missing_scale(self):
        code = """
        function setup() {
            createCanvas(600, 600, WEBGL);
            noLoop();
            rect(-100, -100, 200, 200);
        }
        """
        res = calculate_brush_utilization(code)
        assert res["has_cheat"] is False
        assert "[CRITIQUE] Missing brush.scaleBrushes()" in res["critique"]

    def test_brush_utilization_anti_cheat_text(self):
        code = """
        function setup() {
            createCanvas(600, 600, WEBGL);
            textSize(32);
            text("Sunflower art", 0, 0);
        }
        """
        res = calculate_brush_utilization(code)
        assert res["has_cheat"] is True
        assert res["brush_score"] == 0.0
        assert res["cheat_reason"] == "TEXT_IN_CANVAS_HACK"


class TestDiagnosticScorecard:
    """Test DiagnosticScorecard generation in RewardComposer."""

    def test_scorecard_formatting_valid(self):
        composer = RewardComposer([
            CompileRewardComponent(weight=0.20),
            VisualRichnessRewardComponent(weight=0.40),
            BrushUtilizationRewardComponent(weight=0.40),
        ])
        
        render_res = {"success": True}
        code = """
        function setup() {
            createCanvas(600, 600, WEBGL);
            brush.scaleBrushes(3);
            brush.fill("#123456", 50);
            brush.rect(0, 0, 100, 100);
        }
        """
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            path = f.name
        try:
            arr = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
            Image.fromarray(arr).save(path)
            
            comp_res = composer.compute(
                render_result=render_res,
                image_path=path,
                prompt="Watercolor forest",
                code=code
            )
            scorecard = composer.generate_scorecard(comp_res, prompt="Watercolor forest")
            
            assert "DIAGNOSTIC SCORECARD" in scorecard
            assert "COMPILE" in scorecard
            assert "VISUAL_RICHNESS" in scorecard
            assert "BRUSH_UTILIZATION" in scorecard
            assert "TOTAL REWARD" in scorecard
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_scorecard_formatting_on_failed_render(self):
        composer = RewardComposer([
            CompileRewardComponent(weight=0.20),
            VisualRichnessRewardComponent(weight=0.40),
        ])
        
        render_res = {"success": False, "error_classification": "SYNTAX_ERROR"}
        comp_res = composer.compute(
            render_result=render_res,
            image_path=None,
            prompt="Broken art",
            code="function setup() {"
        )
        scorecard = composer.generate_scorecard(comp_res, prompt="Broken art")
        
        assert "DIAGNOSTIC SCORECARD" in scorecard
        assert "[FAIL]" in scorecard
        assert "TOTAL REWARD: 0.0000" in scorecard


class TestHardwareMaxSaturation:
    """Test --max hardware saturation scaling."""

    def test_apply_max_hardware_config(self):
        config, _, _ = load_config("local")
        initial_threads = os.cpu_count() or 4
        adjustments = apply_max_hardware_config(config)
        
        assert "torch_threads" in adjustments
        assert adjustments["torch_threads"] == initial_threads
        assert "ram_total_gb" in adjustments
        assert "ram_available_gb" in adjustments

    def test_apply_max_simulated_mps(self):
        config, _, _ = load_config("local")
        config.device.type = "mps"
        adjustments = apply_max_hardware_config(config)
        assert "mps_group_size" in adjustments
        assert "mps_max_new_tokens" in adjustments
        assert config.training.group_size >= 2

    def test_apply_max_simulated_cuda(self):
        config, _, _ = load_config("local")
        config.device.type = "cuda"
        adjustments = apply_max_hardware_config(config)
        assert "cuda_group_size" in adjustments
        assert config.training.group_size >= 4


class TestTemperatureAnnealing:
    """Test exponential temperature annealing schedule."""

    def test_temperature_decay(self):
        t_0 = PaintGRPOTrainer.compute_temperature(0, t_max=0.85, t_min=0.55, tau=100)
        t_50 = PaintGRPOTrainer.compute_temperature(50, t_max=0.85, t_min=0.55, tau=100)
        t_200 = PaintGRPOTrainer.compute_temperature(200, t_max=0.85, t_min=0.55, tau=100)
        t_1000 = PaintGRPOTrainer.compute_temperature(1000, t_max=0.85, t_min=0.55, tau=100)
        
        assert abs(t_0 - 0.85) < 1e-4
        assert t_0 > t_50 > t_200 > t_1000
        assert abs(t_1000 - 0.55) < 0.01  # Approaches t_min asymptotically


class TestDashboardWriter:
    """Test live HTML dashboard generator and XSS escaping."""

    def test_dashboard_file_creation_and_xss_protection(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dash_path = os.path.join(tmpdir, "dashboard.html")
            writer = DashboardWriter(dash_path)
            
            history = [
                {"cycle": 1, "steps_done": 25, "loss": 0.35, "temperature": 0.85},
                {"cycle": 2, "steps_done": 50, "loss": 0.22, "temperature": 0.72},
            ]
            
            # Inject malicious script in prompt & code to test XSS escaping
            malicious_prompt = '<script>alert("xss")</script> & Sunflower'
            malicious_code = 'function setup() { let x = "<img src=x onerror=alert(1)>"; }'
            malicious_scorecard = '[GOOD] <evil_tag> TOTAL: 0.99'
            
            writer.add_sample(
                prompt=malicious_prompt,
                code=malicious_code,
                scorecard=malicious_scorecard,
                reward=0.99,
                step=50
            )
            
            writer.update(history)
            
            assert os.path.exists(dash_path)
            with open(dash_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Must contain escaped HTML entities
            assert '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;' in content or '&lt;script&gt;alert' in content
            assert '&lt;img src=x' in content or '&lt;img' in content
            assert '&lt;evil_tag&gt;' in content
            # Must NOT contain raw unescaped script tags from user inputs
            assert '<script>alert("xss")</script>' not in content
            assert '<img src=x onerror=alert(1)>' not in content
