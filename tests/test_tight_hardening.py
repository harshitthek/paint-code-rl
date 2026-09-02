"""Tight Hardening & Adversarial Stress Tests for Paint-Code-RL.

Covers:
1. Zero-Cost & Security Gatekeeper Invariants (Zero Paid API leaks)
2. Numerical Reward Hardening & Boundary Defense (NaN, Inf, Type safety)
3. Adversarial Prompt Injection & Code Extraction Stress Tests
4. Multi-Hardware Saturation & VRAM Boundary Tests
5. Renderer Process Fault Tolerance & Safe Recovery
"""
import os
import sys
import math
import pytest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from paint_rl.rewards.validation import validate_reward_bundle, RewardValidationError
from paint_rl.rewards.composer import RewardComposer
from paint_rl.rewards.components import RewardComponent, RewardResult
from paint_rl.utils.code_extractor import robust_extract_js_code
from paint_rl.config.core import load_config, apply_max_hardware_config, ConfigurationError
from paint_rl.models.registry import ModelRegistry


# ============================================================
# 1. Zero-Cost & Security Gatekeeper Invariants
# ============================================================

class TestZeroCostAndSecurityGatekeeper:
    """Enforce the Zero-Cost Invariant: never silently leak external API calls."""

    def test_paid_api_blocked_in_free_mode(self):
        """In FREE mode, selecting paid providers must strictly raise or fail closed."""
        caps = {"cuda_available": True, "gpu_count": 2, "vram_gb": [16.0, 16.0]}
        sel = ModelRegistry.select_models("FREE", caps, allow_paid_api=False)
        assert sel["policy_device"] == "cuda:0"
        # Must not designate external API endpoints
        assert "openai" not in sel.get("judge_provider", "").lower()
        assert "anthropic" not in sel.get("judge_provider", "").lower()

    def test_openai_api_key_env_does_not_trigger_paid_in_local_mode(self, monkeypatch):
        """Presence of OPENAI_API_KEY must not bypass the FREE/LOCAL safety invariant."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-mock-fake-key-that-must-never-be-called")
        config, _, _ = load_config("local")
        assert config.safety.allow_external_apis is False


# ============================================================
# 2. Numerical Reward Hardening & Boundary Defense
# ============================================================

class TestNumericalRewardHardening:
    """Verify that rewards never propagate NaN, Inf, or invalid data types."""

    @pytest.mark.parametrize("bad_val", [
        math.nan, float("nan"),
        float("inf"), float("-inf"),
        "0.85", [0.85], {"score": 0.85}, None, True, False
    ])
    def test_validate_reward_bundle_rejects_corrupted_types(self, bad_val):
        """Must reject any non-finite float or malformed data type."""
        with pytest.raises(RewardValidationError):
            validate_reward_bundle({"total": bad_val})

    def test_validate_reward_bundle_accepts_valid_floats(self):
        """Valid floats within reasonable bounds must pass."""
        for val in [0.0, 0.0001, 0.5, 1.0, 1.99, 1e-6]:
            bundle = validate_reward_bundle({"total": val, "component_a": val})
            assert bundle["total"] == val

    def test_composer_fail_closed_on_exploding_component(self):
        """If a component throws an exception, composer must not raise or produce NaN."""
        class ExplodingComponent(RewardComponent):
            @property
            def name(self): return "exploding"
            @property
            def version(self): return "v1.0"
            def compute(self, **kwargs):
                raise ZeroDivisionError("Hardware division by zero")

        composer = RewardComposer([
            (ExplodingComponent(), 0.50)
        ])
        res = composer.compute(render_result={"success": True}, code="function setup() {}")
        assert not math.isnan(res["total"])
        assert not math.isinf(res["total"])
        assert res["total"] == 0.0
        assert "COMPONENT_ERROR" in res["components"][0].error_state


# ============================================================
# 3. Adversarial Prompt Injection & Code Extraction
# ============================================================

class TestAdversarialCodeExtraction:
    """Stress test code extraction against adversarial inputs and malformed LLM outputs."""

    def test_prompt_injection_fence_handling(self):
        """When LLM outputs multiple markdown fences, extract the valid p5.js sketch."""
        adversarial_text = """
Here is an example:
```javascript
// Malicious or decoy snippet
const decoy = 1;
```
Now here is the real generative artwork code:
```javascript
function setup() {
    createCanvas(600, 600, WEBGL);
    background(240);
    brush.scaleBrushes(3);
}
function draw() {
    brush.line(0, 0, 100, 100);
}
```
Hope you like this artwork!
"""
        extracted = robust_extract_js_code(adversarial_text)
        assert "createCanvas" in extracted
        assert "WEBGL" in extracted
        assert "brush.scaleBrushes" in extracted

    def test_thinking_tag_with_backticks(self):
        """Code extractor must strip deep reasoning tags even if they contain backticks."""
        raw = """<think>
Let's design a p5.js sketch:
```javascript
let unclosed = 1;
```
I should use WEBGL.
</think>
```javascript
function setup() {
    createCanvas(600, 600, WEBGL);
    background(255);
}
```"""
        code = robust_extract_js_code(raw)
        assert "<think>" not in code
        assert "unclosed" not in code
        assert "createCanvas(600, 600, WEBGL)" in code

    def test_code_with_nested_backticks_in_strings(self):
        """Handle JavaScript code that itself contains backtick template literals."""
        raw = """```javascript
function setup() {
    createCanvas(600, 600, WEBGL);
    let title = `generative artwork #1`;
    background(200);
}
```"""
        code = robust_extract_js_code(raw)
        assert "let title = `generative artwork #1`" in code

    def test_unfenced_raw_setup_function(self):
        """If model forgets markdown fences completely, extract raw function setup."""
        raw = """Here is the code directly:
function setup() {
    createCanvas(600, 600, WEBGL);
    background(245, 243, 238);
}
function draw() {
    circle(0, 0, 50);
}"""
        code = robust_extract_js_code(raw)
        assert "function setup()" in code
        assert "function draw()" in code


# ============================================================
# 4. Multi-Hardware Saturation & VRAM Boundary Tests
# ============================================================

class TestHardwareSaturationBoundaries:
    """Verify hardware auto-tuning across simulated VRAM configurations."""

    def test_low_vram_t4_enforces_1_5b_model(self):
        """On a 15.6GB Tesla T4, 7B would OOM; it must select 1.5B."""
        import torch
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        trainer.device = torch.device("cuda")
        
        # Simulate CUDA with 15.6 GB VRAM
        mock_props = MagicMock()
        mock_props.total_memory = 15.6 * 1e9
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            model_id = trainer._resolve_model_id()
            assert "1.5B" in model_id

    def test_high_vram_a100_allows_7b_model(self):
        """On an 80GB A100, 7B should be selected."""
        import torch
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        trainer.device = torch.device("cuda")
        
        mock_props = MagicMock()
        mock_props.total_memory = 80.0 * 1e9
        
        with patch("torch.cuda.is_available", return_value=True), \
             patch("torch.cuda.get_device_properties", return_value=mock_props):
            model_id = trainer._resolve_model_id()
            assert "7B" in model_id

    def test_apply_max_hardware_scales_threads_and_groups(self):
        """--max must scale group_size without exceeding safe ceilings."""
        config, _, _ = load_config("local")
        config.device.type = "cuda"
        adjustments = apply_max_hardware_config(config)
        assert adjustments["cuda_group_size"] >= 4
        assert config.training.group_size >= 4


# ============================================================
# 5. Renderer Process Fault Tolerance
# ============================================================

class TestRendererFaultTolerance:
    """Verify renderer service handles dead ports and bad responses gracefully."""

    def test_health_check_on_dead_port_returns_false(self):
        """Probing an unused port must immediately return False without raising."""
        from paint_rl.renderer.manager import RendererService
        dead_service = RendererService(port=59999, timeout=1.0)
        assert dead_service.is_healthy() is False

    def test_render_handles_dead_service_gracefully(self):
        """Rendering when service is down returns explicit error classification."""
        from paint_rl.renderer.manager import RendererService
        dead_service = RendererService(port=59999, timeout=1.0)
        # Mock ensure_started to fail
        with patch.object(dead_service, "ensure_started", return_value=False):
            res = dead_service.render("function setup() {}")
            assert res["success"] is False
            assert res["error_classification"] == "RENDERER_UNAVAILABLE"
