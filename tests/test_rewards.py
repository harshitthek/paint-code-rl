"""Tests for reward pipeline, config, cache, checkpoint, model registry.

These are REAL tests against real module implementations — not mocks.
"""
import sys
import os
import math
import json
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


# ============================================================
# Configuration Tests
# ============================================================

class TestConfig:
    def test_load_base_config(self):
        from paint_rl.config.core import load_config
        config, config_json, config_hash = load_config("local")
        assert config.model.id == "Qwen/Qwen2.5-Coder-7B-Instruct"
        assert config.storage.base_path == "artifacts"
        assert config.reward.weights.compile == 0.10
        assert config.reward.weights.hpsv3 == 0.30
        assert config.reward.weights.pairwise == 0.60

    def test_config_has_safety(self):
        from paint_rl.config.core import load_config
        config, _, _ = load_config("local")
        assert hasattr(config, "safety")
        assert config.safety.allow_external_apis is False

    def test_config_has_aesthetic(self):
        from paint_rl.config.core import load_config
        config, _, _ = load_config("local")
        assert hasattr(config, "aesthetic")
        assert config.aesthetic.provider == "clip"

    def test_config_hash_deterministic(self):
        from paint_rl.config.core import load_config
        _, _, hash1 = load_config("local")
        _, _, hash2 = load_config("local")
        assert hash1 == hash2

    def test_config_extra_keys_ignored(self):
        """Config with extra unknown keys should not crash."""
        from paint_rl.config.core import load_config
        # local.yaml has safety.allow_external_apis which wasn't in old schema
        config, _, _ = load_config("local")
        assert config is not None

    def test_kaggle_config_overlay(self):
        from paint_rl.config.core import load_config
        config, _, _ = load_config("kaggle")
        assert config.storage.base_path == "/kaggle/working/artifacts"


# ============================================================
# Reward Validation Tests
# ============================================================

class TestRewardValidation:
    def test_valid_bundle(self):
        from paint_rl.rewards.validation import validate_reward_bundle
        bundle = {"total": 1.5}
        result = validate_reward_bundle(bundle)
        assert result["total"] == 1.5

    def test_nan_rejected(self):
        from paint_rl.rewards.validation import validate_reward_bundle, RewardValidationError
        with pytest.raises(RewardValidationError, match="NaN"):
            validate_reward_bundle({"total": math.nan})

    def test_inf_rejected(self):
        from paint_rl.rewards.validation import validate_reward_bundle, RewardValidationError
        with pytest.raises(RewardValidationError, match="Infinity"):
            validate_reward_bundle({"total": math.inf})

    def test_none_rejected(self):
        from paint_rl.rewards.validation import validate_reward_bundle, RewardValidationError
        with pytest.raises(RewardValidationError, match="None"):
            validate_reward_bundle({"total": None})

    def test_missing_total_rejected(self):
        from paint_rl.rewards.validation import validate_reward_bundle, RewardValidationError
        with pytest.raises(RewardValidationError, match="total"):
            validate_reward_bundle({})


# ============================================================
# Reward Components Tests
# ============================================================

class TestRewardComponents:
    def test_compile_reward_success(self):
        from paint_rl.rewards.components import CompileRewardComponent
        comp = CompileRewardComponent(weight=0.10)
        result = comp.compute(render_result={"success": True, "image_path": "/tmp/test.png"})
        assert result.raw_score == 1.0
        assert result.weighted_score == pytest.approx(0.10)
        assert result.error_state is None

    def test_compile_reward_failure(self):
        from paint_rl.rewards.components import CompileRewardComponent
        comp = CompileRewardComponent(weight=0.10)
        result = comp.compute(render_result={"success": False, "error_classification": "PARSE_ERROR"})
        assert result.raw_score == 0.0
        assert result.weighted_score == 0.0
        assert result.error_state == "PARSE_ERROR"

    def test_reward_result_fields(self):
        from paint_rl.rewards.components import RewardResult
        r = RewardResult(raw_score=0.8, weight=0.3, weighted_score=0.24,
                         component_name="test", latency_ms=10.5)
        assert r.component_name == "test"
        assert r.latency_ms == 10.5
        assert r.error_state is None


# ============================================================
# Reward Composer Tests
# ============================================================

class TestRewardComposer:
    def test_composer_with_compile_only(self):
        from paint_rl.rewards.components import CompileRewardComponent
        from paint_rl.rewards.composer import RewardComposer
        composer = RewardComposer(components=[CompileRewardComponent(weight=0.10)])
        bundle = composer.compute(
            render_result={"success": True, "image_path": "/tmp/test.png"},
            prompt="test"
        )
        assert bundle["total"] == pytest.approx(0.10)
        assert len(bundle["components"]) == 1

    def test_composer_skips_downstream_on_render_failure(self):
        from paint_rl.rewards.components import CompileRewardComponent, AestheticRewardComponent
        from paint_rl.rewards.composer import RewardComposer

        # Mock a scorer that should NOT be called
        class FailScorer:
            def score(self, image_path, prompt):
                raise RuntimeError("Should not be called on render failure")

        composer = RewardComposer(components=[
            CompileRewardComponent(weight=0.10),
            AestheticRewardComponent(weight=0.30, scorer=FailScorer()),
        ])
        bundle = composer.compute(
            render_result={"success": False, "error_classification": "TIMEOUT"},
            prompt="test"
        )
        # Compile = 0 (failed), aesthetic = 0 (skipped)
        assert bundle["total"] == 0.0
        assert bundle["components"][1].error_state == "SKIPPED_RENDER_FAILED"


# ============================================================
# Cache Tests
# ============================================================

class TestCache:
    def test_cache_operations(self):
        from paint_rl.storage.cache import CacheManager
        db_path = "artifacts/test_cache_unit.db"
        if os.path.exists(db_path):
            os.remove(db_path)

        cache = CacheManager(db_path)
        key = cache.generate_vlm_key("c_hash", "r_hash", "draw a circle", "gpt-4o-mini", "v1")

        # Miss
        assert cache.get_vlm(key) is None

        # Set
        cache.set_vlm(key, "left", {"raw": True}, 1.5, "gpt-4o-mini")

        # Hit
        res = cache.get_vlm(key)
        assert res is not None
        assert res["decision"] == "left"
        assert res["cache_hit"] is True

        # Cleanup — close connection before deleting on Windows
        try:
            cache.conn.close()
        except Exception:
            pass
        try:
            os.remove(db_path)
        except PermissionError:
            pass  # Windows may still hold the file


# ============================================================
# Model Registry Tests
# ============================================================

class TestModelRegistry:
    def test_cuda_selection(self):
        from paint_rl.models.registry import ModelRegistry
        caps = {"cuda_available": True, "gpu_count": 2, "vram_gb": [16.0, 16.0]}
        sel = ModelRegistry.select_models("FREE", caps, allow_paid_api=False)
        assert sel["policy_device"] == "cuda:0"

    def test_cpu_fallback(self):
        from paint_rl.models.registry import ModelRegistry
        caps = {"cuda_available": False, "gpu_count": 0, "vram_gb": []}
        sel = ModelRegistry.select_models("FREE", caps, allow_paid_api=False)
        assert sel["policy_device"] == "cpu"


# ============================================================
# Async Rollout Tests
# ============================================================

class TestAsyncRollout:
    def test_worker_crash_handling(self):
        from paint_rl.trainer.async_rollout import RolloutEngine
        engine = RolloutEngine(max_workers=2)

        def dummy_reward(p, c, r, s):
            if s == 0:
                raise ValueError("Intentional crash")
            return {"total": 1.0}

        results = engine.generate_and_evaluate(
            prompts=["a", "b"], codes=["c", "d"],
            reference_paths=["x", "y"], seeds=[0, 1],
            reward_fn=dummy_reward
        )
        assert results[0]["error_class"] == "WORKER_CRASH"
        assert results[1]["total"] == 1.0


# ============================================================
# Judge Provider Tests
# ============================================================

class TestJudgeProviders:
    def test_mock_judge(self):
        from paint_rl.judges.providers import MockJudgeProvider
        judge = MockJudgeProvider()
        result = judge.compare("a.png", "b.png", "test prompt")
        assert result["status"] == "mock_tie"
        assert result["score"] == 0.5

    def test_factory_returns_local_by_default(self):
        from paint_rl.judges.providers import create_judge_provider
        from paint_rl.config.core import ACTIVE_CONFIG
        # When ACTIVE_CONFIG is loaded (base.yaml has judge.provider=local),
        # factory should return LocalVLMProvider
        if ACTIVE_CONFIG is not None:
            judge = create_judge_provider()
            assert judge.provider_name == "local_vlm"
        else:
            # If config not loaded, returns mock
            judge = create_judge_provider(config=None)
            assert judge.provider_name == "mock"

    def test_openai_blocked_in_local_mode(self):
        from paint_rl.judges.providers import create_judge_provider
        from paint_rl.config.core import load_config
        config, _, _ = load_config("local")
        # Base config now has safety.allow_external_apis=false and judge.provider=local
        # Trying to create OpenAI provider should fail
        if hasattr(config, "safety") and not config.safety.allow_external_apis:
            config.judge.provider = "openai"
            with pytest.raises(ValueError, match="ConfigurationError"):
                create_judge_provider(config)


# ============================================================
# Dataset Tests
# ============================================================

class TestDataset:
    def test_prompts_v1_exists(self):
        assert os.path.exists("datasets/prompts_v1.jsonl")

    def test_prompts_v1_valid_jsonl(self):
        with open("datasets/prompts_v1.jsonl") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 40  # At least 40 prompts

        for line in lines:
            data = json.loads(line)
            assert "prompt_id" in data
            assert "prompt" in data
            assert "category" in data
            assert len(data["prompt"]) > 10

    def test_validation_set_exists(self):
        assert os.path.exists("datasets/validation.jsonl")
        with open("datasets/validation.jsonl") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 5

    def test_test_set_exists(self):
        assert os.path.exists("datasets/test.jsonl")
        with open("datasets/test.jsonl") as f:
            lines = [l for l in f if l.strip()]
        assert len(lines) >= 3

    def test_no_duplicate_prompt_ids(self):
        ids = set()
        with open("datasets/prompts_v1.jsonl") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    pid = data["prompt_id"]
                    assert pid not in ids, f"Duplicate prompt_id: {pid}"
                    ids.add(pid)


# ============================================================
# Telemetry Tests
# ============================================================

class TestTelemetry:
    def test_experiment_logger_creates_run_id(self):
        from paint_rl.telemetry.core import ExperimentLogger
        logger = ExperimentLogger(config_hash="test_hash")
        assert logger.run_id.startswith("run_")
        assert logger.config_hash == "test_hash"


# ============================================================
# Multi-Signal Visual & Anti-Cheat Reward Tests
# ============================================================

class TestMultiSignalVisualRewards:
    @pytest.fixture
    def blank_image(self, tmp_path):
        from PIL import Image
        img_path = str(tmp_path / "blank.png")
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        img.save(img_path)
        return img_path

    @pytest.fixture
    def colorful_image(self, tmp_path):
        from PIL import Image, ImageDraw
        img_path = str(tmp_path / "colorful.png")
        img = Image.new("RGB", (200, 200), color=(240, 240, 235))
        draw = ImageDraw.Draw(img)
        # Draw colorful elements
        draw.rectangle([20, 20, 100, 100], fill=(220, 50, 50))
        draw.ellipse([80, 80, 180, 180], fill=(30, 120, 200))
        draw.line([10, 190, 190, 10], fill=(40, 180, 60), width=5)
        img.save(img_path)
        return img_path

    def test_blank_canvas_rejected(self, blank_image):
        from paint_rl.rewards.aesthetic import calculate_visual_richness
        res = calculate_visual_richness(blank_image)
        assert res["is_blank"] is True
        assert res["richness_score"] == 0.0

    def test_colorful_canvas_richness(self, colorful_image):
        from paint_rl.rewards.aesthetic import calculate_visual_richness
        res = calculate_visual_richness(colorful_image)
        assert res["is_blank"] is False
        assert res["richness_score"] > 0.4
        assert res["color_count"] > 1

    def test_brush_utilization_clean_code(self):
        from paint_rl.rewards.aesthetic import calculate_brush_utilization
        code = """
        function setup() {
            createCanvas(600, 600, WEBGL);
            background(245, 243, 238);
            brush.load();
            brush.scaleBrushes(3);
            noLoop();
        }
        function draw() {
            translate(-width/2, -height/2);
            brush.fill('#1a759f', 160);
            brush.fillBleed(0.3, 'out');
            brush.rect(100, 100, 400, 400);
            brush.set('charcoal', '#3a6073', 2);
            brush.line(50, 300, 550, 300);
        }
        """
        res = calculate_brush_utilization(code)
        assert res["has_cheat"] is False
        assert res["brush_score"] > 0.7
        assert "scaleBrushes" in res["features_used"]
        assert "watercolor_fill" in res["features_used"]

    def test_text_in_canvas_cheat_detected(self):
        from paint_rl.rewards.aesthetic import calculate_brush_utilization
        cheat_code = """
        function setup() {
            createCanvas(600, 600, WEBGL);
            background(255);
            textSize(32);
            text("a field of wildflowers", 100, 100);
        }
        """
        res = calculate_brush_utilization(cheat_code)
        assert res["has_cheat"] is True
        assert res["cheat_reason"] == "TEXT_IN_CANVAS_HACK"
        assert res["brush_score"] == 0.0

    def test_visual_richness_reward_component(self, colorful_image):
        from paint_rl.rewards.components import VisualRichnessRewardComponent
        comp = VisualRichnessRewardComponent(weight=0.25)
        res = comp.compute(image_path=colorful_image)
        assert res.raw_score > 0.4
        assert res.weighted_score > 0.1
        assert res.component_name == "visual_richness"

    def test_brush_utilization_reward_component(self):
        from paint_rl.rewards.components import BrushUtilizationRewardComponent
        comp = BrushUtilizationRewardComponent(weight=0.15)
        code = "function setup() { createCanvas(600, 600, WEBGL); brush.load(); brush.scaleBrushes(3); noLoop(); } function draw() { brush.fill('#ff0000', 100); brush.rect(50, 50, 100, 100); }"
        res = comp.compute(code=code)
        assert res.raw_score > 0.5
        assert res.weighted_score > 0.05
        assert res.component_name == "brush_utilization"

    def test_full_composite_reward_composer(self, colorful_image):
        from paint_rl.rewards.components import (
            CompileRewardComponent,
            VisualRichnessRewardComponent,
            BrushUtilizationRewardComponent,
        )
        from paint_rl.rewards.composer import RewardComposer

        composer = RewardComposer([
            CompileRewardComponent(weight=0.20),
            VisualRichnessRewardComponent(weight=0.50),
            BrushUtilizationRewardComponent(weight=0.30),
        ])

        code = "function setup() { createCanvas(600, 600, WEBGL); brush.load(); brush.scaleBrushes(3); noLoop(); } function draw() { brush.fill('#ff0000', 100); brush.rect(50, 50, 100, 100); }"
        bundle = composer.compute(
            render_result={"success": True, "image_path": colorful_image},
            image_path=colorful_image,
            code=code,
            prompt="draw colorful art"
        )

        assert bundle["total"] > 0.5
        assert len(bundle["components"]) == 3
        assert bundle["error_class"] is None

