"""Tests for code extractor, authoritative prompting, dataset ChatML structure, and renderer reliability."""
import os
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from paint_rl.utils.code_extractor import robust_extract_js_code
from paint_rl.config.prompts import SYSTEM_PROMPT


class TestCodeExtractor:
    def test_closed_markdown_fence(self):
        text = "Here is the code:\n```javascript\nfunction setup() { createCanvas(600, 600, WEBGL); }\nfunction draw() {}\n```\nHope you like it!"
        extracted = robust_extract_js_code(text)
        assert extracted == "function setup() { createCanvas(600, 600, WEBGL); }\nfunction draw() {}"

    def test_unclosed_markdown_fence(self):
        # Truncated completion missing closing ```
        text = "```javascript\nfunction setup() {\n    createCanvas(600, 600, WEBGL);\n    background(245, 243, 238);\n    brush.load();"
        extracted = robust_extract_js_code(text)
        assert "```" not in extracted
        assert "function setup()" in extracted
        assert "brush.load();" in extracted

    def test_thinking_tag_removal(self):
        text = "<think>\nLet's write a p5.js sketch with watercolor brush.\n</think>\n```javascript\nfunction setup() { createCanvas(600, 600, WEBGL); }\n```"
        extracted = robust_extract_js_code(text)
        assert "<think>" not in extracted
        assert extracted == "function setup() { createCanvas(600, 600, WEBGL); }"

    def test_raw_function_setup_fallback(self):
        text = "Sure! Here is your generative art:\n\nfunction setup() {\n    createCanvas(600, 600, WEBGL);\n}\nfunction draw() {\n    brush.circle(0, 0, 50);\n}\n\nThis creates a circle."
        extracted = robust_extract_js_code(text)
        assert extracted.startswith("function setup()")
        assert "createCanvas" in extracted

    def test_empty_input(self):
        assert robust_extract_js_code("") == ""
        assert robust_extract_js_code(None) == ""


class TestPromptAndDatasetPipeline:
    def test_system_prompt_rules(self):
        assert "createCanvas(600, 600, WEBGL)" in SYSTEM_PROMPT
        assert "brush.load()" in SYSTEM_PROMPT
        assert "brush.scaleBrushes(3)" in SYSTEM_PROMPT
        assert "push()" in SYSTEM_PROMPT
        assert "brush.pushMatrix" in SYSTEM_PROMPT  # Checked as a negative constraint in rules

    def test_grpo_load_dataset_conversational(self):
        from paint_rl.trainer.grpo import PaintGRPOTrainer
        trainer = PaintGRPOTrainer()
        dataset = trainer.load_dataset(split="train")
        assert len(dataset) > 0
        first_sample = dataset[0]
        assert "prompt" in first_sample
        assert isinstance(first_sample["prompt"], list)
        assert len(first_sample["prompt"]) == 2
        assert first_sample["prompt"][0]["role"] == "system"
        assert first_sample["prompt"][1]["role"] == "user"
        assert "p5.js" in first_sample["prompt"][1]["content"]


class TestRendererHardening:
    def test_renderer_session_initialized(self):
        from paint_rl.renderer.manager import RendererService
        renderer = RendererService(port=3000)
        assert hasattr(renderer, "_session")
        assert renderer.base_url == "http://127.0.0.1:3000"
