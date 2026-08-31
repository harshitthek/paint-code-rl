"""Judge providers for pairwise image comparison."""
from abc import ABC, abstractmethod
import os
import time
import json
import gc


class JudgeProvider(ABC):
    @abstractmethod
    def compare(self, candidate_path: str, reference_path: str, prompt: str) -> dict:
        """Compare two images and return {score, status, latency}."""
        ...

    @property
    @abstractmethod
    def provider_name(self) -> str: ...


class LocalVLMProvider(JudgeProvider):
    """Uses Qwen2-VL-2B-Instruct for local pairwise comparison.
    ~4.5GB FP16. Fits on 16GB Mac and T4.
    Supports sequential loading (unload policy -> load VLM -> judge -> unload).
    """

    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", device="auto"):
        self.model_id = model_id
        import torch
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        self._model = None
        self._processor = None

    def _ensure_loaded(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        except ImportError:
            raise RuntimeError(
                "transformers and qwen-vl-utils are required for LocalVLMProvider. "
                "Install: pip install transformers qwen-vl-utils torch torchvision"
            )

        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto" if self.device != "cpu" else None,
        )
        if self.device == "cpu":
            self._model = self._model.float()
        self._processor = AutoProcessor.from_pretrained(self.model_id)

    def compare(self, candidate_path: str, reference_path: str, prompt: str) -> dict:
        start_time = time.time()
        self._ensure_loaded()

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise RuntimeError("qwen-vl-utils is required. Install: pip install qwen-vl-utils")

        comparison_prompt = (
            f"You are an art judge. Compare Image 1 (left) and Image 2 (right).\n"
            f"Art prompt: \"{prompt}\"\n"
            f"Which image is more aesthetically pleasing and better matches the prompt?\n"
            f"Respond with ONLY a JSON object: {{\"decision\": \"left\"}} or {{\"decision\": \"right\"}} or {{\"decision\": \"tie\"}}"
        )

        # Direction 1: candidate=left, reference=right
        messages_lr = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(candidate_path)}"},
            {"type": "image", "image": f"file://{os.path.abspath(reference_path)}"},
            {"type": "text", "text": comparison_prompt},
        ]}]

        vote1 = self._infer(messages_lr)

        # Direction 2: reference=left, candidate=right (direction invariance)
        messages_rl = [{"role": "user", "content": [
            {"type": "image", "image": f"file://{os.path.abspath(reference_path)}"},
            {"type": "image", "image": f"file://{os.path.abspath(candidate_path)}"},
            {"type": "text", "text": comparison_prompt},
        ]}]

        vote2 = self._infer(messages_rl)

        latency = time.time() - start_time

        if "error" in (vote1, vote2):
            return {"score": 0.0, "status": "VLM_API_ERROR", "latency": latency}
        if "malformed" in (vote1, vote2):
            return {"score": 0.0, "status": "VLM_MALFORMED_RESPONSE", "latency": latency}

        # vote1: candidate is left. "left" means candidate wins.
        # vote2: candidate is right. "right" means candidate wins.
        if vote1 == "left" and vote2 == "right":
            return {"score": 1.0, "status": "win", "latency": latency}
        if vote1 == "right" and vote2 == "left":
            return {"score": 0.0, "status": "loss", "latency": latency}

        status = "explicit_tie" if (vote1 == "tie" and vote2 == "tie") else "orientation_disagreement"
        return {"score": 0.5, "status": status, "latency": latency}

    def _infer(self, messages: list) -> str:
        """Run VLM inference and parse decision."""
        import torch
        try:
            from qwen_vl_utils import process_vision_info

            text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, _ = process_vision_info(messages)

            inputs = self._processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            ).to(self._model.device)

            with torch.no_grad():
                generated_ids = self._model.generate(**inputs, max_new_tokens=64)
                trimmed = generated_ids[0][inputs.input_ids.shape[1]:]
                output = self._processor.decode(trimmed, skip_special_tokens=True).strip()

            # Parse JSON decision
            try:
                parsed = json.loads(output)
                decision = parsed.get("decision", "malformed")
                if decision in ("left", "right", "tie"):
                    return decision
                return "malformed"
            except json.JSONDecodeError:
                # Try to extract decision from free-form text
                output_lower = output.lower()
                if "left" in output_lower:
                    return "left"
                if "right" in output_lower:
                    return "right"
                if "tie" in output_lower:
                    return "tie"
                return "malformed"

        except Exception as e:
            print(f"[LocalVLMProvider] Inference error: {e}")
            return "error"

    def unload(self):
        """Free memory for sequential execution."""
        import torch
        if self._model is not None:
            del self._model
            self._model = None
        if self._processor is not None:
            del self._processor
            self._processor = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    @property
    def provider_name(self) -> str:
        return "local_vlm"


class OpenAIJudgeProvider(JudgeProvider):
    """Uses OpenAI API. Only available in PAID mode."""

    def __init__(self, model_id: str = "gpt-4o-mini"):
        from paint_rl.rewards.pairwise_vlm import OpenAIJudgeProvider as _Legacy
        self._impl = _Legacy(model_id=model_id)

    def compare(self, candidate_path: str, reference_path: str, prompt: str) -> dict:
        from paint_rl.config.core import ACTIVE_CONFIG
        if ACTIVE_CONFIG is not None and hasattr(ACTIVE_CONFIG, "safety"):
            if not ACTIVE_CONFIG.safety.allow_external_apis:
                raise ValueError(
                    "ConfigurationError: OpenAI judge is not allowed in FREE/LOCAL mode. "
                    "Set safety.allow_external_apis=true or use judge.provider=local."
                )
        return self._impl.compare(candidate_path, reference_path, prompt)

    @property
    def provider_name(self) -> str:
        return "openai"


class MockJudgeProvider(JudgeProvider):
    """Returns deterministic scores. ONLY for unit tests and DRY_RUN."""

    def compare(self, candidate_path: str, reference_path: str, prompt: str) -> dict:
        return {"score": 0.5, "status": "mock_tie", "latency": 0.0}

    @property
    def provider_name(self) -> str:
        return "mock"


def create_judge_provider(config=None) -> JudgeProvider:
    """Factory function to create the appropriate judge provider based on config."""
    if config is None:
        from paint_rl.config.core import ACTIVE_CONFIG
        config = ACTIVE_CONFIG

    if config is None:
        return MockJudgeProvider()

    provider_type = config.judge.provider

    if provider_type == "mock":
        return MockJudgeProvider()
    elif provider_type == "local":
        model_id = config.judge.model if config.judge.model != "gpt-4o-mini" else "Qwen/Qwen2-VL-2B-Instruct"
        return LocalVLMProvider(model_id=model_id)
    elif provider_type == "openai":
        if hasattr(config, "safety") and not config.safety.allow_external_apis:
            raise ValueError(
                "ConfigurationError: judge.provider=openai but safety.allow_external_apis=false. "
                "Use judge.provider=local for FREE/LOCAL mode."
            )
        return OpenAIJudgeProvider(model_id=config.judge.model)
    else:
        raise ValueError(f"Unknown judge provider: {provider_type}")
