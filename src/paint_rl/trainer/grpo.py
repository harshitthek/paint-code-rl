"""Canonical GRPO trainer for Paint-Code-RL.

Performs real end-to-end training:
generation -> render verification -> multi-signal rewards -> advantages -> loss -> backward -> optimizer step.

Zero fake counters. Zero mock updates.
Supports interactive continuous cyclic training, temperature annealing, and resource saturation.
"""
import os
import sys
import json
import torch

# Neutralize Kaggle's incompatible pre-installed torchao 0.10.0 to prevent PEFT crash
try:
    import torchao
    from packaging import version
    if version.parse(torchao.__version__) < version.parse("0.16.0"):
        import sys
        sys.modules["torchao"] = None
except Exception:
    pass

# Heavy dependencies (transformers, trl, peft, datasets) are imported lazily
# inside their respective methods to prevent 40s module import overhead on Windows/macOS.

from paint_rl.config.prompts import SYSTEM_PROMPT
from paint_rl.utils.code_extractor import robust_extract_js_code


def get_compute_device():
    """Device-agnostic compute device selector (CUDA, Apple MPS, CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Maximum safe training parameters per device type
_MPS_SAFE_LIMITS = {
    "max_batch_size": 2,
    "max_group_size": 2,
    "max_new_tokens": 320,
}

# Model selection table by device capability
_MODEL_BY_DEVICE = {
    "mps": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "cpu": "Qwen/Qwen2.5-Coder-0.5B-Instruct",
}


def _find_repo_root():
    """Find the repository root by walking up from this file."""
    curr = os.path.dirname(os.path.abspath(__file__))
    for _ in range(6):
        if os.path.exists(os.path.join(curr, "pyproject.toml")):
            return curr
        parent = os.path.dirname(curr)
        if parent == curr:
            break
        curr = parent
    return os.getcwd()


class PaintGRPOTrainer:
    """Canonical GRPO trainer for Paint-Code-RL."""

    def __init__(self, config=None, renderer_service=None):
        self.config = config
        self.device = get_compute_device()
        self.renderer = renderer_service
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._repo_root = _find_repo_root()

    def select_model_id(self):
        """Select model ID based on available hardware."""
        if self.device.type == "mps":
            return _MODEL_BY_DEVICE["mps"]
        elif self.device.type == "cuda":
            import torch
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                vram_gb = 8.0
            if vram_gb >= 20:
                return "Qwen/Qwen2.5-Coder-7B-Instruct"
            return "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        return _MODEL_BY_DEVICE["cpu"]

    def _resolve_model_id(self):
        """Resolve final model ID, enforcing device constraints."""
        config_model = self.config.model.id if self.config and hasattr(self.config, 'model') and self.config.model else None
        device_model = self.select_model_id()
        
        if self.device.type in ("mps", "cpu"):
            if config_model and config_model != device_model:
                print(f"[INFO] Config model '{config_model}' overridden to "
                      f"'{device_model}' for {self.device.type} compatibility")
            return device_model
        elif self.device.type == "cuda":
            import torch
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            except Exception:
                vram_gb = 8.0
            if vram_gb < 20.0 and config_model and config_model != device_model:
                print(f"[INFO] Config model '{config_model}' overridden to "
                      f"'{device_model}' for CUDA ({vram_gb:.1f} GB VRAM < 20 GB) compatibility")
                return device_model
        
        return config_model or device_model

    def _get_dtype(self):
        """Get appropriate dtype for the device."""
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32

    def _get_safe_batch_params(self):
        """Get memory-safe batch size and group size for current device."""
        if self.config:
            batch_size = self.config.training.batch_size
            group_size = self.config.training.group_size
        else:
            batch_size = 2
            group_size = 2
        
        if self.device.type == "mps":
            group_size = min(max(2, group_size), _MPS_SAFE_LIMITS["max_group_size"])
            batch_size = max(group_size, min(batch_size, _MPS_SAFE_LIMITS["max_batch_size"]))
            if batch_size % group_size != 0:
                batch_size = group_size
        elif self.device.type == "cpu":
            group_size = 2
            batch_size = 2
        else:
            if batch_size < group_size:
                batch_size = group_size
            elif batch_size % group_size != 0:
                batch_size = (batch_size // group_size) * group_size
        
        return batch_size, group_size

    def _get_max_new_tokens(self):
        """Get generation length, capped for memory-constrained devices."""
        tokens = self.config.generation.max_new_tokens if self.config else 256
        if self.device.type == "mps":
            tokens = min(tokens, _MPS_SAFE_LIMITS["max_new_tokens"])
        elif self.device.type == "cpu":
            tokens = min(tokens, 128)
        return tokens

    def load_model(self):
        """Load model and tokenizer, respecting device constraints and attention config."""
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        model_id = self._resolve_model_id()
        dtype = self._get_dtype()
        
        print(f"[PaintGRPOTrainer] Loading model: {model_id}")
        print(f"[PaintGRPOTrainer] Device: {self.device} | Dtype: {dtype}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        model_config = AutoConfig.from_pretrained(model_id)
        model_config.sliding_window = None
        model_config.use_sliding_window = False
        
        model_kwargs = {
            "config": model_config,
            "torch_dtype": dtype,
        }
        if self.device.type in ("cuda", "mps", "cpu"):
            model_kwargs["attn_implementation"] = "sdpa"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            **model_kwargs
        ).to(self.device)
        
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        print(f"[PaintGRPOTrainer] Model loaded: {model_id} "
              f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)")
        
        return self.model, self.tokenizer

    def build_reward_functions(self):
        """Build multi-tier reward functions for GRPO training."""
        from paint_rl.rewards.aesthetic import calculate_visual_richness, calculate_brush_utilization

        def structural_syntax_reward(prompts, completions, **kwargs):
            rewards = []
            for completion in completions:
                code = robust_extract_js_code(completion)
                brush_meta = calculate_brush_utilization(code)
                
                if brush_meta.get("has_cheat"):
                    rewards.append(0.0)
                    continue
                
                score = 0.0
                if "setup" in code and "createCanvas" in code and "WEBGL" in code:
                    score += 0.40
                elif "setup" in code and "createCanvas" in code:
                    score += 0.25
                elif "function" in code:
                    score += 0.10
                
                score += 0.60 * brush_meta.get("brush_score", 0.0)
                rewards.append(round(min(1.0, score), 3))
            return rewards

        def render_and_visual_richness_reward(prompts, completions, **kwargs):
            if not self.renderer:
                return [0.5 for _ in completions]
            
            batch_items = []
            for i, completion in enumerate(completions):
                code = robust_extract_js_code(completion)
                prompt_str = ""
                if i < len(prompts):
                    p = prompts[i]
                    if isinstance(p, list) and len(p) > 1 and isinstance(p[1], dict):
                        prompt_str = p[1].get("content", "")
                    else:
                        prompt_str = str(p)
                batch_items.append({"code": code, "seed": 42, "prompt": prompt_str})
            
            try:
                batch_results = self.renderer.render_batch(batch_items, return_base64=False)
            except Exception:
                batch_results = [{"success": False} for _ in completions]
            
            rewards = []
            for i, res in enumerate(batch_results):
                try:
                    if res.get("success") and res.get("image_path"):
                        richness_meta = calculate_visual_richness(res["image_path"])
                        if richness_meta.get("is_blank"):
                            r = 0.05
                        else:
                            richness_score = richness_meta.get("richness_score", 0.0)
                            r = round(min(1.0, 0.35 + 0.65 * richness_score), 3)
                        rewards.append(r)
                        if getattr(self, "_active_dashboard_writer", None):
                            p_txt = batch_items[i].get("prompt", "") if i < len(batch_items) else ""
                            c_txt = batch_items[i].get("code", "") if i < len(batch_items) else ""
                            self._active_dashboard_writer.add_sample(
                                prompt=p_txt,
                                code=c_txt,
                                image_path=res.get("image_path"),
                                scorecard=richness_meta.get("critique", ""),
                                reward=r,
                                step=getattr(self, "_current_step", 0)
                            )
                    else:
                        rewards.append(0.0)
                except Exception:
                    rewards.append(0.0)
            return rewards

        return [structural_syntax_reward, render_and_visual_richness_reward]

    def load_dataset(self, split="train"):
        """Load prompts from versioned dataset files formatted as conversational ChatML messages."""
        from datasets import Dataset

        if split == "train":
            filename = "prompts_v1.jsonl"
        else:
            filename = f"{split}.jsonl"
        
        dataset_path = os.path.join(self._repo_root, "datasets", filename)
        records = []
        
        if os.path.exists(dataset_path):
            with open(dataset_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        prompt_text = data.get("prompt", data.get("text", ""))
                        if prompt_text:
                            records.append({
                                "prompt": [
                                    {"role": "system", "content": SYSTEM_PROMPT},
                                    {"role": "user", "content": f"Create generative art in p5.js: {prompt_text}"}
                                ],
                                "prompt_id": data.get("prompt_id", ""),
                                "category": data.get("category", ""),
                                "difficulty": data.get("difficulty", "")
                            })
        
        if not records:
            print(f"[WARN] Dataset not found at {dataset_path}, using fallback prompts")
            records = [
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Create generative art in p5.js: Paint a field of wildflowers with soft watercolor washes and organic brush textures"}
                    ],
                    "prompt_id": "fallback_001",
                    "category": "flower",
                    "difficulty": "easy"
                },
                {
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": "Create generative art in p5.js: A towering oak tree in autumn with leaves of gold and crimson"}
                    ],
                    "prompt_id": "fallback_002",
                    "category": "tree",
                    "difficulty": "medium"
                }
            ]
        
        return Dataset.from_list(records)

    def _clear_memory(self):
        """Aggressively free memory between phases. Critical for MPS."""
        import gc
        gc.collect()
        if self.device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        elif self.device.type == "cuda":
            torch.cuda.empty_cache()

    def train(self, max_steps=None, checkpoint_dir=None, dataset=None):
        """Run actual GRPO training loop."""
        if not self.model or not self.tokenizer:
            self.load_model()
        
        if dataset is None:
            dataset = self.load_dataset()
        reward_funcs = self.build_reward_functions()
        
        from peft import LoraConfig
        from trl import GRPOTrainer, GRPOConfig

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        output_dir = checkpoint_dir or os.path.join(
            self._repo_root, "artifacts", "checkpoints", "grpo_run"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        batch_size, num_gens = self._get_safe_batch_params()
        max_new_tokens = self._get_max_new_tokens()
        steps = max_steps or (self.config.training.max_steps if self.config else 100)
        lr = self.config.training.learning_rate if self.config else 5e-6
        
        print(f"[PaintGRPOTrainer] Training with real GRPOTrainer:")
        print(f"  batch_size={batch_size}, num_generations={num_gens}")
        print(f"  max_steps={steps}, lr={lr}")
        print(f"  max_new_tokens={max_new_tokens}")
        print(f"  output_dir={output_dir}")
        
        prompt_batch_size = num_gens
        grad_accum = max(1, batch_size // num_gens)
        
        grpo_kwargs = {
            "output_dir": output_dir,
            "learning_rate": lr,
            "per_device_train_batch_size": prompt_batch_size,
            "gradient_accumulation_steps": grad_accum,
            "num_generations": num_gens,
            "max_completion_length": max_new_tokens,
            "max_steps": steps,
            "logging_steps": 1,
            "save_steps": steps,
            "save_strategy": "steps",
            "report_to": "none",
            "temperature": self.config.generation.temperature if self.config else 0.7,
            "lr_scheduler_type": "constant",
        }
        
        if self.device.type == "mps":
            grpo_kwargs["gradient_checkpointing"] = True
            grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
            grpo_kwargs["fp16"] = False
            grpo_kwargs["bf16"] = False
        elif self.device.type == "cuda":
            grpo_kwargs["gradient_checkpointing"] = True
            grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
            grpo_kwargs["fp16"] = not torch.cuda.is_bf16_supported()
            grpo_kwargs["bf16"] = torch.cuda.is_bf16_supported()
        else:
            grpo_kwargs["gradient_checkpointing"] = False
            grpo_kwargs["fp16"] = False
            grpo_kwargs["bf16"] = False
            grpo_kwargs["use_cpu"] = True
        
        training_args = GRPOConfig(**grpo_kwargs)
        
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )
        
        train_result = self.trainer.train()
        
        final_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_dir, exist_ok=True)
        self.trainer.save_model(final_dir)
        print(f"[PaintGRPOTrainer] [OK] Model saved to {final_dir}")
        
        try:
            from paint_rl.trainer.checkpoint_validator import CheckpointValidator
            safetensors_path = os.path.join(final_dir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                CheckpointValidator.validate_safetensors(safetensors_path)
                print(f"[PaintGRPOTrainer] [OK] Checkpoint validated: {safetensors_path}")
        except Exception as e:
            print(f"[PaintGRPOTrainer] [WARN] Checkpoint validation failed: {e}")
        
        self._clear_memory()
        return train_result

    def one_step_test(self):
        """Execute exactly 1 GRPO step for hardware validation."""
        checkpoint_dir = os.path.join(
            self._repo_root, "artifacts", "checkpoints", "step_1_test"
        )
        batch_size, _ = self._get_safe_batch_params()
        raw_dataset = self.load_dataset("train")
        take_count = min(batch_size, len(raw_dataset))
        dataset = raw_dataset.select(range(take_count))
        return self.train(max_steps=1, checkpoint_dir=checkpoint_dir, dataset=dataset)

    @staticmethod
    def compute_temperature(step, t_max=0.85, t_min=0.55, tau=100):
        """Exponential temperature annealing schedule."""
        import math
        return t_min + (t_max - t_min) * math.exp(-step / tau)

    def train_cyclic(self, steps_per_cycle=25, max_steps=None, 
                     checkpoint_dir=None, unattended=False,
                     enable_dashboard=False, dashboard_path=None):
        """Interactive continuous cyclic training loop."""
        output_dir = checkpoint_dir or os.path.join(
            self._repo_root, "artifacts", "checkpoints", "cyclic_run"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.model or not self.tokenizer:
            self.load_model()
        
        dataset = self.load_dataset("train")
        reward_funcs = self.build_reward_functions()
        
        from peft import LoraConfig
        from trl import GRPOTrainer, GRPOConfig

        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        batch_size, num_gens = self._get_safe_batch_params()
        max_new_tokens = self._get_max_new_tokens()
        
        # Dashboard setup
        dashboard_writer = None
        if enable_dashboard:
            try:
                from paint_rl.telemetry.dashboard import DashboardWriter
                dash_file = dashboard_path or os.path.join(self._repo_root, "artifacts", "dashboard.html")
                dashboard_writer = DashboardWriter(dash_file)
                print(f"[Dashboard] Initialized: {dash_file}")
            except Exception as e:
                print(f"[Dashboard] Warning: Could not initialize dashboard: {e}")
        
        total_steps_done = 0
        cycle_num = 0
        cycles_to_run = 1
        all_metrics = []
        
        print("\n[PaintGRPOTrainer] Cyclic Training Started")
        print(f"  Steps/cycle: {steps_per_cycle} | Batch: {batch_size} | Group: {num_gens}")
        print(f"  Max tokens: {max_new_tokens} | Device: {self.device}")
        print(f"  Mode: {'Unattended' if unattended else 'Interactive'}")
        self._active_dashboard_writer = dashboard_writer
        self._current_step = total_steps_done
        
        while True:
            cycle_num += 1
            
            if max_steps and total_steps_done >= max_steps:
                print(f"\n[PaintGRPOTrainer] Reached max_steps={max_steps}. Stopping.")
                break
            
            cycle_steps = steps_per_cycle
            if max_steps:
                cycle_steps = min(cycle_steps, max_steps - total_steps_done)
            
            current_temp = self.compute_temperature(total_steps_done)
            print("=" * 60)
            print(f"  CYCLE {cycle_num}: Steps {total_steps_done + 1} -> {total_steps_done + cycle_steps}")
            print(f"  Target Temperature: {current_temp:.3f} | LR: 5e-6")
            print("=" * 60)
            
            self._clear_memory()
            prompt_batch_size = num_gens
            grad_accum = max(1, batch_size // num_gens)
            
            grpo_kwargs = {
                "output_dir": output_dir,
                "learning_rate": 5e-6,
                "per_device_train_batch_size": prompt_batch_size,
                "gradient_accumulation_steps": grad_accum,
                "num_generations": num_gens,
                "max_completion_length": max_new_tokens,
                "max_steps": cycle_steps,
                "logging_steps": 1,
                "save_strategy": "no",
                "report_to": "none",
                "temperature": current_temp,
                "lr_scheduler_type": "constant",
            }
            
            if self.device.type == "mps":
                grpo_kwargs["gradient_checkpointing"] = True
                grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
                grpo_kwargs["fp16"] = False
                grpo_kwargs["bf16"] = False
            elif self.device.type == "cuda":
                grpo_kwargs["gradient_checkpointing"] = True
                grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
                grpo_kwargs["fp16"] = not torch.cuda.is_bf16_supported()
                grpo_kwargs["bf16"] = torch.cuda.is_bf16_supported()
            else:
                grpo_kwargs["gradient_checkpointing"] = False
                grpo_kwargs["fp16"] = False
                grpo_kwargs["bf16"] = False
                grpo_kwargs["use_cpu"] = True
            
            training_args = GRPOConfig(**grpo_kwargs)
            
            self.trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=reward_funcs,
                args=training_args,
                train_dataset=dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config if cycle_num == 1 else None,
            )
            
            train_output = self.trainer.train()
            total_steps_done += cycle_steps
            if hasattr(self.trainer, "model") and self.trainer.model is not None:
                self.model = self.trainer.model
            
            # Extract real RL training metrics from log history
            log_hist = self.trainer.state.log_history if hasattr(self.trainer, "state") and self.trainer.state else []
            rewards_list = [entry["reward"] for entry in log_hist if "reward" in entry]
            grad_norms = [entry["grad_norm"] for entry in log_hist if "grad_norm" in entry]
            entropies = [entry["entropy"] for entry in log_hist if "entropy" in entry]
            
            mean_reward = float(sum(rewards_list) / len(rewards_list)) if rewards_list else 0.0
            mean_grad_norm = float(sum(grad_norms) / len(grad_norms)) if grad_norms else 0.0
            mean_entropy = float(sum(entropies) / len(entropies)) if entropies else 0.0
            cycle_loss = getattr(train_output, 'training_loss', 0.0)
            
            cycle_metric = {
                "cycle": cycle_num,
                "steps_done": total_steps_done,
                "loss": float(cycle_loss) if cycle_loss else 0.0,
                "reward": round(mean_reward, 4),
                "grad_norm": round(mean_grad_norm, 4),
                "entropy": round(mean_entropy, 4),
                "temperature": float(current_temp),
            }
            all_metrics.append(cycle_metric)
            
            print(f"\n  [Cycle {cycle_num} Summary] Steps: {total_steps_done} | Mean Reward: {mean_reward:.3f} | Grad Norm: {mean_grad_norm:.4f} | Temp: {current_temp:.3f}")
            
            if dashboard_writer:
                dashboard_writer.update(all_metrics)
            
            cycles_to_run -= 1
            if cycles_to_run > 0:
                continue
            
            if unattended:
                if max_steps and total_steps_done >= max_steps:
                    break
                cycles_to_run = 1
                continue
            
            try:
                user_input = input(
                    f"\n  Continue training? [y (1 cycle) / n (save & quit) / <number> (N cycles)]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                user_input = "n"
            
            if user_input in ("n", "no", "q", "quit", "exit"):
                print("\n[PaintGRPOTrainer] Saving checkpoint and exiting...")
                break
            elif user_input.isdigit() and int(user_input) > 0:
                cycles_to_run = int(user_input)
                print(f"[PaintGRPOTrainer] Running next {cycles_to_run} cycles autonomously.")
            else:
                cycles_to_run = 1
        
        final_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_dir, exist_ok=True)
        if self.trainer:
            self.trainer.save_model(final_dir)
            print(f"[PaintGRPOTrainer] [OK] Final checkpoint saved to {final_dir}")
        
        from paint_rl.trainer.checkpoint_validator import CheckpointValidator
        safetensors_path = os.path.join(final_dir, "adapter_model.safetensors")
        if os.path.exists(safetensors_path):
            try:
                CheckpointValidator.validate_safetensors(safetensors_path)
                print(f"[PaintGRPOTrainer] [OK] Checkpoint validated: {safetensors_path}")
            except Exception as e:
                print(f"[PaintGRPOTrainer] [WARN] Checkpoint validation failed: {e}")
        
        self._clear_memory()
        
        return {
            "total_steps": total_steps_done,
            "total_cycles": cycle_num,
            "metrics": all_metrics,
            "final_loss": all_metrics[-1]["loss"] if all_metrics else None,
        }
