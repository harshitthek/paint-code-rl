"""Canonical GRPO trainer for Paint-Code-RL.

Handles device-aware model selection, memory-safe batch sizes,
gradient checkpointing with PEFT/LoRA, conversational ChatML prompts,
multi-tier reward composition, and MPS-specific optimizations for Apple Silicon.
"""
import os
import sys
import json
import torch
from datasets import Dataset

from paint_rl.config.prompts import SYSTEM_PROMPT
from paint_rl.utils.code_extractor import robust_extract_js_code


def get_compute_device():
    """Device-agnostic compute device selection per AGENTS.md Golden Rule #2."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Maximum safe training parameters per device type (16GB Apple Silicon budget)
_MPS_SAFE_LIMITS = {
    "max_batch_size": 2,
    "max_group_size": 2,
    "max_new_tokens": 256,
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
    """Canonical GRPO trainer for Paint-Code-RL.
    
    Device-aware behavior:
    - MPS: Forces 1.5B model, float32, batch_size=2, group_size=2, gradient_checkpointing with non-reentrant mode
    - CUDA: Uses config model (7B if enough VRAM), float16/bfloat16
    - CPU: Forces 0.5B model, float32
    """
    
    def __init__(self, config=None, renderer_service=None):
        self.config = config
        self.device = get_compute_device()
        self.renderer = renderer_service
        self.model = None
        self.tokenizer = None
        self.trainer = None
        self._repo_root = _find_repo_root()
    
    def select_model_id(self):
        """Select model ID based on device capabilities.
        
        On MPS/CPU, always overrides to a memory-safe model regardless of config.
        On CUDA, respects config but validates against available VRAM.
        """
        if self.device.type in _MODEL_BY_DEVICE:
            return _MODEL_BY_DEVICE[self.device.type]
        
        # CUDA: check VRAM
        if self.device.type == "cuda":
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb >= 20:
                    return "Qwen/Qwen2.5-Coder-7B-Instruct"
            except Exception:
                pass
            return "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        
        return "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    def _resolve_model_id(self):
        """Resolve final model ID: device constraints override config."""
        config_model = self.config.model.id if self.config else None
        device_model = self.select_model_id()
        
        # On MPS/CPU, always use the device-safe model
        if self.device.type in ("mps", "cpu"):
            if config_model and config_model != device_model:
                print(f"[INFO] Config model '{config_model}' overridden to "
                      f"'{device_model}' for {self.device.type} compatibility")
            return device_model
        
        # On CUDA, use config model if set, otherwise device selection
        return config_model or device_model
    
    def _get_dtype(self):
        """Get appropriate dtype for the device.
        
        MPS: float32 (FP16 causes gradient instabilities in GRPO backward pass on MPS)
        CUDA: float16 (bfloat16 if supported)
        CPU: float32
        """
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        # MPS and CPU: float32 for stability
        return torch.float32
    
    def _get_safe_batch_params(self):
        """Get memory-safe batch size and group size for current device.
        
        TRL GRPOTrainer invariant: batch_size must be a positive multiple of group_size!
        """
        if self.config:
            batch_size = self.config.training.batch_size
            group_size = self.config.training.group_size
        else:
            batch_size = 2
            group_size = 2
        
        # Enforce MPS-safe limits
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
        
        # Explicitly configure causal attention without sliding window mismatches
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
        
        # Critical for LoRA + Gradient Checkpointing:
        # Ensures input embedding outputs require gradients so PyTorch's checkpoint function
        # does not skip backward graph construction.
        if hasattr(self.model, "enable_input_require_grads"):
            self.model.enable_input_require_grads()
        
        print(f"[PaintGRPOTrainer] Model loaded: {model_id} "
              f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.0f}M params)")
        
        return self.model, self.tokenizer
    
    def build_reward_functions(self):
        """Build multi-tier reward functions for GRPO training.
        
        Returns list of callables matching TRL GRPOTrainer reward_funcs API:
        1. Structural & Natural-Media Reward (p5.brush utilization & anti-cheat)
        2. Render Execution & Visual Richness Reward (pixel entropy & canvas coverage)
        """
        from paint_rl.rewards.aesthetic import calculate_visual_richness, calculate_brush_utilization

        def structural_syntax_reward(prompts, completions, **kwargs):
            rewards = []
            for completion in completions:
                code = robust_extract_js_code(completion)
                brush_meta = calculate_brush_utilization(code)
                
                # Anti-Cheat: Text-in-image or empty trivial code
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
                
                # Add brush structural utilization score (up to 0.60)
                score += 0.60 * brush_meta.get("brush_score", 0.0)
                rewards.append(round(min(1.0, score), 3))
            return rewards

        def render_and_visual_richness_reward(prompts, completions, **kwargs):
            rewards = []
            for i, completion in enumerate(completions):
                if not self.renderer:
                    rewards.append(0.5)
                    continue
                code = robust_extract_js_code(completion)
                try:
                    # Extract prompt text for renderer context
                    prompt_str = ""
                    if i < len(prompts):
                        p = prompts[i]
                        if isinstance(p, list) and len(p) > 1 and isinstance(p[1], dict):
                            prompt_str = p[1].get("content", "")
                        else:
                            prompt_str = str(p)
                    
                    res = self.renderer.render(code, seed=42, prompt=prompt_str)
                    if res.get("success") and res.get("image_path"):
                        # Inspect rendered canvas pixels (anti-blank & color variance)
                        richness_meta = calculate_visual_richness(res["image_path"])
                        if richness_meta.get("is_blank"):
                            # Blank canvas hack penalization
                            rewards.append(0.05)
                        else:
                            # 0.35 base render success + 0.65 visual richness score
                            richness_score = richness_meta.get("richness_score", 0.0)
                            total_r = 0.35 + 0.65 * richness_score
                            rewards.append(round(min(1.0, total_r), 3))
                    else:
                        rewards.append(0.0)
                except Exception:
                    rewards.append(0.0)
            return rewards

        return [structural_syntax_reward, render_and_visual_richness_reward]
    
    def load_dataset(self, split="train"):
        """Load prompts from versioned dataset files formatted as conversational ChatML messages.
        
        Formatting as [{"role": "system", ...}, {"role": "user", ...}] instructs TRL GRPOTrainer's
        maybe_apply_chat_template to invoke Qwen's tokenizer chat template, guaranteeing
        that instruction-tuned models generate executable p5.js code rather than conversational prose.
        """
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
            # Fallback prompts for smoke testing when dataset is missing
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
        """Run actual GRPO training loop.
        
        Args:
            max_steps: Number of training steps (default from config).
            checkpoint_dir: Directory to save checkpoints.
            dataset: HuggingFace Dataset (default loads from prompts_v1.jsonl).
            
        Returns:
            TrainOutput from TRL GRPOTrainer.
        """
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
        
        if not self.model or not self.tokenizer:
            self.load_model()
        
        if dataset is None:
            dataset = self.load_dataset()
        reward_funcs = self.build_reward_functions()
        
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
        steps = max_steps if max_steps is not None else (
            self.config.training.max_steps if self.config else 1
        )
        
        print(f"[PaintGRPOTrainer] Training params: "
              f"batch_size={batch_size}, num_generations={num_gens}, "
              f"max_new_tokens={max_new_tokens}, max_steps={steps}")
        
        # Build GRPOConfig with device-appropriate settings
        grpo_kwargs = dict(
            output_dir=output_dir,
            learning_rate=5e-6,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1,
            num_generations=num_gens,
            max_completion_length=max_new_tokens,
            max_steps=steps,
            logging_steps=1,
            save_steps=max(1, steps),
            report_to="none",
            temperature=0.7,
        )
        
        # Enable gradient checkpointing ONLY on GPU/MPS with constrained VRAM
        if self.device.type == "mps":
            grpo_kwargs["gradient_checkpointing"] = True
            grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
        
        if self.device.type == "cpu":
            grpo_kwargs["use_cpu"] = True
        
        training_args = GRPOConfig(**grpo_kwargs)
        
        self._clear_memory()
        
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_funcs=reward_funcs,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer,
            peft_config=peft_config,
        )
        
        print(f"[PaintGRPOTrainer] Starting training on {self.device}...")
        train_result = self.trainer.train()
        
        # Save final adapter
        final_dir = os.path.join(output_dir, "final_adapter")
        self.trainer.save_model(final_dir)
        
        # Validate checkpoint
        try:
            from paint_rl.trainer.checkpoint_validator import CheckpointValidator
            safetensors_path = os.path.join(final_dir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                CheckpointValidator.validate_safetensors(safetensors_path)
                print(f"[PaintGRPOTrainer] [OK] Checkpoint validated: {safetensors_path}")
        except ImportError:
            pass
        except Exception as e:
            print(f"[PaintGRPOTrainer] [ERROR] Checkpoint validation failed: {e}")
            raise
        
        self._clear_memory()
        return train_result
    
    def one_step_test(self):
        """Execute exactly 1 GRPO step for hardware validation."""
        checkpoint_dir = os.path.join(
            self._repo_root, "artifacts", "checkpoints", "step_1_test"
        )
        batch_size, _ = self._get_safe_batch_params()
        dataset = self.load_dataset("train").select(range(batch_size))
        return self.train(max_steps=1, checkpoint_dir=checkpoint_dir, dataset=dataset)

    @staticmethod
    def compute_temperature(step, t_max=0.85, t_min=0.55, tau=100):
        """Exponential temperature annealing schedule.
        
        T(step) = T_min + (T_max - T_min) * exp(-step / tau)
        
        Starts at T=0.85 (exploring diverse brush strokes) and smoothly
        anneals to T=0.55 (perfecting high-scoring algorithms).
        """
        import math
        return t_min + (t_max - t_min) * math.exp(-step / tau)

    def train_cyclic(self, steps_per_cycle=25, max_steps=None, 
                     checkpoint_dir=None, unattended=False,
                     enable_dashboard=False, dashboard_path=None):
        """Interactive cyclic continuous training engine.
        
        Executes training in discrete cycles. At the end of each cycle:
        1. Renders sample artworks for active prompts
        2. Prints diagnostic scorecard to console
        3. Updates live HTML dashboard (if enabled)
        4. Prompts user: [y (1 cycle) / n (save & quit) / <number> (N cycles)]
        
        Model, tokenizer, and optimizer remain permanently resident in
        GPU/MPS memory across cycles — no reloading between cycles.
        
        Args:
            steps_per_cycle: Training steps per interactive cycle.
            max_steps: Total step budget (None = unlimited).
            checkpoint_dir: Directory for checkpoints.
            unattended: If True, skip interactive prompts (run until max_steps).
            enable_dashboard: If True, write live dashboard HTML.
            dashboard_path: Path for dashboard.html output.
        """
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
        
        if not self.model or not self.tokenizer:
            self.load_model()
        
        dataset = self.load_dataset()
        reward_funcs = self.build_reward_functions()
        
        peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        output_dir = checkpoint_dir or os.path.join(
            self._repo_root, "artifacts", "checkpoints", "cyclic_run"
        )
        os.makedirs(output_dir, exist_ok=True)
        
        batch_size, num_gens = self._get_safe_batch_params()
        max_new_tokens = self._get_max_new_tokens()
        
        # Dashboard setup
        dashboard_writer = None
        if enable_dashboard:
            try:
                from paint_rl.telemetry.dashboard import DashboardWriter
                dash_path = dashboard_path or os.path.join(
                    self._repo_root, "artifacts", "dashboard.html"
                )
                dashboard_writer = DashboardWriter(dash_path)
                print(f"[PaintGRPOTrainer] Dashboard: file://{dash_path}")
            except ImportError:
                print("[WARN] Dashboard module not found, skipping dashboard")
        
        total_steps_done = 0
        cycle_num = 0
        cycles_to_run = 1  # Start with 1 cycle
        all_metrics = []
        
        print(f"\n[PaintGRPOTrainer] Cyclic Training Started")
        print(f"  Steps/cycle: {steps_per_cycle} | Batch: {batch_size} | Group: {num_gens}")
        print(f"  Max tokens: {max_new_tokens} | Device: {self.device}")
        print(f"  Mode: {'Unattended' if unattended else 'Interactive'}")
        print()
        
        while True:
            cycle_num += 1
            
            # Check total step budget
            if max_steps and total_steps_done >= max_steps:
                print(f"\n[PaintGRPOTrainer] Reached max_steps={max_steps}. Stopping.")
                break
            
            # Calculate steps for this cycle
            cycle_steps = steps_per_cycle
            if max_steps:
                remaining = max_steps - total_steps_done
                cycle_steps = min(steps_per_cycle, remaining)
            
            # Temperature annealing
            temperature = self.compute_temperature(total_steps_done)
            
            print(f"\n{'='*60}")
            print(f"  CYCLE {cycle_num} | Steps {total_steps_done+1}-{total_steps_done+cycle_steps} | T={temperature:.3f}")
            print(f"{'='*60}")
            
            # Build GRPOConfig for this cycle
            grpo_kwargs = dict(
                output_dir=output_dir,
                learning_rate=5e-6,
                per_device_train_batch_size=batch_size,
                gradient_accumulation_steps=1,
                num_generations=num_gens,
                max_completion_length=max_new_tokens,
                max_steps=cycle_steps,
                logging_steps=1,
                save_steps=max(1, cycle_steps),
                report_to="none",
                temperature=temperature,
            )
            
            if self.device.type == "mps":
                grpo_kwargs["gradient_checkpointing"] = True
                grpo_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
            
            if self.device.type == "cpu":
                grpo_kwargs["use_cpu"] = True
            
            training_args = GRPOConfig(**grpo_kwargs)
            self._clear_memory()
            
            # Create trainer (reuses same model weights from previous cycle)
            self.trainer = GRPOTrainer(
                model=self.model,
                reward_funcs=reward_funcs,
                args=training_args,
                train_dataset=dataset,
                processing_class=self.tokenizer,
                peft_config=peft_config if cycle_num == 1 else None,
            )
            
            # Run training cycle
            train_result = self.trainer.train()
            total_steps_done += cycle_steps
            
            # Collect metrics
            cycle_metrics = {
                "cycle": cycle_num,
                "steps_done": total_steps_done,
                "loss": round(train_result.training_loss, 4),
                "temperature": round(temperature, 3),
            }
            all_metrics.append(cycle_metrics)
            
            # Print cycle summary
            print(f"\n  Cycle {cycle_num} Complete (Steps 1-{total_steps_done})")
            print(f"  Loss: {cycle_metrics['loss']} | Temperature: {cycle_metrics['temperature']}")
            
            # Update dashboard
            if dashboard_writer:
                try:
                    dashboard_writer.update(all_metrics)
                except Exception as e:
                    print(f"[WARN] Dashboard update failed: {e}")
            
            # Check if we've exhausted budget in unattended mode
            if unattended:
                if max_steps and total_steps_done >= max_steps:
                    break
                continue
            
            # Interactive prompt
            if cycles_to_run > 1:
                cycles_to_run -= 1
                continue
            
            try:
                user_input = input(
                    f"\n  Continue training? [y (1 cycle) / n (save & quit) / <number> (N cycles)]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                user_input = "n"
            
            if user_input in ("n", "q", "quit", "no"):
                print("\n[PaintGRPOTrainer] Saving and exiting...")
                break
            elif user_input in ("y", "yes", ""):
                cycles_to_run = 1
            else:
                try:
                    cycles_to_run = max(1, int(user_input))
                    print(f"  Running {cycles_to_run} cycles autonomously...")
                except ValueError:
                    print("  Invalid input, running 1 more cycle...")
                    cycles_to_run = 1
        
        # Final save
        final_dir = os.path.join(output_dir, "final_adapter")
        os.makedirs(final_dir, exist_ok=True)
        if self.trainer:
            self.trainer.save_model(final_dir)
            print(f"[PaintGRPOTrainer] [OK] Final checkpoint saved to {final_dir}")
        
        # Validate checkpoint
        try:
            from paint_rl.trainer.checkpoint_validator import CheckpointValidator
            safetensors_path = os.path.join(final_dir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path):
                CheckpointValidator.validate_safetensors(safetensors_path)
                print(f"[PaintGRPOTrainer] [OK] Checkpoint validated")
        except (ImportError, Exception):
            pass
        
        self._clear_memory()
        
        return {
            "total_steps": total_steps_done,
            "total_cycles": cycle_num,
            "metrics": all_metrics,
            "final_loss": all_metrics[-1]["loss"] if all_metrics else None,
        }
