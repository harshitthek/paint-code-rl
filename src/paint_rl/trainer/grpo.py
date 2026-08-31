import os
import torch
from datasets import Dataset

def get_compute_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

class PaintGRPOTrainer:
    """Canonical GRPO trainer for Paint-Code-RL."""
    
    def __init__(self, config=None, renderer_service=None):
        self.config = config
        self.device = get_compute_device()
        self.renderer = renderer_service
        self.model = None
        self.tokenizer = None
        self.trainer = None
    
    def select_model_id(self):
        if self.device.type == "mps":
            return "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        elif self.device.type == "cuda":
            import torch
            try:
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if vram_gb >= 20:
                    return "Qwen/Qwen2.5-Coder-7B-Instruct"
            except: pass
            return "Qwen/Qwen2.5-Coder-1.5B-Instruct"
        return "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    
    def load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_id = self.config.model.id if self.config else self.select_model_id()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.device.type in ["cuda", "mps"] else torch.float32
        ).to(self.device)
        return self.model, self.tokenizer
    
    def build_reward_functions(self):
        def code_syntax_reward(prompts, completions, **kwargs):
            rewards = []
            for completion in completions:
                code = completion
                if isinstance(code, list):
                    if len(code) > 0 and isinstance(code[-1], dict) and "content" in code[-1]:
                        code = code[-1]["content"]
                    else: code = str(code)
                if "setup" in code and "createCanvas" in code:
                    rewards.append(1.0)
                elif "function" in code:
                    rewards.append(0.5)
                else: rewards.append(0.0)
            return rewards

        def render_validity_reward(prompts, completions, **kwargs):
            rewards = []
            for completion in completions:
                if not self.renderer:
                    rewards.append(0.5)
                    continue
                code = completion
                if isinstance(code, list):
                    if len(code) > 0 and isinstance(code[-1], dict) and "content" in code[-1]:
                        code = code[-1]["content"]
                    else: code = str(code)
                res = self.renderer.render(code, seed=42)
                if res.get("success"): rewards.append(1.0)
                else: rewards.append(0.0)
            return rewards
        return [code_syntax_reward, render_validity_reward]
    
    def load_dataset(self, split="train"):
        import json
        dataset_path = os.path.join("datasets", f"{split}.jsonl" if split != "train" else "prompts_v1.jsonl")
        prompts = []
        if os.path.exists(dataset_path):
            with open(dataset_path) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        prompts.append({"prompt": data.get("prompt", data.get("text", ""))})
        if not prompts:
            prompts = [
                {"prompt": "Create a generative watercolor painting of a mountain landscape in p5.js and p5.brush."},
                {"prompt": "Draw an abstract geometric composition with dynamic brush strokes in p5.js."}
            ]
        return Dataset.from_list(prompts)
    
    def train(self, max_steps=None, checkpoint_dir=None):
        from trl import GRPOTrainer, GRPOConfig
        from peft import LoraConfig
        if not self.model or not self.tokenizer: self.load_model()
        dataset = self.load_dataset()
        reward_funcs = self.build_reward_functions()
        peft_config = LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM")
        
        output_dir = checkpoint_dir or os.path.join("artifacts", "checkpoints", "grpo_run")
        os.makedirs(output_dir, exist_ok=True)
        batch_size = self.config.training.batch_size if self.config else 1
        num_gens = self.config.training.group_size if self.config else 2
        
        training_args = GRPOConfig(
            output_dir=output_dir, learning_rate=5e-6, per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=1, num_generations=num_gens, max_prompt_length=128, max_completion_length=256,
            max_steps=max_steps if max_steps is not None else (self.config.training.max_steps if self.config else 1),
            logging_steps=1, save_steps=1, report_to="none"
        )
        self.trainer = GRPOTrainer(model=self.model, reward_funcs=reward_funcs, args=training_args, train_dataset=dataset, peft_config=peft_config)
        train_result = self.trainer.train()
        
        final_dir = os.path.join(output_dir, "final_adapter")
        self.trainer.save_model(final_dir)
        try:
            from paint_rl.trainer.checkpoint_validator import CheckpointValidator
            safetensors_path = os.path.join(final_dir, "adapter_model.safetensors")
            if os.path.exists(safetensors_path): CheckpointValidator.validate_safetensors(safetensors_path)
        except ImportError: pass
        return train_result
    
    def one_step_test(self):
        checkpoint_dir = os.path.join("artifacts", "checkpoints", "step_1_test")
        return self.train(max_steps=1, checkpoint_dir=checkpoint_dir)
