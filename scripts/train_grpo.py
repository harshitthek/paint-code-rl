#!/usr/bin/env python3
"""Thin CLI entrypoint for GRPO training.

Sets up MPS fallback, auto-detects device, and delegates to PaintGRPOTrainer.

Usage:
    python scripts/train_grpo.py --mode one_step
    python scripts/train_grpo.py --mode train --max-steps 100
"""
import os
import sys
import argparse

# Set MPS fallback BEFORE any torch/transformers imports
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def _print_system_info():
    """Print system info for debugging hardware validation runs."""
    import torch
    import psutil
    
    ram_gb = round(psutil.virtual_memory().total / 1e9, 2)
    avail_gb = round(psutil.virtual_memory().available / 1e9, 2)
    
    print("=" * 60)
    print("   PAINT-CODE-RL: GRPO TRAINING")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    mps_avail = hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    print(f"  MPS available: {mps_avail}")
    print(f"  RAM: {ram_gb} GB total / {avail_gb} GB available")
    print(f"  MPS fallback: {os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', 'not set')}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            vram = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f} GB)")
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description='Paint-Code-RL GRPO Training')
    parser.add_argument('--mode', choices=['one_step', 'train', 'resume'], default='one_step',
                        help='Training mode: one_step (hardware validation), train (cyclic), resume')
    parser.add_argument('--max-steps', type=int, default=None,
                        help='Maximum training steps (overrides config)')
    parser.add_argument('--steps-per-cycle', type=int, default=25,
                        help='Number of training steps per interactive cycle (default: 25)')
    parser.add_argument('--unattended', action='store_true',
                        help='Run without interactive prompts (for CI/Kaggle/headless runs)')
    parser.add_argument('--max', action='store_true',
                        help='Maximize hardware resource saturation (RAM, GPU, CPU cores)')
    parser.add_argument('--dashboard', action='store_true',
                        help='Enable live auto-refreshing HTML dashboard (artifacts/dashboard.html)')
    parser.add_argument('--checkpoint-dir', type=str, default='artifacts/checkpoints',
                        help='Directory for saving checkpoints')
    parser.add_argument('--no-renderer', action='store_true',
                        help='Skip renderer startup (syntax reward only)')
    args = parser.parse_args()
    
    _print_system_info()
    
    # Start renderer (unless explicitly skipped)
    renderer = None
    if not args.no_renderer:
        from paint_rl.renderer.manager import RendererService
        renderer = RendererService(port=3000)
        print("Starting renderer daemon...")
        if renderer.ensure_started(max_wait_sec=15):
            print("[OK] Renderer ready")
        else:
            print("[WARN] Renderer failed to start (will use syntax reward only)")
            print("   Fix: cd renderer && npm install && node server.js")
            renderer = None
    
    # Load config (auto-detects MPS and merges overlay)
    from paint_rl.config.core import ACTIVE_CONFIG, apply_max_hardware_config
    from paint_rl.trainer.grpo import PaintGRPOTrainer
    
    if args.max:
        adjustments = apply_max_hardware_config(ACTIVE_CONFIG)
        print(f"[--max] Hardware saturated: {adjustments}")
    
    trainer = PaintGRPOTrainer(config=ACTIVE_CONFIG, renderer_service=renderer)
    
    print(f"\nDevice: {trainer.device}")
    print(f"Model: {trainer._resolve_model_id()}")
    print(f"Dtype: {trainer._get_dtype()}")
    batch_size, group_size = trainer._get_safe_batch_params()
    print(f"Batch size: {batch_size} | Group size: {group_size}")
    print(f"Max tokens: {trainer._get_max_new_tokens()}")
    print()
    
    if args.mode == 'one_step':
        print("Running 1-step hardware validation...")
        result = trainer.one_step_test()
        print("\n[OK] 1-step GRPO validation COMPLETE")
        print(f"   Training loss: {result.training_loss:.4f}")
    else:
        print(f"Starting cyclic training (steps_per_cycle={args.steps_per_cycle}, max_steps={args.max_steps})...")
        result = trainer.train_cyclic(
            steps_per_cycle=args.steps_per_cycle,
            max_steps=args.max_steps,
            checkpoint_dir=args.checkpoint_dir,
            unattended=args.unattended,
            enable_dashboard=args.dashboard
        )
        print("\n[OK] Cyclic Training COMPLETE")
        print(f"   Total steps: {result.get('total_steps', 0)} | Total cycles: {result.get('total_cycles', 0)}")


if __name__ == '__main__':
    main()
