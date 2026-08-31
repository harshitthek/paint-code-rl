#!/usr/bin/env python3
"""Thin CLI entrypoint for GRPO training."""
import argparse
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from paint_rl.trainer.grpo import PaintGRPOTrainer
from paint_rl.renderer.manager import RendererService

def main():
    parser = argparse.ArgumentParser(description='Paint-Code-RL GRPO Training')
    parser.add_argument('--mode', choices=['one_step', 'train', 'resume'], default='one_step')
    parser.add_argument('--max-steps', type=int, default=None)
    parser.add_argument('--checkpoint-dir', type=str, default='artifacts/checkpoints')
    args = parser.parse_args()
    
    renderer = RendererService(port=3000)
    renderer.ensure_started()
    
    from paint_rl.config.core import ACTIVE_CONFIG
    trainer = PaintGRPOTrainer(config=ACTIVE_CONFIG, renderer_service=renderer)
    
    if args.mode == 'one_step':
        trainer.one_step_test()
    else:
        trainer.train(max_steps=args.max_steps, checkpoint_dir=args.checkpoint_dir)

if __name__ == '__main__':
    main()
