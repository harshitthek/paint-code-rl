# ADR-008: Multi-Signal Visual RL Reward Matrix & Ephemeral Sandboxing

**Status:** Accepted  
**Context:**  
Initial GRPO training evaluated models using coarse binary compilation checks and static aesthetic prompts. This led to policy failure modes:
1. **The Blank Canvas Hack:** Solid colored screens receiving maximum compile rewards with zero error risk.
2. **Semantic Drift:** Policy ignoring prompt subjects because the scorer only compared against generic "beautiful image" anchors.
3. **Text-in-Image Hacking:** LLMs drawing `text()` primitives to exploit CLIP token matching.
4. **Attention Warnings:** PyTorch SDPA causal attention emitting sliding-window warnings when loading Qwen2.5-Coder.

**Decision:**  
1. Implement a hierarchical **5-tier verifiable visual reward matrix**:
   - $R_{\text{compile}}$ (binary execution gate).
   - $R_{\text{prompt}}$ (differential CLIP cosine similarity against the user prompt vs negative blank anchor).
   - $R_{\text{richness}}$ (closed-form pixel standard deviation, active canvas coverage ratio, and color palette entropy).
   - $R_{\text{brush}}$ (p5.brush natural-media feature detector + anti-text cheat filter).
   - $R_{\text{aesthetic}}$ (global composition harmony).
2. Fix SDPA sliding window attention cleanly at the `AutoConfig` level (`sliding_window = None`, `use_sliding_window = False`).
3. Maintain ephemeral clean browser contexts to eliminate WebGL shader and memory leaks across long training runs.

**Consequences:**  
- Stronger policy steering toward the specific text prompt.
- Total elimination of blank-canvas and single-dot reward hacks.
- Zero kernel warning logs during model loading.
- Sub-50ms render evaluation on Apple Silicon Metal GPUs.
