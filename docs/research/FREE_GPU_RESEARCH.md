# FREE GPU RESEARCH & LOCAL VLM BENCHMARKS

*Date: August 2026*

## 1. Free Compute Environments

To achieve a true `$0 default path` for Phase-0, we evaluated legitimate free compute sources without relying on expiring credits.

| Platform | Free GPU Target | Limits | Suitability |
| :--- | :--- | :--- | :--- |
| **Kaggle Notebooks** | 2x T4 (16GB each) | ~30 hrs/week, 9hr session | **Excellent.** Highly predictable, excellent persistence via `/kaggle/working`. 32GB total VRAM supports quantized 7B training + Local VLM. |
| **Google Colab** | 1x T4 (16GB) | 12hr session, idle timeouts | **Good.** Good for DRY_RUN or small prototypes. Unpredictable preemption makes long GRPO runs frustrating. |
| **Lightning AI** | 1x T4 | 15 credits/mo (~22 hrs) | **Fair.** Persistent IDE is nice, but credit limit is hard-capped. |
| **SageMaker Studio Lab** | N/A | Discontinued (July 2026) | **N/A** |

**Conclusion:** `Kaggle` is the official backend for `FREE` mode. The architecture must checkpoint aggressively (every 15-30 mins) to survive 9-hour session walls.

## 2. Local Open-Weight VLM Selection

Since `FREE` mode forbids OpenAI/Anthropic API costs, we must deploy a local VLM Judge. The primary constraint is VRAM (fitting alongside the training policy on 16GB-24GB cards).

| Model | Size / VRAM | Pairwise Suitability | Instruction Following |
| :--- | :--- | :--- | :--- |
| **Qwen2.5-VL (7B)** | ~14GB (FP16) | **High.** Dominant all-rounder. | Excellent. Consistently outputs strict JSON decisions (`left`, `right`, `tie`). |
| **Pixtral (12B)** | ~24GB (FP16) | **High.** Strong visual understanding. | Good, but heavier on VRAM. |
| **Nemotron 3 Nano Omni** | ~60GB (FP16) | **Highest.** The industry leader for reasoning. | Unusable in `FREE` mode (requires multiple GPUs). |

**Conclusion:** The `LocalJudgeProvider` will default to **Qwen2.5-VL-7B** (quantized to INT4/INT8 if necessary) to fit alongside the Policy model on Kaggle's dual T4s.
