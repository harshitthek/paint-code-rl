# System Architecture Specification

> Complete architectural reference for the `paint-code-rl` generative art reinforcement learning platform.

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Data & Prompt Layer
        PromptDB[(Versioned Prompts\ndatasets/prompts_v1.jsonl)]
        ChatML[ChatML Conversational Formatter]
        PromptDB --> ChatML
    end

    subgraph Generation & Policy Subsystem
        ChatML --> Policy[Qwen2.5-Coder Policy Model]
        TempSched[Temperature Annealing Schedule\nT = 0.85 -> 0.55] -.-> Policy
        Policy --> CodeGen[Generated p5.js / p5.brush Program]
    end

    subgraph Sandboxed WebGL Execution Engine
        CodeGen --> BatchClient[Renderer Manager Client]
        BatchClient --> HttpServer[Node.js Express Daemon :3000\nPOST /render_batch]
        HttpServer --> PuppeteerPool[Headless Chromium Context Pool]
        PuppeteerPool --> WebGLCtx[ANGLE Metal / SwiftShader WebGL2]
        WebGLCtx --> Base64Stream[In-Memory Base64 Screenshot Stream]
    end

    subgraph Multi-Signal Reward Matrix
        Base64Stream --> RewardComposer[RewardComposer Engine]
        CodeGen --> RewardComposer
        ChatML --> RewardComposer

        RewardComposer --> R1[1. Compile Gate: 0.10]
        RewardComposer --> R2[2. Prompt Alignment CLIP: 0.35]
        RewardComposer --> R3[3. Visual Richness & Laplacian: 0.25]
        RewardComposer --> R4[4. Brush Utilization & Anti-Cheat: 0.15]
        RewardComposer --> R5[5. Aesthetic Scorer: 0.15]

        R1 --> TotalR[Total Advantage & Loss Calculation]
        R2 --> TotalR
        R3 --> TotalR
        R4 --> TotalR
        R5 --> TotalR
    end

    subgraph Optimization & Cyclic Engine
        TotalR --> GRPOTrainer[TRL GRPOTrainer]
        GRPOTrainer --> LoRAUpdate[LoRA Parameter Optimization]
        LoRAUpdate --> Policy
        GRPOTrainer --> CyclicLoop[Cyclic Continuous Training Loop]
    end

    subgraph Diagnostic & Telemetry Subsystem
        RewardComposer --> Scorecard[Diagnostic Scorecard Report]
        CyclicLoop --> Scorecard
        CyclicLoop --> LiveDash[Live HTML Dashboard\nartifacts/dashboard.html]
        CyclicLoop --> JSONLLog[Structured JSONL Logs\nartifacts/logs/]
    end
```

---

## 2. Core Subsystems

### A. Policy & Generation Subsystem
* **Model Selection:** Automatically selects target model based on available hardware:
  * `Apple Silicon (MPS):` `Qwen/Qwen2.5-Coder-1.5B-Instruct` (FP32, LoRA rank $r=8$)
  * `NVIDIA CUDA (>= 20GB):` `Qwen/Qwen2.5-Coder-7B-Instruct`
  * `CPU Development:` `Qwen/Qwen2.5-Coder-0.5B-Instruct`
* **Sliding Window Clean Setup:** Injected `sliding_window = None` at `AutoConfig` level to prevent sliding window attention mismatch warnings.
* **Temperature Annealing:** Controlled by `PaintGRPOTrainer.compute_temperature(step)`:
  $$\text{Temperature}(\text{step}) = 0.55 + 0.30 \cdot \exp\left(-\frac{\text{step}}{100}\right)$$

### B. Sandboxed WebGL Execution Engine
* **Technology:** Node.js Express server (`renderer/server.js`) hosting Puppeteer Chromium sandboxes (`renderer/sandbox.js`).
* **GPU Acceleration:**
  * macOS: Native Metal GPU rasterization via `--use-gl=angle` and `--use-angle=metal`.
  * Linux / Windows: SwiftShader software rasterizer via `--use-gl=angle` and `--use-angle=swiftshader-webgl`.
* **Security & Isolation:**
  * Outbound network exfiltration blocked via request interception (`page.setRequestInterception(true)`).
  * Path traversal blocked by strict `runId` sanitization.
  * Signal tampering prevented by immutable `Object.defineProperty(window, 'signalRenderComplete', ...)`.
  * Guaranteed ephemeral page closure and temporary file unlinking in `finally` blocks.

### C. 5-Tier Verifiable Visual Reward Matrix
Every candidate generation is evaluated across 5 non-hackable visual and structural criteria:

| Tier | Component | Weight | Metric Formula & Function |
| :--- | :--- | :--- | :--- |
| **1** | **Compile Gate** | `0.10` | Binary $0/1$ render verification with exact error classification (`TIMEOUT`, `PARSE_ERROR`, `NO_CANVAS`, `RUNTIME_ERROR`). |
| **2** | **Prompt Alignment** | `0.35` | Differential CLIP cosine similarity calibrated against negative blank anchors: $\Delta = \text{sim}(\text{prompt}) - \text{sim}(\text{blank})$. |
| **3** | **Visual Richness** | `0.25` | $0.30 \cdot \text{Coverage} + 0.25 \cdot \text{Std} + 0.20 \cdot \text{PaletteEntropy} + 0.25 \cdot \text{Laplacian}(\nabla^2 I)$. |
| **4** | **Brush Utilization** | `0.15` | Structured p5.brush media detector (`scaleBrushes`, `brush.fill`, `wash`, `bleed`) + instant zero-reward penalty on `text()` cheat attempts. |
| **5** | **Global Aesthetic** | `0.15` | Composition balance and harmony evaluated via `CLIPAestheticScorer` / `ImageReward`. |

---

## 3. Interactive Cyclic Continuous Training

```mermaid
sequenceDiagram
    autonumber
    actor User as Engineer / Researcher
    participant Trainer as PaintGRPOTrainer (grpo.py)
    participant GPU as Policy Model in VRAM
    participant Renderer as WebGL Sandbox (:3000)
    participant Composer as RewardComposer
    participant Dash as Dashboard (dashboard.html)

    Note over Trainer,GPU: Model & Tokenizer loaded once (permanently resident)

    loop Training Cycles (N steps per cycle)
        Trainer->>GPU: Generate Candidate Programs (G=4)
        GPU-->>Trainer: Completions
        Trainer->>Renderer: POST /render_batch (Completions)
        Renderer-->>Trainer: In-memory Base64 Render Results
        Trainer->>Composer: compute(renders, code, prompts)
        Composer-->>Trainer: Rewards + Diagnostic Scorecard
        Trainer->>GPU: Compute GRPO Loss & Backprop Optimizer Step
        Trainer->>Dash: Update Loss Curves & Artwork Gallery
        Trainer->>User: Print Diagnostic Scorecard & Cycle Summary
        
        alt Interactive Mode
            Trainer->>User: Prompt: Continue? [y / n / <number>]
            User-->>Trainer: User Response
        else Unattended Mode
            Note over Trainer: Continues autonomously to next cycle
        end
    end

    Trainer->>GPU: Save final LoRA adapter & safetensors
```

---

## 4. Hardware Sizing & Scaling Matrix

| Platform | Compute Device | Precision | Default Group ($G$) | `--max` Group ($G$) | Max Completion Tokens |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Apple Silicon M4 (16GB)** | `mps` (Metal) | `float32` | $4$ | $6$ | $448$ |
| **Apple Silicon M4 Max (32GB+)** | `mps` (Metal) | `float32` | $4$ | $8$ | $512$ |
| **Kaggle Dual Tesla T4 (30GB)** | `cuda:0` / `cuda:1` | `float16` | $4$ | $6$ | $450$ |
| **Cloud A100 GPU (40GB/80GB)** | `cuda` | `bfloat16` | $4$ | $8$ | $512$ |
| **Local PC / CI Environment** | `cpu` | `float32` | $2$ | $2$ | $256$ |

---

## 5. Architectural Documents & ADR Links
- [ADR-001: Headless Chromium WebGL Architecture](docs/decisions/ADR-001-puppeteer-webgl.md)
- [ADR-008: Multi-Signal Visual RL & Sandboxing](docs/decisions/ADR-008-multimodal-visual-rl-and-sandboxing.md)
- [ADR-009: Interactive Cyclic Training & Hardware Saturation](docs/decisions/ADR-009-interactive-cyclic-training-and-hardware-saturation.md)
- [Architectural Red-Team Analysis & Synthesis](docs/research/ARCHITECTURAL_REDTEAM_AND_SYNTHESIS.md)
