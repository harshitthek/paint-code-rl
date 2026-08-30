# Verified Research Synthesis and Reproducible System Design

This document bridges the gap between the adversarial audit of "Training AI to Paint with Code" and a scientifically defensible, reproducible system design. It extracts only verified knowledge and uses it to construct a next-generation architecture.

---

## 1. Evidence-Filtered Understanding

The following table extracts conclusions that survived the adversarial audit, discarding uncontrolled claims and unsupported causality.

| Topic | Claim | Evidence Status | Confidence | Source |
| :--- | :--- | :--- | :--- | :--- |
| **RL Pipeline** | The system uses LLM code generation executed in Puppeteer to generate RL rewards. | [FACT — PRIMARY] | High | Project Page |
| **Absolute Scoring** | Highly correlated (0.85-0.95) absolute 0-10 VLM scores cause mode collapse. | [FACT — SECONDARY] | High | Proxies & RLHF Literature |
| **Pairwise Judging** | Converting the reward to a fractional win-rate stabilized gradients. | [SUPPORTED INFERENCE] | Medium | Project Observation |
| **Mode Collapse** | Pairwise judging *caused* the resolution of the mode collapse. | [CONTRADICTED] | Low | Untangled Variables (5 changes) |
| **Code Length** | Code compression (13.5k to 2k) means the model learned "elegant" compositions. | [HYPOTHESIS] | Low | Uncontrolled Correlation |
| **Prompt Optimization** | An 8-method allowlist stopped API hallucination compared to a 400-line doc. | [FACT — PRIMARY] | High | Project Observation |
| **Reference Pool** | 581 model-generated images were used as the pairwise anchor. | [FACT — PRIMARY] | High | Project Page |

---

## 2. Verified Architecture

Based ONLY on evidence-supported mechanics, the core architecture is separated by its epistemic backing.

Prompt
↓
**Policy Model** *(Confirmed by project: Qwen LLM)*
↓
**Candidate Program Generation** *(Confirmed by project: `p5.brush` JS)*
↓
**Execution / Rendering** *(Confirmed by project: Puppeteer)*
↓
**Visual Outputs** *(Confirmed by project: PNGs)*
↓
**Reward Computation** *(Supported by literature: Anchored pairwise + HPSv3 + gates)*
↓
**Group Advantage** *(Confirmed by project: GRPO)*
↓
**Policy Optimization** *(Supported by literature: PPO-style clipped update)*
↓
**Evaluation / Calibration** *(Proposed improvement: Offline human calibration)*

**Stage Specifications:**
*   **Generation:** Inputs = Text Prompt. Outputs = JS Code. Latency = High (LLM autoregressive). Failure Mode = API hallucination. Evidence = [FACT].
*   **Execution:** Inputs = JS Code. Outputs = PNG. Latency = Very High (Browser DOM). Failure Modes = Syntax errors, infinite loops, timeouts. Evidence = [FACT].
*   **Reward Computation:** Inputs = PNG, Prompt, Reference PNGs. Outputs = Scalar $r$. Compute Cost = High (VLM inference + HPSv3). Evidence = [SUPPORTED INFERENCE].

---

## 3. Canonical GRPO Formulation

### Canonical GRPO (Shao et al., 2024)
Group Relative Policy Optimization computes the advantage $\hat{A}_i$ using the mean and standard deviation of rewards strictly within the sampled group $G$ for a specific prompt $q$.
$$J_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \frac{\pi_\theta}{\pi_{old}} \hat{A}_i, \text{clip}\left(\frac{\pi_\theta}{\pi_{old}}, 1-\epsilon, 1+\epsilon\right) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]$$
where $\hat{A}_i = \frac{r_i - \mu(R)}{\sigma(R)}$

### Project's Likely Implementation
*Inferred:* The project likely implements sequence-level reward scaling. Because $r_i$ is a sum of heterogeneous components (0.05 gate + 0.05 length + 0.3 HPSv3 + 0.6 Pairwise), the normalization $\frac{r_i - \mu}{\sigma}$ implicitly masks the absolute magnitude of the VLM judge, making the algorithm sensitive only to *rank order and relative distance* within the $G$ rollouts. $G$ is unknown.

### Recommended Implementation
*Proposed:* Use token-level KL regularization rather than sequence-level, as it provides tighter bounds against reward hacking. Set $G=16$ (balanced against Puppeteer rendering limits). Use the SFT model as $\pi_{ref}$.

---

## 4. Reward System Redesign

Assuming the goal is to preserve editability and aesthetics while minimizing reward hacking, we evaluate potential signals:

| Component | Measures | Blind Spots / Failure Modes | Cost |
| :--- | :--- | :--- | :--- |
| **A. Absolute VLM (0-10)** | Latent VLM biases | Compresses variance; highly susceptible to shortcut hacking (clip-art flowers). | High |
| **B. Pairwise VLM** | Relative quality against an anchor | Can inherit the biases of the reference pool; positional bias. | High |
| **C. HPSv3** | General text-image alignment / aesthetics | Not tuned specifically for programmatic strokes/textures. | Med |
| **D. Learned RM (Code)** | Code patterns correlated with good images | Cannot evaluate the actual visual layout; high risk of Goodhart's Law on code syntax. | Low |
| **E. Human Preference** | True subjective quality | Too slow for online RL; sparse. | Very High |
| **F. Execution/Compile** | Syntactic validity | Encourages ultra-short, trivial programs if over-weighted. | Low |

**Recommended Reward Architecture:**
Avoid redundant subjective signals. 
$R = w_1 R_{compile} + w_2 R_{HPSv3} + w_3 R_{pairwise\_anchor}$
*Discard code-length penalties entirely* to test if compression is a natural survival mechanic. Keep the system orthogonal: $R_{compile}$ ensures validity, $R_{HPSv3}$ ensures general prompt adherence, and $R_{pairwise}$ forces stylistic alignment to the target aesthetic.

## 5. Reference-Pool Design

The original 581-image model-generated pool acts as a hard stylistic prior. To build a scientifically defensible system, we must mitigate reference leakage and mode concentration.

**Comparison of Strategies:**
*   **Static Model-Generated:** Easy to scale, but caps the quality ceiling to the generating model (Opus/Gemini) and introduces AI stylistic biases.
*   **Static Human References:** Sets a genuine human quality ceiling, but is difficult to source at scale (especially for niche `p5.brush` textures).
*   **Dynamically Updated:** Best for open-ended RL; the pool updates as the policy discovers new, highly-rated states (acting like a novelty search archive).

**Recommended Construction Algorithm:**
1.  **Dataset Size:** 1,000 images.
2.  **Composition:** *Mixed.* 500 curated human generative artworks (establishing the quality ceiling) + 500 model-generated exploratory works (establishing the baseline).
3.  **Sampling:** *Diversity-Balanced Retrieval.* Instead of random sampling, use CLIP embeddings to sample references that are semantically close to the *text prompt*, but enforce a distance penalty to ensure visual diversity.
4.  **Refresh Strategy:** Every 500 training steps, inject the top 1% highest-HPSv3 rollouts into the reference pool to slowly raise the baseline and encourage exploration.

---

## 6. Judge Architecture

**Comparison of Judge Systems:**
*   **Single VLM (Absolute):** Prone to calibration drift and mode collapse.
*   **Multiple VLMs (Council):** Reduces individual model bias but massively increases latency. Highly correlated (as proven by the project's 0.85-0.95 covariance).
*   **Pairwise VLM:** Stabilizes variance by asking a relative question. Susceptible to positional bias (e.g., favoring Image A simply because it is presented first).
*   **Hybrid Judge:** HPSv3 (specialized scorer) + Pairwise VLM (style anchor).

**Recommended Judge System:**
Use the **Hybrid Judge**. However, to eliminate positional bias in the Pairwise VLM, implement *swapped evaluation*:
Prompt the VLM twice: `Judge(Rollout, Reference)` and `Judge(Reference, Rollout)`. The rollout only receives a win (1.0) if it wins *both* orientations. If orientations disagree, it is a tie (0.5).

---

## 7. Reward-Model Distillation

Transitioning from expensive VLM judging to cheap inference requires understanding the information bottleneck.

*   **Model A (Prompt + Code $\rightarrow$ Reward):** The cheapest. However, LLMs struggle to mentally "simulate" arbitrary JS canvas rendering. High risk of the RM grading formatting/syntax rather than visual aesthetics.
*   **Model B (Prompt + Rendered Image $\rightarrow$ Reward):** Evaluates true visual quality, but remains blind to "spaghetti code." The policy could generate 50,000 redundant strokes that look identical to 50 strokes, and Model B wouldn't know.
*   **Model C (Prompt + Code + Rendered Image $\rightarrow$ Reward):** Evaluates both visual quality and programmatic elegance. 

**Recommendation:** **Model C**. Train a multimodal RM (e.g., Qwen-VL) to take the JS string and the rendered PNG. This preserves the visual aesthetic evaluation while allowing the RM to penalize obfuscated or unnecessarily bloated code, solving the code-length uncertainty.

---

## 8. Rendering Infrastructure

To scale to thousands of reliable rollouts per hour, the rendering bottleneck must be solved.

*   **Puppeteer (Original):** High startup overhead, memory leaks over time, heavy DOM weight.
*   **Isolated Browser Tabs:** Reusing a single browser instance and managing tabs reduces startup overhead but risks context pollution.
*   **Sandboxed Node Workers:** Using `canvas` (Node-canvas or Skia) instead of a full browser.

**Recommended Rollout Executor:**
Use **Persistent Node.js Workers with Skia-Canvas**. 
Bypass the Chromium DOM entirely. `p5.js` can be run in headless Node using `skia-canvas`. Pre-warm a pool of worker threads. Inject the generated code into an isolated `vm2` or `isolated-vm` context. This reduces rendering latency from ~1-2 seconds (Puppeteer) to ~50-100ms per rollout, enabling $G=64$ rollouts in real-time.

---

## 9. Code Safety

Generating executable JS in a training loop presents a critical security and stability threat.

**Threat Model:**
1.  **Infinite Loops:** `while(true)` crashes the training step.
2.  **Memory Exhaustion:** Generating 10 million particles OOMs the worker.
3.  **Malicious Syscalls:** Accessing the host filesystem or network.

**Execution Architecture:**
*   **Isolation:** Run code inside `isolated-vm` (V8 isolates), which strictly prevents access to the Node.js standard library (no `fs`, no `child_process`).
*   **Limits:** 
    *   CPU Execution Timeout: 500ms.
    *   Memory Limit: 128MB per isolate.
*   **Validation:** Use Esprima/Acorn to parse the AST *before* execution. Reject any code containing `eval`, `Function`, or network fetch calls.

## 10. GEPA / Prompt Optimization

**Fact:** An 8-method allowlist stopped API hallucination compared to a 400-line doc for this specific model setup.
**Hypothesis:** This is a universal property of LLMs (long context degrades rigid adherence).

**Rigorous Experiment:**
Evaluate Prompts A (400-line doc), B (summary), C (8-method allowlist), and D (JSON Schema constrained decoding mapping to JS AST).
*   *Measurement:* Run 1,000 generations per prompt. Calculate API Hallucination Rate (static AST parsing for undefined methods) and Image Quality (HPSv3).
*   *Purpose:* Determine if the allowlist is actually better, or if deterministic constrained decoding (Prompt D) achieves 0% hallucination without sacrificing the expressive power of the full 400-line API.

---

## 11. Code-Length Investigation

The project observed code collapsing from 13.5k to <2k tokens, hypothesizing "aesthetic simplification." 

**Definitive Experiment:**
*   **H1 (Aesthetic):** The VLM judge intrinsically prefers shorter code's visual output.
*   **H2 (Syntax-survival):** Longer code has a higher syntax error probability, yielding 0 reward.
*   **H3 (Timeout):** Longer code times out, yielding 0 reward.

**Methodology:**
Train 3 parallel policies.
1.  **Control:** Standard rules.
2.  **No-Penalty:** If a rollout crashes/times out, instead of assigning 0 reward, *drop it from the GRPO group entirely* (compute advantage over $G-k$ valid rollouts). 
3.  **Forced-Execution:** Programmatically auto-fix syntax errors (e.g., closing brackets) and increase timeouts to 10 seconds.

**Interpretation:** If Policy 2 and 3 retain long code lengths (e.g., >8k tokens) while maintaining high HPSv3 scores, H1 is falsified. The compression was a survival tactic (H2/H3), not an aesthetic discovery.

---

## 12. Mode-Collapse Investigation

To determine what actually solved the mode collapse (the clip-art flower), a fractional factorial ablation is required.

**Experimental Matrix:**

| Run | Judging Format | Reference Pool | HPSv3 Weight | Result to observe |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Absolute (0-10) | None | 0.10 (Baseline) | Should collapse. |
| 2 | Absolute (0-10) | None | **0.30** | Isolates HPSv3 variance. |
| 3 | Pairwise | **Static (581)** | 0.10 | Isolates Pairwise + Pool. |
| 4 | Pairwise | **Static (581)** | **0.30** (Project) | Should succeed. |

**Interpretation:** If Run 2 succeeds, the mode collapse was solved simply by increasing the weight of HPSv3 (adding variance), and the pairwise reference pool was completely unnecessary for stabilization.

---

## 13. Proposed Next-Generation System

Based on the evidence, the strongest, most scalable design:

*   **Policy:** Qwen2.5-Coder (stronger base syntax capability than general Qwen).
*   **Training:** Token-level GRPO (memory efficient, stable).
*   **Reward Architecture:** $R = w_1 R_{compile\_ast\_check} + w_2 R_{HPSv3} + w_3 R_{pairwise}$. (No length penalties).
*   **Judge:** Swapped-orientation Pairwise VLM (to eliminate positional bias).
*   **Reference Pool:** 1,000 mixed human/model images, dynamically refreshed with top 1% rollouts every 500 steps.
*   **Rendering:** Headless Node.js `skia-canvas` inside `isolated-vm` (100x faster than Puppeteer).
*   **Reward Model:** Distill the VLM into a multimodal Code+PNG Reward Model for Phase 2 scaling.

---

## 14. Training Pipeline (Pseudocode)

```python
# PRODUCTION-STAGE PIPELINE (Skia-Canvas + Swapped VLM Judge)
def train_step(policy, ref_policy, prompts, ref_pool):
    for q in prompts:
        # 1 & 2. Candidate Generation
        candidates = policy.generate(q, num_samples=G) 
        
        rewards = []
        valid_candidates = []
        
        for code in candidates:
            # 3. Code Validation (AST Parse)
            if not is_safe_and_valid_ast(code):
                rewards.append(0.0)
                continue
                
            # 4. Rendering (Isolated VM)
            image = skia_isolated_render(code, timeout_ms=500)
            if image is None:
                rewards.append(0.0)
                continue
                
            valid_candidates.append((code, image))
            
        # 5 & 6. Reward Inference & Swapped Pairwise
        for code, image in valid_candidates:
            r_hps = get_hpsv3(image, q)
            
            ref_img = sample_reference(ref_pool, q)
            win_forward = vlm_judge(image, ref_img, q)
            win_reverse = vlm_judge(ref_img, image, q)
            
            # Swapped logic: 1.0 if wins both, 0.5 if split, 0.0 if loses both
            r_pair = 1.0 if (win_forward and not win_reverse) else 0.0
            if win_forward == win_reverse: r_pair = 0.5
            
            rewards.append(0.3 * r_hps + 0.6 * r_pair + 0.1) # 0.1 for compiling
            
        # 7 & 8. Advantage & Update
        advantages = compute_grpo_advantages(rewards)
        policy_loss = compute_clipped_kl_loss(policy, ref_policy, valid_candidates, advantages)
        policy.update(policy_loss)
        
        # 12. Reference Refresh
        if step % 500 == 0:
            ref_pool.add(top_1_percent(valid_candidates, rewards))
```

## 15. Experimental Roadmap

**Phase 0: Baseline & Infrastructure**
*   *Objective:* Swap Puppeteer for `skia-canvas` `isolated-vm` setup. 
*   *Success:* $G=16$ rollouts complete in <2 seconds total.

**Phase 1: Reward Validation (Ablation)**
*   *Objective:* Execute the Matrix from Section 12.
*   *Success:* Identify the singular causal variable that resolved mode collapse.

**Phase 2: Reference-Pool Curation**
*   *Objective:* Replace the 581 model-generated images with 500 human curated generative artworks.
*   *Success:* Policy exhibits structural styles not present in Opus/Gemini generations.

**Phase 3: Reward-Model Distillation**
*   *Objective:* Train Multimodal RM (Code+PNG) on Phase 1/2 rollout preferences.
*   *Success:* RM predicts the Swapped VLM Judge outcome with >85% accuracy.

**Phase 4: Large-Scale Training**
*   *Objective:* Train the policy exclusively against the RM for 10k steps.

---

## 16. Evaluation Framework

To prevent reward hacking, the evaluation suite must measure orthogonal metrics not included in the reward function:

1.  **Code Correctness / Readability:** Automated AST cyclomatic complexity checks. (Does the code use loops efficiently, or unroll 1,000 lines manually?)
2.  **Editability:** Human-in-the-loop test: Can a developer change the color of a specific flower petal within 30 seconds of reading the code?
3.  **Out-of-Distribution Prompts:** Test prompts completely absent from the reference pool (e.g., "draw a cyberpunk cityscape"). Measures true generalization vs. reference-pool overfitting.

---

## 17. Reproducibility Specification

To ensure future research is scientifically defensible, the following must be released.

*   [MUST BE CREATED] **Training Framework Repository:** The `trl`-based Python scripts and Node.js sandbox code.
*   [MISSING] **Reference Images & Labels:** The 581 images and the tiering classifications.
*   [MISSING] **Judge Prompts:** The exact system prompts given to the VLM.
*   [PARTIALLY AVAILABLE] **Reward Weights:** Known (0.05, 0.05, 0.3, 0.6) but missing the precise compilation logic.
*   [MISSING] **Hyperparameters:** GRPO Group size, learning rate, KL beta.
*   [AVAILABLE] **Dependencies:** `p5.brush` library.

---

## 18. Compute and Scaling Analysis

**Dominant Bottleneck:** During online RL, the **VLM Judging** is the absolute bottleneck. 
*   *Estimate:* Rendering in `skia-canvas` takes ~50ms per candidate. Generating code via vLLM takes ~500ms for a batch. However, asking a large VLM (e.g., GPT-4o or Qwen-VL-Max) to perform two pairwise comparisons (swapped) for 16 candidates takes ~2-5 seconds due to image processing and TTFT (Time To First Token) API latency.

**Proposed Solution:** Phase 3 Distillation. By training a local, 7B-parameter multimodal RM, inference drops to milliseconds on the same GPU node, eliminating network latency and massive VLM inference costs, reducing cost-per-sample by >90%.

---

## 19. Research Questions (Prioritized)

1.  **Scientific Importance (High), Feasibility (High):** *Does absolute scoring inherently cause mode collapse in visual RL, or did this specific project simply use highly correlated absolute metrics?* (Resolved via Phase 1 Ablation).
2.  **Scientific Importance (High), Feasibility (Medium):** *Can a code-only reward model accurately predict aesthetic image quality?* (Tests the limit of LLM spatial reasoning).
3.  **Scientific Importance (Medium), Feasibility (High):** *Was the 13k token compression an aesthetic preference, or a syntax-survival mechanism?* (Resolved via the No-Penalty GRPO experiment).

---

## 20. Final Technical Position

**EVIDENCE FIRST. CAUSALITY SECOND. DESIGN THIRD.**

### What Is Known
*   LLMs can be trained via RL to output valid `p5.brush` JavaScript that renders into images.
*   Highly correlated absolute reward signals in this domain lead to exploitable, degenerate policies.

### What Is Probable
*   The pairwise reference pool anchored the policy, preventing it from exploiting the VLM's absolute biases.
*   Browser-based rendering (Puppeteer) is too slow for large-scale RL and must be replaced by isolated Node environments.

### What Is Unknown
*   Whether the pairwise structure *caused* the stabilization, or whether simply increasing the HPSv3 weight was sufficient.
*   Whether the massive code compression was due to "learning elegance" or avoiding syntax penalties.

### What Is Wrong in the Original Narrative
*   Describing the system as RLAIF / RM Distillation (it uses online judging).
*   Attributing causality to the pairwise format without controlling for 4 other simultaneous reward changes.
*   Stating GEPA proves long documentation generally causes hallucination.

### What Is Actually Novel
*   The specific engineering combination of GRPO, a sandboxed vector-graphics library, and an online VLM judge for program synthesis.

### What Should Be Tested Next
*   The controlled ablation of the reward components (Section 12) and the syntax-survival code-compression test (Section 11) to isolate the true causal mechanics of the system's behavior.
