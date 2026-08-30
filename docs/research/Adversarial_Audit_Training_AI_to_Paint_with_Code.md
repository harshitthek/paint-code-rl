# Adversarial Audit: Training AI to Paint with Code

This document provides a rigorous, skeptical re-evaluation of the claims made in the initial research dossier for "Training AI to Paint with Code." It aggressively separates canonical facts from inferred implementations, uncontrolled observations, and speculative hypotheses.

---

## 1. Claim Audit

| Claim from Previous Dossier | Verification Verdict | Epistemic Status | Explanation |
| :--- | :--- | :--- | :--- |
| Project uses canonical GRPO | **PARTIALLY TRUE** | [SUPPORTED INFERENCE] | The project claims to use GRPO, but the exact hyperparameters and clipping values are not public. |
| The system uses RLAIF / Reward Model Distillation | **INCORRECT** | [CONTRADICTED] | The project explicitly states that training a reward model was a "next step, which we did not get to." The current system uses an online VLM judge, not a distilled RM. |
| "Pairwise judging solved mode collapse" | **UNSUPPORTED CAUSALITY** | [HYPOTHESIS] | The rubric change simultaneously altered the reference pool, the components (9 down to 4), and HPSv3 weighting (0.1 to 0.3). Causality cannot be isolated to the pairwise format alone. |
| "Model learned winning compositions didn't need verbose code" | **WEAKLY SUPPORTED** | [HYPOTHESIS] | The token drop (13.5k to <2k) correlates with higher reward, but may simply be a survival mechanic against syntax errors and execution timeouts. |
| The Pairwise Reward is an Elo/Thurstone model | **INCORRECT** | [FACT - SECONDARY SOURCE] | The system merely computes a raw win-rate against two random references (`wins / 2.0`). It does not update an Elo rating or fit a Bradley-Terry probability distribution. |
| GEPA optimization stopped API hallucination | **TRUE, BUT CONTEXTUAL**| [FACT - PRIMARY SOURCE] | The project observed that a GEPA-evolved prompt (an 8-method allowlist) stopped hallucinations. Whether this proves "long documentation intrinsically causes hallucination" is an overgeneralization. |

---

## 2. Corrections to the Previous Dossier

1.  **RLAIF Terminology:** The previous report improperly labeled the reference pool setup as Reward Model Distillation. The system uses an *online, inference-time VLM judge* comparing rollouts to static references. Distillation would imply training a smaller scalar model on those references, which the authors explicitly noted they did not do.
2.  **HPSv3 Attribution:** The previous report failed to correctly identify HPSv3. It is authored by Ma et al. (ICCV 2025), utilizing the HPDv3 dataset of 1.08M text-image pairs.
3.  **GEPA Attribution:** GEPA (Genetic-Pareto) is a 2025 framework by Agrawal et al. (Stanford/UC Berkeley). The previous report did not adequately explain that GEPA relies on *reflective LLM evaluation* to mutate prompts rather than just random search.
4.  **Elo/Ranking Terminology:** The reward function is a simple fractional win-rate (r = wins/2), not a true pairwise preference learning algorithm (like DPO) or a Bradley-Terry model.

---

## 3. Missing Research (Newly Investigated)

*   **DeepSeekMath (Shao et al., 2024):** The foundational GRPO paper. It establishes that GRPO eliminates the value network by computing advantage solely relative to the responses generated from the *same* prompt.
*   **Proxy Compression Hypothesis (Gao et al., 2022):** The phenomenon where a complex goal (aesthetic art) is compressed into a scalar (VLM judge score), inevitably leading to the policy exploiting the compression (the clip-art flower).
*   **Visual RL Constraints:** Literature confirms that programmatic art generation via RL heavily penalizes long token lengths simply because the joint probability of a syntax error increases exponentially with sequence length.

---

## 4. Revised Architecture

*Note: This architecture carefully distinguishes documented facts from inferences.*

1.  **Generation [FACT]:** LLM (Qwen) receives a prompt and generates a `p5.brush` JavaScript string.
2.  **Execution [FACT]:** The JS is executed in Puppeteer.
3.  **Reward Computation [FACT]:**
    *   R_gate: Binary (compiles and uses brush) [Weight: 0.05]
    *   R_len: Binary length check [Weight: 0.05]
    *   R_hps: HPSv3 score [Weight: 0.30]
    *   R_pair: Win-rate against 2 random references from the 581-image pool, judged by a VLM. [Weight: 0.60]
4.  **Advantage Calculation [SUPPORTED INFERENCE]:** Using canonical GRPO, A_i = (r_i - mean(R)) / std(R) across G rollouts.
5.  **Policy Update [SUPPORTED INFERENCE]:** Clipped PPO-style update with a KL divergence penalty against the reference (SFT) model.

---

## 5. Causal Analysis: The Mode Collapse Resolution

**The Claim:** Shifting to pairwise comparison and a reference pool solved the mode collapse (the generic clip-art flower).
**The Reality:** The authors changed five variables simultaneously between the failed run and the successful run:
1.  Dropped 5 correlated subjective judges.
2.  Dropped the GPT-5.4/Gemini prompt-adherence council.
3.  Increased HPSv3 weight from 0.10 to 0.30.
4.  Introduced the 581-image reference pool.
5.  Changed the VLM prompt from absolute (0-10) to pairwise ("which is better?").

**Verdict:** [UNSUPPORTED CAUSALITY]. It is impossible to definitively state that pairwise judging solved the mode collapse. Increasing HPSv3's weight by 3x could have provided the necessary variance. Alternatively, dropping the highly correlated absolute judges simply removed the dominant, exploitable gradient.

## 6. Novelty Audit

A rigorous comparison with prior work in visual RL and code generation.

| Component | Prior Work? | Exact Precedent | What This Project Adds |
| :--- | :--- | :--- | :--- |
| **RL for Programmatic Art** | Yes | SPIRAL (2018), PaintBot | Extends vector/brush strokes to arbitrary JavaScript using LLMs instead of CNNs. |
| **VLM-as-a-Judge** | Yes | GPT-4V evaluation papers | Applies it dynamically inside an RL loop for raw code output. |
| **Pairwise Win-Rate Reward** | Yes | Standard RLHF pipelines | Uses a *static, external reference pool* for the comparison rather than comparing two actively generated policies against each other. |
| **Multimodal GRPO** | Yes | DeepSeekMath (text), emerging VLM GRPO | One of the earliest documented applications of GRPO to generative visual code. |

**Verdict:** [SUPPORTED INFERENCE] The project is not fundamentally inventing new algorithms; it is a novel *engineering combination* of GRPO, p5.brush, and static-reference VLM evaluation.

---

## 7. Reproducibility Audit

A deeper search was conducted for the project's GitHub, configs, and training code.

*   **GitHub Repositories:** Neither Surya Narreddi nor Cameron Franz have publicly published the training framework for this specific project. Narreddi’s other open-source work (e.g., Gossip, RuneBench) is available, but the "Paint with Code" repo is hidden/unreleased.
*   **Technical Report:** The blog post mentions a "full technical report will be published in June 26." As of August 2026, extensive web and arXiv searches reveal no such report.
*   **Missing Variables:** Exact Qwen version (e.g., 1.5, 2.0, or Coder), GRPO group size (G), learning rate, and the 581-image reference dataset are entirely missing.
*   **Verdict:** [UNVERIFIED]. The system currently has **LOW reproducibility**. It cannot be reconstructed exactly without the reference pool and the specific VLM judge prompt configurations.

---

## 8. Research Gaps

The strongest unanswered questions remaining from this project:
1.  **Does code length actually correlate with aesthetic quality?** Or is length reduction purely an artifact of syntax-error avoidance?
2.  **What is the selection bias of the 581-image pool?** Since the pool was generated by Opus/Gemini, does the Qwen policy merely learn the latent stylistic biases of Opus rather than true artistic quality?
3.  **Does the VLM judge actually evaluate code?** Or does it only evaluate the PNG? If the latter, it remains blind to "spaghetti code."

---

## 9. Recommended Experiments (Ablations)

To isolate causality, the following controlled experiments are necessary:

1.  **Code Compression Falsification:**
    *   *Design:* Train a policy where syntax errors do *not* result in a 0 reward (e.g., render whatever parses, ignore the rest).
    *   *Goal:* Distinguish whether code compression is an aesthetic choice or a survival mechanic.
2.  **Absolute vs. Pairwise Falsification:**
    *   *Design:* Keep the 4-component rubric and the 0.30 HPSv3 weight, but revert the VLM judge back to an absolute 0-10 scale.
    *   *Goal:* Prove whether the pairwise *format* resolved the mode collapse, or whether it was simply dropping the 5 redundant judges.
3.  **Reference Pool Ablation:**
    *   *Design:* Train one policy against the 581 model-generated images, and another against 581 curated human-made p5.js artworks.
    *   *Goal:* Measure how heavily the policy overfits to the reference pool's latent biases.
4.  **Reward Model Design:**
    *   *Design:* Compare (A) Code-only Reward Model, (B) Rendered PNG Reward Model, (C) Multimodal Code+PNG Reward Model.
    *   *Goal:* Determine if a code-only RM can predict aesthetic quality well enough to bypass the Puppeteer bottleneck.

---

## 10. Revised Final Assessment

The initial report presented the project's narrative somewhat uncritically. This adversarial audit scales back several causal claims. 

The project is a strong **engineering demonstration** of applying GRPO to visual code generation. However, its research conclusions—specifically that pairwise judging "solved" mode collapse, that the model "learned" elegant code compression, and that GEPA proves long documentation intrinsically causes hallucination—are **unsupported by controlled ablation**. Multiple variables were changed simultaneously between training runs, making causal attribution impossible. Furthermore, the reliance on a static, model-generated reference pool strongly implies that the system is performing a localized form of style distillation rather than generalized artistic learning.

**What We Can Reliably Conclude:**
1. [FACT] Standard absolute VLM scoring (0-10) across highly correlated subjective metrics results in rapid reward hacking and mode collapse in this architecture.
2. [FACT] Puppeteer rendering inside an active RL loop presents a severe computational bottleneck.
3. [SUPPORTED INFERENCE] Replacing a 400-line API document with a strict 8-method allowlist reduces API hallucination for this specific Qwen model.
4. [SUPPORTED INFERENCE] Anchoring a VLM judge to a static reference pool successfully stabilizes the RL gradient, preventing the policy from drifting into unconstrained mode collapse.

---

## Confidence Table

| Conclusion | Evidence Strength | Causal Confidence | Main Remaining Uncertainty |
| :--- | :--- | :--- | :--- |
| Absolute scoring causes mode collapse | Strong (Empirical observation) | High | None; well-supported by RL literature. |
| Pairwise judging *solved* mode collapse | Weak (Uncontrolled variable changes) | Low | Did HPSv3 up-weighting actually solve it instead? |
| Code length reduction is aesthetic | Weak (Correlation only) | Low | Is it just avoiding syntax/timeout errors? |
| GEPA allowlist stops API hallucination | Strong (Primary evidence) | Medium | Does this hold for models with longer context windows? |
| Reference pool stabilizes gradients | Medium (Theoretical + Empirical) | High | Does it overfit to the pool's biases? |
