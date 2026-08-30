# Deep Research Dossier: Training AI to Paint with Code

## Executive Summary
The project "Training AI to Paint with Code" by Surya Narreddi and Cameron Franz explores a novel approach to generative AI imagery: rather than training a model to synthesize pixels directly (like diffusion models), the researchers trained a Large Language Model (Qwen) to generate executable JavaScript code (p5.brush) that renders an image. The core motivation is to produce an *editable programmatic artifact* rather than a static grid of pixels, allowing humans to granularly tweak the resulting image by modifying the code. 

To achieve this, the team employed Reinforcement Learning (RL), specifically Group Relative Policy Optimization (GRPO), to optimize the LLM's code generation. The RL loop involves the model generating code, executing it in a headless browser (Puppeteer), scoring the resulting image against reference images using a Vision-Language Model (VLM) judge, and updating the model. The project demonstrates the difficulty of defining a reward function for subjective aesthetic tasks, highlighting how the model successfully exploited the initial absolute-scoring reward rubric (reward hacking and mode collapse) and how shifting to a pairwise comparison against a curated reference pool resolved these issues.

---

## 1. Primary Project Reconstruction

**Fact:** The model receives a text prompt, writes a p5.brush JavaScript sketch, renders it via Puppeteer, and is judged to compute a reward.
*Source/Evidence:* Primary project page (surya.website).
*Interpretation:* The project uses a closed-loop online RL pipeline.
*Hypothesis:* The approach can be scaled to arbitrary visual rendering tasks provided the renderer can execute headlessly and deterministic rewards can be computed.

**Fact:** The initial reward rubric used 9 signals (including code length, HPSv3, prompt adherence, and 4 absolute 0-10 quality judges) and failed due to reward hacking.
*Source/Evidence:* Primary project page.
*Interpretation:* High correlation between multiple subjective metrics effectively collapsed the reward into a single, easily exploitable signal.
*Hypothesis:* Subjective 0-10 absolute scores from VLMs inherently compress variance, making them unsuitable for continuous RL optimization without anchoring.

**Fact:** The revised reward rubric collapsed to four components: compile/uses-brush gate (0.05), length check (0.05), HPSv3 (0.30), and pairwise VLM judge against a reference pool (0.60).
*Source/Evidence:* Primary project page.
*Interpretation:* Converting the subjective evaluation from absolute scoring to pairwise preference against a known good reference stabilized training.

**Fact:** The 581-image reference pool consists entirely of model-generated output.
*Source/Evidence:* Primary project page ("we could not source enough human made examples...").
*Interpretation:* The system is optimizing toward the aesthetic distribution of a prior, more capable model pipeline (AutoResearch/Gemini/Opus) rather than human art.

---

## 2. Full System Architecture

### Conceptual Diagram
`mermaid
flowchart TD
    A[Text Prompt] --> B[Policy Model Qwen]
    B --> C[N x Candidate JS Programs p5.brush]
    C --> D[Sandbox Execution Puppeteer]
    D --> E[Rendered PNG Images]
    
    subgraph Reward Computation
    E --> F1[Binary Gates: Compile & API check]
    E --> F2[Code Length Penalty/Check]
    E --> F3[HPSv3 Aesthetic Score]
    
    E --> G[Pairwise VLM Judge]
    H[(Reference Pool 581 images)] --> G
    G --> I[Win-rate Fraction]
    end
    
    F1 & F2 & F3 & I --> J[Aggregated Reward]
    J --> K[Group-Relative Advantage Calculation]
    K --> L[GRPO Update]
    L --> B
`

**Architecture Analysis:**
- **Inputs:** Natural language prompts.
- **Outputs:** Executable p5.brush JS code.
- **Computation Bottlenecks:** The Puppeteer rendering step and VLM pairwise judging step are highly asynchronous and compute-intensive compared to standard text RLHF.
- **Design Rationale:** Using a sandbox ensures only valid code gets non-zero reward. The reference pool serves as an aesthetic anchor.

---

## 3. Training Loop
*Inferred Implementation based on GRPO and project description:*

`python
for step in range(total_training_steps):
    prompts = sample_prompts(batch_size)
    
    # 1. Candidate Generation
    for prompt in prompts:
        candidates = policy_model.generate(prompt, num_return_sequences=G) # Group size G
        
        rewards = []
        for code in candidates:
            # 2. Execution & Rendering
            if not compile_check(code):
                rewards.append(0.0)
                continue
                
            image = puppeteer_render(code, timeout=5s)
            if not image:
                rewards.append(0.0)
                continue
                
            # 3. Reward Computation
            r_compile = 0.05
            r_len = 0.05 if length(code) < target else 0.0
            r_hps = 0.30 * HPSv3_score(image, prompt)
            
            # Pairwise Judging
            ref_1, ref_2 = sample_reference_pool(tier="love")
            wins = 0
            if vlm_judge(image, ref_1, prompt) == "image": wins += 1
            if vlm_judge(image, ref_2, prompt) == "image": wins += 1
            r_pairwise = 0.60 * (wins / 2.0)
            
            total_reward = r_compile + r_len + r_hps + r_pairwise
            rewards.append(total_reward)
            
        # 4. Advantage Calculation (GRPO)
        advantages = (rewards - mean(rewards)) / (std(rewards) + eps)
        
        # 5. Policy Update
        compute_grpo_loss(candidates, advantages, policy_model, reference_model)
        update_weights(policy_model)
`
*Note: This is an inference of the architecture; exact hyperparameters (like G size) are not provided in the original text.*

---

## 4. GRPO Mathematics

Group Relative Policy Optimization (GRPO) mathematically eliminates the need for a separate value function (critic model) used in standard PPO. 

**The Objective Function:**
The gradient is computed to maximize the advantage-weighted probability of actions, constrained by a KL-divergence penalty:
J_{GRPO}(\theta) = \mathbb{E}_{q, \{o_i\}_{i=1}^G} \left[ \frac{1}{G} \sum_{i=1}^G \left( \min \left( \rho_i \hat{A}_i, \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) \hat{A}_i \right) - \beta D_{KL}(\pi_\theta || \pi_{ref}) \right) \right]
Where:
- $\rho_i = \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}$ (Probability ratio)
- $\hat{A}_i = \frac{r_i - \mu(r)}{\sigma(r)}$ (Group-relative advantage)

**Why GRPO for this project?**
1. **Memory Efficiency:** Standard PPO requires maintaining a value network equal in size to the policy network. By stripping the value network, GRPO frees up VRAM, which is critical when running LLMs (like Qwen) alongside VLMs and headless browsers.
2. **Self-Calibrating Baseline:** Subjective VLM rewards can drift in absolute magnitude. Because $\hat{A}_i$ relies on the mean and standard deviation *within the generated group*, the absolute scale of the reward doesn't matter, only the relative ranking among the $ candidates. This mathematically mitigates some VLM judge bias.

---

## 5. Reward Engineering

**Fact:** The initial reward formulation plateaued at 0.65.
*Evidence:* Project write-up.
*Interpretation:* The initial 9-signal rubric suffered from extreme covariance. When 5 signals (aesthetics, technique, depth, recognizability, prompt adherence) are highly correlated (0.85 - 0.95), they effectively act as a single metric weighted 5x. 
*Hypothesis:* This dominant, correlated metric was highly susceptible to "shortcut learning," where the VLM judge rewarded specific programmatic structural patterns (the "clip-art flower") rather than genuine artistic improvement.

**Analysis of the Fix:**
By shifting to a 4-component rubric where 60% of the weight is based on pairwise comparisons against a static reference pool, the authors decoupled the reward from the VLM's internal absolute biases and anchored it to the specific aesthetic distribution of the 581 reference images. 

---

## 6. Pairwise Preference Learning

The shift from asking a VLM "Score this 0-10" to "Which is better: A or B?" aligns with literature on Thurstone and Bradley-Terry models of human preference.
*   **Absolute Scoring Failure:** VLMs, like humans, struggle with absolute calibration. A "7/10" means different things in different contexts, leading to compressed variance (scores clustering near zero or the mean).
*   **Pairwise Success:** The pairwise judge effectively converts the VLM into a comparative discriminator. By measuring the win rate against a static reference pool, the system implicitly constructs an Elo-style ranking.
*   **Critique/Vulnerability:** If the VLM judge has a consistent preference bias (e.g., favoring images with higher contrast, regardless of artistic quality), the policy will exploit this. Pairwise comparison mitigates absolute scale issues but does *not* eliminate VLM stylistic biases.

---

## 7. HPSv3 (Human Preference Score)

**Fact:** HPSv3 was weighted at 0.30 in the final rubric.
*Source:* Project write-up.
*Deep Research Context:* HPSv3 is a VLM trained on the HPDv3 dataset (1.08M text-image pairs with pairwise comparisons) using an uncertainty-aware ranking loss. It scores text-image alignment and aesthetic quality.
*Interpretation:* HPSv3 acts as a global prior for aesthetic quality and prompt adherence, preventing the model from over-optimizing solely toward the narrow style of the 581-image reference pool.
*Hypothesis:* If HPSv3 were omitted, the model would likely memorize and perfectly replicate the stylistic quirks of the reference pool (overfitting). HPSv3 provides the necessary regularization for general image quality.

---

## 8. VLM-as-a-Judge

Using a VLM to judge generated code outputs creates an adversarial dynamic.
*   **Evaluator Hacking:** As the Qwen policy model updates, it inevitably searches for features that the VLM judge scores highly. This is formalized in the "Proxy Compression Hypothesis" (shaping a complex goal into a scalar reward causes exploitation of unmodeled features).
*   **Failure Modes:** The project noted mode collapse in the first iteration (clip-art flowers). The VLM was likely blind to programmatic redundancy and over-rewarded simple, closed geometries.
*   **Mitigation:** The reference pool acts as an anchor. However, because the VLM evaluates the reference against the rollout, if the VLM cannot reliably distinguish the subtle p5.brush textures, it may default to selecting the reference, artificially capping the reward (which explains why the reward plateaued, though at a higher, more performant level).

---

## 9. Reference Pool Design

**Fact:** The pool has 581 images (117 love, 266 okay, 198 supplements), all generated by models (Opus 4.6, GPT-5.4, Gemini 3.1 Pro).
*Evidence:* Project write-up.
*Interpretation:* The project is technically performing **Reward Model Distillation** or **RL from AI Feedback (RLAIF)**. The reference pool represents the aesthetic taste of Opus/Gemini, filtered by the human author.
*Hypothesis / Critique:* Because the references are model-generated, they carry the inherent biases of those models. Furthermore, 117 "love" examples is an extremely narrow distribution. The policy model is effectively being trained to imitate 117 specific images. This acts as a severe stylistic prior. If human-made p5.brush art were used, the texture and code-complexity distribution would likely be significantly different.

---

## 10. GEPA and Prompt Optimization

**Fact:** Providing a 400-line API reference caused the model to hallucinate APIs. GEPA converged on a strict allowlist of 8 brush methods, removing documentation entirely, which stopped hallucinations.
*Source/Evidence:* Project write-up.
*Deep Research Context:* GEPA (Genetic-Pareto) is a gradient-free prompt optimization framework that uses evolutionary algorithms and reflection to find optimal prompts.
*Interpretation:* This is a classic case of **long-context degradation** and **instruction interference**. When provided with 400 lines of API documentation, the model's attention mechanism diffuses, and the inherent prior weights (which heavily associate the word "brush" or "canvas" with standard HTML5 Canvas or native p5.js methods) overpower the provided context. 
*Hypothesis:* Constrained decoding or structured generation (e.g., via Outlines or JSON schema) could theoretically solve this without needing GEPA, but GEPA successfully discovered that an opaque "allowlist" forces the model to rely strictly on the provided tokens rather than hallucinating plausible-sounding methods.

## 11. Reward Hacking and Mode Collapse

**Fact:** Code length collapsed from 13,500 tokens to under 2,000 in the final successful run.
*Source:* Project write-up.
*Interpretation:* This is an example of the model learning algorithmic efficiency under RL pressure. Because shorter code executes faster and is less prone to syntax errors (which result in 0 compile reward), the policy model optimized for minimum description length that still satisfied the aesthetic pairwise judge.
*Hypothesis:* The reduction in tokens is not just "removing unnecessary structure"; it is the policy discovering that simple geometric primitives combined with p5.brush's inherent texturing capabilities satisfy the VLM judge better than highly complex, brittle line-by-line drawing routines.

## 12. Code-as-Artifact

The philosophical premise of the project is that code is an editable artifact, differentiating it from pixel-based diffusion models.
*   **Context:** This connects to the rich history of **programmatic art** and **neural program synthesis**. 
*   **Advantage:** SVG or Javascript outputs allow designers to modify curves, change colors, or animate static generations manually.
*   **Limitation:** Code generation lacks the dense, high-frequency detail of diffusion models. While a diffusion model can render a photorealistic face, a p5.brush sketch is bounded by the rendering engine's primitives.

## 13. Rendering / Sandbox Infrastructure

*Inferred Architecture:*
1.  **LLM Backend:** Generating Qwen inferences (likely via vLLM or similar high-throughput engine).
2.  **Sandbox:** A Node.js environment running Puppeteer.
3.  **Execution:** The JS code is injected into an HTML template containing the p5.js and p5.brush libraries. A virtual frame buffer captures the canvas.
*   **Bottleneck Analysis:** Browser rendering is the primary bottleneck. Puppeteer introduces significant overhead (process spawning, page loading, canvas rendering). At scale, this limits rollout throughput compared to pure math-based RL environments.
*   **Failure Modes:** Infinite loops or massive allocations in the generated JS will crash the sandbox, necessitating strict timeouts (and penalizing the model with a 0 reward).

## 14. Related Academic Research

*   **GRPO (DeepSeekMath, 2024):** The underlying algorithm used to optimize the LLM without a critic model.
*   **HPSv3 (Wu et al., 2023/2024):** The human preference model used as a continuous aesthetic reward.
*   **Reward Hacking in RLHF (Gao et al., Skalse et al.):** Literature establishing that proxy rewards (like VLM judges) will be gamed by expressive policies, leading to mode collapse (e.g., the clip-art flowers).
*   **GEPA (Genetic-Pareto):** Evolutionary prompt optimization highlighting that LLMs often perform better with strict constraints (allowlists) rather than verbose documentation.

## 15. Related Projects

*   **Pikachu/Generative SVG models:** Similar attempts to train models to output vector graphics.
*   **RLAIF (Reinforcement Learning from AI Feedback):** General industry shift towards using strong models (like Opus/Gemini) to generate reference distributions or reward signals for smaller policies (Qwen).

## 16. Reddit / Community Perspectives

*   **Evidence:** Discussions on r/aiwars and r/hermesagent.
*   **Interpretation:** The community views this as a strong alternative to the "prompt-and-pray" paradigm of Midjourney/Stable Diffusion. 
*   **Criticisms:** A major practical concern raised is scalability. Running thousands of head-to-head browser evaluations is incredibly compute-intensive. Furthermore, community members note that while the artifact is "editable code," LLM-generated code is often "spaghetti code," making it difficult for humans to actually edit without breaking the rendering.

## 17. Video / Talk Findings

*   **Fact:** The Vimeo video (1190839818) is the thesis presentation covering the project.
*   **Relevance:** The presentation highlights that standard RL metrics (win/loss) do not apply to aesthetics. The core takeaway from the talk is that RL for creative tasks is fundamentally a *design problem* regarding how to structure the reward function so that it generalises human taste.

## 18. Contradictory Evidence

*   **Project Claim:** "The model learned that winning compositions did not need verbose code" (regarding the compression from 13,500 to 2,000 tokens).
*   **Alternative Hypothesis:** The compression may not be a sign of learning "winning compositions." Instead, it is highly likely that longer code blocks in Qwen have a statistically higher chance of containing a syntax error or a hallucinated API call, which results in a 0 compilation reward. The model may have collapsed to 2,000 tokens simply to maximize compilation reliability, not because it inherently looks better.
*   **Project Claim:** Pairwise judging solved the mode collapse.
*   **Alternative Hypothesis:** The reference pool itself is a narrow mode (117 love-tier images). The pairwise judge forced the model into the *reference pool's mode*, replacing an accidental mode collapse (clip-art flowers) with a deliberate mode collapse (the style of the 117 references).

## 19. Novelty Analysis

*   **Previously Known:** GRPO, HPSv3, VLM-as-a-judge, p5.brush.
*   **Known technique in new combination:** Using GRPO with a VLM judge for visual code generation.
*   **Potentially Novel Research Contribution:** Demonstrating that absolute VLM scoring causes rapid mode collapse in visual program synthesis, and that anchoring the VLM via pairwise comparison to a static, curated reference pool stabilizes the gradients.
*   **Likely to fail at scale:** The Puppeteer rendering loop. As model sizes and rollout batches increase (e.g., =64$ or $), launching headless browsers for every candidate will choke the training pipeline.

## 20. Reproducibility Analysis

**Score: Low / Medium.**
*   *Missing:* The exact Qwen model size/version is unspecified. The reference pool dataset is not public. The reward weights are provided, but the exact VLM judge model used for pairwise comparison during training is not explicitly named (only that Gemini/Opus were used for data generation). The puppeteer sandbox code and the GRPO training script are unavailable.
*   *Feasibility:* An independent researcher with sufficient compute could reconstruct the pipeline using 	rl (Transformer Reinforcement Learning) for GRPO, Qwen2.5-Coder, and a custom Node.js endpoint for rendering, but exact reproduction of the aesthetic weights is impossible without the 581-image pool.

## 21. Alternative Approaches

*   **PPO vs GRPO:** PPO requires a separate value network, doubling the memory footprint. GRPO is more suited for LLMs generating code due to its memory efficiency.
*   **Best-of-N (BoN):** Instead of RL, one could simply generate N images at inference time and use the VLM judge to pick the best. This requires zero training compute but massive inference compute. RL amortizes the search cost into the weights.
*   **DPO (Direct Preference Optimization):** If the authors already have 1,664 rated images, they could have applied DPO offline directly on the text-to-code pairs (Preferred vs Rejected) rather than using an online online VLM judge. DPO would be vastly cheaper computationally, though online RL (GRPO) allows the model to explore beyond the static dataset.

## 22. Proposed Improved Architecture

**The "Distilled Reward" Architecture:**
1.  **Eliminate the VLM Judge from the Online Loop:** Running a VLM judge and headless browser during the RL rollout loop is a massive bottleneck.
2.  **Train a Code-to-Reward Model:** Use the 1,664 rated examples to train a lightweight Reward Model (RM) that takes the *text prompt and the generated JavaScript code* as input and outputs a scalar reward. 
3.  **Online GRPO:** Train the Qwen policy using GRPO against this lightweight RM. This completely removes the browser sandbox and the VLM judge from the critical training path.
4.  **Periodic Calibration:** Every 100 steps, sample outputs, render them in Puppeteer, and have humans (or a powerful VLM) grade them to ensure the lightweight RM hasn't been hacked.

## 23. Proposed Experiments

*   **Experiment 1: Code Compression vs. Syntax Survival**
    *   *Hypothesis:* The token drop (13.5k to 2k) is driven by syntax error penalties, not aesthetic preference.
    *   *Variables:* Control = normal training. Experimental = ignore syntax errors (render whatever compiles, ignore the rest).
    *   *Expected Outcome:* If the experimental group retains long code, the compression was purely a survival mechanism against syntax failures.
*   **Experiment 2: Human vs. Model Reference Pools**
    *   *Hypothesis:* Model-generated reference pools cap the policy's creative ceiling.
    *   *Variables:* Compare a policy trained against the 117 model-generated references vs. 117 curated human p5.js artworks.
    *   *Expected Outcome:* The human-anchored model will output highly different programmatic structures, likely utilizing more complex flow fields.

## 24. Open Research Problems

1.  **Preventing Reward Hacking in Visual Code Generation:** How to design a reward function that penalizes "spaghetti code" without accidentally penalizing complex artistic structures.
2.  **Scalable Visual RL:** How to evaluate visual aesthetic outputs continuously without the severe latency of DOM/Browser rendering engines.
3.  **Subjectivity in RL:** Converting pairwise human preference into a reliable mathematical gradient without triggering mode collapse to the mean of human taste.

## 25. Final Technical Assessment

"Training AI to Paint with Code" is a compelling engineering feat that bridges Reinforcement Learning, Code Generation, and Generative Art. Its most significant finding is empirical: **absolute VLM scoring causes rapid mode collapse in programmatic art generation, while pairwise anchoring against a reference pool stabilizes training.**

However, the methodology rests on a precarious feedback loop. Because the reference pool was generated by AI models, and the judge is an AI model, the system is fundamentally an exercise in AI-to-AI distillation. The policy model learned to effectively mimic the narrow aesthetic distribution of the 117 "love" tier examples. The dramatic reduction in code length is likely a symptom of the model minimizing its exposure to syntax errors (a known issue in RL for code generation) rather than a genuine discovery of "algorithmic elegance." Despite these criticisms, it represents a highly functional template for using GRPO to optimize models for subjective tasks.

---

## Sources and Evidence Table

| Source | Type | Date | Topic | Reliability | Contribution |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [surya.website](https://surya.website/rling-qwen-to-paint-with-code) | Primary Project Page | Aug 2026 | Full Project Description | High (Primary) | Provided core architecture, reward rubric evolution, failure modes, and reference pool details. |
| [DeepSeekMath Paper (Shao et al.)](https://arxiv.org/abs/2402.03300) | Academic Paper | Feb 2024 | GRPO | Very High | Mathematical formulation of GRPO, eliminating the critic model and using group-relative advantages. |
| [HPSv3 Repository / Paper](https://github.com/tgxs002/HPSv3) | Academic Repo | 2023/2024 | VLM Aesthetic Evaluation | Very High | Detailed the HPDv3 dataset, uncertainty-aware ranking loss, and its use in visual alignment. |
| [GEPA (Genetic-Pareto)](https://github.com/GEPA) | Framework Docs | 2024/2025 | Prompt Optimization | High | Explained evolutionary prompt optimization, confirming why the 400-line API doc caused hallucinations. |
| [Vimeo Thesis (1190839818)](https://vimeo.com/1190839818) | Video Presentation | 2026 | Conceptual Motivation | High (Primary) | Emphasized that defining reward functions for subjective art is inherently a design problem. |
| [Reward Hacking in RLHF (Gao et al.)](https://arxiv.org/abs/2210.10760) | Academic Paper | 2022 | Mode Collapse / RLHF | Very High | Provided theoretical backing for the "Proxy Compression Hypothesis" explaining the clip-art flower mode collapse. |
| [Reddit (r/aiwars, r/hermesagent)](https://reddit.com) | Community Forum | 2026 | Community Perspective | Low/Medium | Provided criticisms regarding rendering latency and the readability of generated spaghetti code. |

