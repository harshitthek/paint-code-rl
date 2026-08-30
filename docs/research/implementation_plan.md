# Deep Research: Adversarial Audit Phase

This plan outlines the steps for the second, adversarial pass on the "Training AI to Paint with Code" research dossier.

## Proposed Changes

### 1. Verification of Technical Claims
- **GRPO:** Search for and verify the exact math in the DeepSeekMath paper to ensure accurate representation, separating canonical GRPO from inferred implementation.
- **Pairwise Preference:** Audit the terminology (win-rate vs. Elo/Thurstone) used for the pairwise judging.
- **Terminology Correction:** Fix the incorrect application of "RLAIF" and "Reward Model Distillation" since the authors explicitly stated training a reward model was a future step.
- **HPSv3 & GEPA:** Deeply search for the original papers/authors for HPSv3 and GEPA to correct any misattributions and fully understand their mechanisms.

### 2. Causal Falsification & Alternative Explanations
- **Code Compression:** Investigate alternative hypotheses for the token drop (e.g., syntax-error probability, timeout probability) and design experiments to distinguish them.
- **Mode Collapse Resolution:** Design an ablation matrix to isolate the true cause of the mode collapse resolution, as multiple variables were changed simultaneously in the project.
- **Reference Pool:** Analyze the consequences of using a 581-image model-generated reference pool, including selection bias and mode concentration.

### 3. Deep Literature Review & Novelty Audit
- Conduct fresh, rigorous searches for prior work on visual RL, VLM reward models, programmatic art, and multimodal GRPO.
- Create a strict Novelty Matrix to separate exact precedents from genuine novelties.

### 4. Deliverable Construction
I will create a new artifact named `Adversarial_Audit_Training_AI_to_Paint_with_Code.md` containing the 10 requested sections, strictly applying epistemic labels (e.g., [FACT - PRIMARY SOURCE], [SUPPORTED INFERENCE]) to every claim.

## User Review Required
> [!IMPORTANT]
> Please review this audit plan. Once approved, I will conduct the deep searches and produce the adversarial audit report.

## Verification Plan
- The resulting artifact will be checked against the 10 required sections and the strict epistemic labeling requirement.
