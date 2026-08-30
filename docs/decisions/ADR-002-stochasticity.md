# ADR-002: Stochasticity in p5.brush

**Status:** Accepted
**Context:** p5.brush is highly stochastic by nature, rendering slightly different strokes even with identical seeds.
**Decision:** Treat stochasticity as environment noise. The RL agent must learn robust programs that produce visually acceptable outputs despite random perturbations.
**Consequences:** Rewards will have inherent variance. Checkpoints must evaluate over multiple rollouts to ascertain true performance gains.
