# ADR-003: Rejection of Soup as Dependency

**Status:** Accepted
**Context:** Soup implements an aggressively memory-efficient layer streaming architecture for low-VRAM devices.
**Decision:** Soup is rejected as a mandatory dependency. Layer streaming fundamentally bottlenecks autoregressive generation in GRPO loops (due to per-token reloading of full model weights).
**Consequences:** We adopt Soup's *engineering patterns* (robust preflight checks, asynchronous rollouts, strict NaN validation) without coupling to its training loop.
