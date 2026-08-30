# ADR-006: Deferred Reference-Pool Refresh

**Status:** Accepted
**Context:** A dynamic, self-improving reference pool helps prevent mode collapse in visual RL but adds significant system complexity.
**Decision:** Defer dynamic refresh to Phase 3. 
**Consequences:** Phase 0 and 1 will rely on a static, curated reference pool. 
