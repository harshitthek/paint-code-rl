# ADR-007: Deferred Distributed Infrastructure

**Status:** Accepted
**Context:** Scaling to Ray or Vast.ai cluster nodes requires robust pub/sub and distributed checkpoints.
**Decision:** Distributed infrastructure is deferred.
**Consequences:** The codebase remains portable and runnable entirely within a single Python process using concurrent.futures.
