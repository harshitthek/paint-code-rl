# ADR-004: FREE / LOCAL / PAID Architecture

**Status:** Accepted
**Context:** Research must be reproducible for engineers without extensive compute budgets, while remaining scalable.
**Decision:** The architecture explicitly branches into FREE (strictly isolated, zero-cost), LOCAL (local capabilities only), and PAID (cloud APIs/GPU).
**Consequences:** Hard configuration blocks prevent silent fallbacks to paid APIs (e.g., OpenAI) in FREE/LOCAL modes.
