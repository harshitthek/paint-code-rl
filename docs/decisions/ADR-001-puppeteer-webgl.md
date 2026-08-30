# ADR-001: Puppeteer/WebGL Retention

**Status:** Accepted
**Context:** The project requires rendering p5.js/p5.brush scripts to PNG for visual reward evaluation. Porting p5.brush to Python/Skia would decouple the research from its exact javascript targets.
**Decision:** We retain Puppeteer/WebGL to ensure perfect 1:1 execution fidelity with actual p5.brush canvas outputs. 
**Consequences:** Requires Node.js and Chromium as system dependencies. Overheads are mitigated via batched async rendering pipelines.
