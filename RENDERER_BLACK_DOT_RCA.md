# Renderer Black Dot Root Cause Analysis

## Symptoms
The renderer sub-system was producing "black dots" or completely trivial/empty output for p5.brush art generation instead of proper strokes and filled shapes.

## Root Causes

1. **p5.brush Initialization & Scaling Missing**
   The p5.brush library (v1.x vendored) requires `brush.scaleBrushes(3)` to properly scale the brushes. Without it, they render microscopically or are essentially invisible. In addition, `brush.load()` needs to be called to initialize the library properly.

2. **template.html Timing Bug**
   The monkeypatch that forced `createCanvas` into `WEBGL` mode was bound to the `window.load` event. However, `p5.js` evaluates and runs `setup()` before `window.load` fires in the Puppeteer environment. As a result, the `createCanvas` override was never applied in time for the art scripts, failing to provide the required WebGL context.

3. **Invalid Brush Types ("watercolor")**
   The template prompts or system code attempted to use `"watercolor"` as a stroke brush. `"watercolor"` is not a valid stroke brush in p5.brush; it's a fill operation method (`brush.fill()`). Valid brush names are: `"HB"`, `"2B"`, `"2H"`, `"cpencil"`, `"pen"`, `"rotring"`, `"spray"`, `"marker"`, `"marker2"`, `"charcoal"`, `"hatch_brush"`.

4. **Puppeteer WebGL Configuration**
   Puppeteer headless was running without necessary WebGL software rasterization flags. This prevented it from correctly rendering WebGL canvases in environments without dedicated GPUs. 
   The required arguments: `--use-gl=angle` and `--use-angle=swiftshader-webgl` were missing.

5. **Sandbox Lifecycle Hook Overwriting `setup()`**
   The sandbox IIFE forcefully overwrote `window.setup()` which broke p5.js's normal auto-initialization loop and lifecycle mechanisms.

## Fixes Implemented

1. **template.html Patching**
   - Removed the `window.addEventListener('load', ...)` wrapping the `createCanvas` monkeypatch.
   - Inserted an inline script execution *immediately after* the p5.js `<script>` tag and *before* p5.brush and user scripts.
   - Added a `window.__FORCE_WEBGL = true` flag.

2. **sandbox.js Improvements**
   - Added `'--use-gl=angle'` and `'--use-angle=swiftshader-webgl'` to Puppeteer launch arguments to ensure proper WebGL 2 rendering headless.
   - Updated the auto-signal wrapper to intelligently intercept `createCanvas` internally rather than overriding `window.setup()`. This wrapper now guarantees `brush.load()` and `brush.scaleBrushes(3)` are called automatically.
   - Added configurable per-phase timeouts to avoid hanging executions on failed/looping code.
   - Added browser console and error log capturing.

3. **Testing**
   - Created `renderer/test_corpus.js` to execute 10 diverse programs, ranging from standard p5.js primitives to specific WebGL and p5.brush operations.
