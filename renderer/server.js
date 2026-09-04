const express = require('express');
const { renderCode, closeBrowser, getActiveRenders, isRecycleRequired } = require('./sandbox');

const app = express();
app.use(express.json({ limit: '5mb' }));

const os = require('os');
const MAX_INFLIGHT = 64;
const CONCURRENT_WORKERS = parseInt(process.env.CONCURRENT_WORKERS, 10) || Math.min(8, Math.max(2, (os.cpus() && os.cpus().length) || 4));

async function mapConcurrent(items, limit, fn) {
    const results = new Array(items.length);
    let index = 0;
    const workers = new Array(Math.min(limit, items.length)).fill(0).map(async () => {
        while (index < items.length) {
            const i = index++;
            results[i] = await fn(items[i], i);
        }
    });
    await Promise.all(workers);
    return results;
}
let inflight = 0;
let jobsProcessed = 0;
const RESTART_AFTER = 100;
let recyclePromise = null;

app.get('/health', (req, res) => {
    res.json({ status: 'ok', inflight, jobsProcessed, activeRenders: getActiveRenders() });
});

const SHUTDOWN_TOKEN = process.env.RENDERER_SHUTDOWN_TOKEN;
if (!SHUTDOWN_TOKEN) {
    console.error("Fatal: RENDERER_SHUTDOWN_TOKEN environment variable must be set.");
    process.exit(1);
}

app.post('/shutdown', async (req, res) => {
    const clientToken = req.headers['x-renderer-token'];
    if (!clientToken || clientToken !== SHUTDOWN_TOKEN) {
        return res.status(401).json({ error: "Unauthorized: Invalid or missing X-Renderer-Token" });
    }
    res.json({ status: 'shutting_down' });
    setTimeout(async () => {
        try { await closeBrowser(true); } catch (e) {}
        process.exit(0);
    }, 50);
});

app.post('/render', async (req, res) => {
    if (inflight >= MAX_INFLIGHT) {
        return res.status(429).json({ success: false, error_classification: "RENDERER_OVERLOAD", runtime_error: "Too many concurrent requests" });
    }
    
    // Await ongoing recycling if any
    if (recyclePromise) {
        await recyclePromise;
    }

    inflight++;
    jobsProcessed++;

    try {
        const { prompt, code, seed } = req.body;
        
        if (!code) {
            return res.status(400).json({ success: false, error_classification: "PARSE_ERROR", runtime_error: "No code provided" });
        }
        
        const runId = 'render_' + Date.now() + '_' + Math.floor(Math.random()*1000);
        const start = Date.now();
        
        const result = await renderCode(code, seed, runId);
        result.render_ms = Date.now() - start;
        
        res.json(result);
    } finally {
        inflight--;
        // Safely recycle browser only when no other requests are in flight
        if ((jobsProcessed >= RESTART_AFTER || isRecycleRequired()) && inflight === 0 && !recyclePromise) {
            console.log("Recycling browser safely (inflight === 0)...");
            recyclePromise = (async () => {
                try {
                    await closeBrowser(true);
                    jobsProcessed = 0;
                } catch (e) {
                    console.error("Error recycling browser:", e);
                } finally {
                    recyclePromise = null;
                }
            })();
            await recyclePromise;
        }
    }
});

let batchSeq = 0;

app.post('/render_batch', async (req, res) => {
    const { items, return_base64 } = req.body;
    if (!Array.isArray(items) || items.length === 0) {
        return res.status(400).json({ success: false, error: "items must be a non-empty array" });
    }
    
    if (inflight + items.length > MAX_INFLIGHT) {
        return res.status(429).json({ success: false, error: "Too many concurrent requests" });
    }
    
    // Atomically reserve capacity before yielding to event loop
    inflight += items.length;
    jobsProcessed += items.length;

    if (recyclePromise) {
        await recyclePromise;
    }
    
    try {
        const results = await mapConcurrent(items, CONCURRENT_WORKERS, async (item, idx) => {
            const runId = 'batch_' + Date.now() + '_' + (++batchSeq) + '_' + idx;
            const start = Date.now();
            const opts = return_base64 ? { return_base64: true } : {};
            try {
                const result = await renderCode(item.code || '', item.seed, runId, opts);
                result.render_ms = Date.now() - start;
                result.batch_index = idx;
                return result;
            } catch (err) {
                return {
                    success: false,
                    image_path: null,
                    error_classification: 'BATCH_RENDER_ERROR',
                    runtime_error: err.toString(),
                    batch_index: idx,
                    render_ms: Date.now() - start
                };
            }
        });
        res.json({ success: true, results });
    } finally {
        inflight -= items.length;
        if ((jobsProcessed >= RESTART_AFTER || isRecycleRequired()) && inflight === 0 && !recyclePromise) {
            recyclePromise = (async () => {
                try {
                    await closeBrowser(true);
                } catch (e) {
                    console.error("Error recycling browser:", e);
                } finally {
                    jobsProcessed = 0;
                    recyclePromise = null;
                }
            })();
            await recyclePromise;
        }
    }
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, '0.0.0.0', async () => {
    console.log(`Renderer server listening on port ${PORT}`);
    try {
        require('./print_hashes.js');
    } catch (e) {}
});
