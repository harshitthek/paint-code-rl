const express = require('express');
const { renderCode, closeBrowser } = require('./sandbox');

const app = express();
app.use(express.json({ limit: '5mb' }));

const MAX_INFLIGHT = 10;
let inflight = 0;
let jobsProcessed = 0;
const RESTART_AFTER = 100;
let isRecycling = false;

app.get('/health', (req, res) => {
    res.json({ status: 'ok', inflight, jobsProcessed });
});

app.post('/render', async (req, res) => {
    if (inflight >= MAX_INFLIGHT) {
        return res.status(429).json({ success: false, error_classification: "RENDERER_OVERLOAD", runtime_error: "Too many concurrent requests" });
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
        if (jobsProcessed >= RESTART_AFTER && inflight === 0 && !isRecycling) {
            isRecycling = true;
            try {
                console.log("Recycling browser safely (inflight === 0)...");
                await closeBrowser();
                jobsProcessed = 0;
            } catch (e) {
                console.error("Error recycling browser:", e);
            } finally {
                isRecycling = false;
            }
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
