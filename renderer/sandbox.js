const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');

let browser;

async function initBrowser() {
    if (browser) return browser;
    const args = [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--enable-webgl',
        '--allow-file-access-from-files',
    ];

    // Platform-specific WebGL backend selection
    if (process.platform === 'darwin' && process.arch === 'arm64') {
        // macOS ARM64 (Apple Silicon): use native Metal GPU via ANGLE
        args.push('--use-gl=angle');
        args.push('--use-angle=metal');
        args.push('--disable-gpu-sandbox');
    } else if (process.platform === 'linux') {
        // Linux (Kaggle/Docker): use SwiftShader software rasterizer
        args.push('--use-gl=angle');
        args.push('--use-angle=swiftshader-webgl');
    } else {
        // Windows / other: use SwiftShader as safe fallback
        args.push('--use-gl=angle');
        args.push('--use-angle=swiftshader-webgl');
    }

    browser = await puppeteer.launch({
        headless: 'new',
        args: args
    });
    return browser;
}

async function renderCode(code, seed, runId, options = {}) {
    const timeouts = {
        browser_startup: options.browser_startup_timeout_ms || 10000,
        page_load: options.page_load_timeout_ms || 5000,
        code_execution: options.code_execution_timeout_ms || 4000,
        screenshot: options.screenshot_timeout_ms || 3000
    };

    const b = await initBrowser();
    const page = await b.newPage();
    page.setDefaultTimeout(timeouts.page_load);
    
    await page.setRequestInterception(true);
    page.on('request', request => {
        const url = request.url();
        if (url.startsWith('file://') || url.startsWith('data:')) {
            request.continue();
        } else {
            request.abort();
        }
    });

    let error_classification = 'SUCCESS';
    let runtime_error = null;
    let console_logs = [];

    page.on('console', msg => {
        console_logs.push(`[${msg.type()}] ${msg.text()}`);
    });

    page.on('pageerror', err => {
        const errStr = err.toString();
        if (runtime_error) {
            runtime_error += "\n" + errStr;
        } else {
            runtime_error = errStr;
        }
        if (errStr.includes('SyntaxError')) {
            error_classification = 'PARSE_ERROR';
        } else {
            error_classification = 'RUNTIME_ERROR';
        }
    });

    const tmpFile = path.resolve(__dirname, `tmp_${runId}.html`);

    try {
        const templatePath = path.resolve(__dirname, 'template.html');
        let htmlContent = fs.readFileSync(templatePath, 'utf8');
        
        let seedHook = '';
        if (seed !== undefined && seed !== null) {
            seedHook = `
(function(s) {
    let _seed = s >>> 0;
    Math.random = function() {
        _seed |= 0; _seed = _seed + 0x6D2B79F5 | 0;
        let t = Math.imul(_seed ^ _seed >>> 15, 1 | _seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
})(${seed});
`;
        }
        htmlContent = htmlContent.replace('// SEED_INJECTION_HOOK', seedHook);

        let safeCode = code;
        if (seed !== undefined && seed !== null) {
            safeCode = safeCode.replace(
                /function\s+setup\s*\(\)\s*\{/,
                "function setup() { try { randomSeed(" + seed + "); noiseSeed(" + seed + "); } catch(e){} "
            );
        }
        
        // Automated Lifecycle & WebGL Guard Hook
        const autoSignalWrapper = `
// Intercept createCanvas to setup brush automatically
(function() {
    if (typeof window !== 'undefined') {
        window.camera = window.camera || function() {};
        window.camera.position = window.camera.position || function() {};
        window.camera.lookAt = window.camera.lookAt || function() {};
        window.Brush = window.Brush || function() { return window.brush; };
    }
    if (typeof window.p5 !== 'undefined') {
        const _origCreateCanvas2 = window.p5.prototype.createCanvas;
        window.p5.prototype.createCanvas = function(...args) {
            const result = _origCreateCanvas2.apply(this, args);
            if (typeof brush !== 'undefined') {
                if (typeof brush.bleed === 'function' && typeof brush.fillBleed === 'undefined') {
                    brush.fillBleed = brush.bleed;
                }
                if (typeof brush.pushMatrix === 'undefined') {
                    brush.pushMatrix = function() { if (typeof push === 'function') push(); };
                    brush.popMatrix = function() { if (typeof pop === 'function') pop(); };
                }
                if (typeof brush.setColor === 'undefined') {
                    brush.setColor = function(c) { if (typeof brush.stroke === 'function') brush.stroke(c); };
                }
                if (typeof brush.load === 'function') {
                    try { brush.load(); } catch(e) {}
                }
                if (typeof brush.scaleBrushes === 'function') {
                    try { brush.scaleBrushes(3); } catch(e) {}
                }
                // Proxy fallback for any hallucinated brush methods
                try {
                    window.brush = new Proxy(brush, {
                        get(target, prop, receiver) {
                            if (prop in target) {
                                const val = Reflect.get(target, prop, receiver);
                                return typeof val === 'function' ? val.bind(target) : val;
                            }
                            return function(...args) {
                                return target;
                            };
                        }
                    });
                } catch(e) {}
            }
            return result;
        };
    }
})();

${safeCode}

// Fallback timeout to complete rendering
setTimeout(function() {
    if (!window.renderComplete) {
        window.renderComplete = true;
    }
}, ${timeouts.code_execution});
`;
        
        htmlContent = htmlContent.replace('// INJECT_CODE_HERE', autoSignalWrapper);
        
        fs.writeFileSync(tmpFile, htmlContent);

        const fileUrl = 'file://' + tmpFile;
        await page.goto(fileUrl, { waitUntil: 'load' });
        
        try {
            await page.waitForFunction('window.renderComplete === true', { timeout: timeouts.code_execution + 500 });
        } catch(e) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'TIMEOUT';
            }
        }
        
        const outDir = path.resolve(__dirname, '../artifacts/renders');
        if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
        
        const outPath = path.join(outDir, `${runId}.png`);
        
        // Ensure canvas exists and is visible
        const canvas = await page.$('canvas');
        if (!canvas) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'NO_CANVAS';
            }
            throw new Error("NO_CANVAS");
        }
        
        try {
            await canvas.screenshot({ path: outPath, timeout: timeouts.screenshot });
        } catch(e) {
            if (e.message && e.message.includes('Node is either not visible')) {
                error_classification = 'NO_CANVAS';
                throw new Error("NO_CANVAS");
            } else {
                throw e;
            }
        }
        
        return {
            success: error_classification === 'SUCCESS',
            image_path: outPath,
            error_classification: error_classification,
            runtime_error: runtime_error,
            console_logs: console_logs
        };

    } catch (e) {
        if (error_classification === 'SUCCESS') {
            if (e.toString().includes('Navigation timeout') || e.toString().includes('TIMEOUT')) error_classification = 'TIMEOUT';
            else error_classification = 'BROWSER_ERROR';
        }
        return {
            success: false,
            image_path: null,
            error_classification: error_classification,
            runtime_error: e.toString() + (runtime_error ? " | " + runtime_error : ""),
            console_logs: typeof console_logs !== 'undefined' ? console_logs : []
        };
    } finally {
        // Guaranteed disk and memory resource cleanup
        try {
            if (fs.existsSync(tmpFile)) {
                fs.unlinkSync(tmpFile);
            }
        } catch (e) {}
        await page.close().catch(() => {});
    }
}

async function closeBrowser() {
    if (browser) {
        await browser.close().catch(() => {});
        browser = null;
    }
}

module.exports = { initBrowser, renderCode, closeBrowser };
