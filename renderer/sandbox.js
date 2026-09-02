const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const vm = require('vm');

function balanceTokens(jsCode) {
    if (!jsCode) return "";
    let openP = (jsCode.match(/\(/g) || []).length;
    let closeP = (jsCode.match(/\)/g) || []).length;
    if (openP > closeP) jsCode += ")".repeat(openP - closeP) + ";";

    let openB = (jsCode.match(/\{/g) || []).length;
    let closeB = (jsCode.match(/\}/g) || []).length;
    if (openB > closeB) jsCode += "\n" + "}".repeat(openB - closeB);

    return jsCode;
}

function autoRepairJS(jsCode) {
    if (!jsCode) return jsCode;
    let candidate = balanceTokens(jsCode);
    try {
        new vm.Script(candidate);
        return candidate;
    } catch (e) {}

    let lines = jsCode.split("\n");
    while (lines.length > 2) {
        lines.pop();
        let cand = balanceTokens(lines.join("\n"));
        try {
            new vm.Script(cand);
            return cand;
        } catch (e) {}
    }
    return candidate;
}

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

    const launchOptions = {
        headless: 'new',
        args: args
    };

    if (process.env.PUPPETEER_EXECUTABLE_PATH && fs.existsSync(process.env.PUPPETEER_EXECUTABLE_PATH)) {
        launchOptions.executablePath = process.env.PUPPETEER_EXECUTABLE_PATH;
    } else if (fs.existsSync('/usr/bin/google-chrome')) {
        launchOptions.executablePath = '/usr/bin/google-chrome';
    } else if (fs.existsSync('/usr/bin/google-chrome-stable')) {
        launchOptions.executablePath = '/usr/bin/google-chrome-stable';
    }
    // Note: Never use /usr/bin/chromium-browser on Linux as Ubuntu packages it as an unusable snap stub

    browser = await puppeteer.launch(launchOptions);
    return browser;
}

async function renderCode(code, seed, runId, options = {}) {
    // Security: sanitize runId to prevent directory traversal
    const safeRunId = String(runId || 'render_' + Date.now()).replace(/[^a-zA-Z0-9_-]/g, '_');
    
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
    const rendererDir = path.resolve(__dirname);
    page.on('request', request => {
        try {
            const url = request.url();
            if (url.startsWith('data:')) {
                request.continue();
            } else if (url.startsWith('file://')) {
                try {
                    let fileUrlPath = new URL(url).pathname;
                    if (process.platform === 'win32' && fileUrlPath.startsWith('/')) {
                        fileUrlPath = fileUrlPath.slice(1);
                    }
                    const resolvedFile = path.resolve(decodeURIComponent(fileUrlPath));
                    const rel = path.relative(rendererDir, resolvedFile);
                    if (!rel.startsWith('..') && !path.isAbsolute(rel)) {
                        request.continue();
                    } else {
                        request.abort('accessdenied');
                    }
                } catch (e) {
                    request.abort('accessdenied');
                }
            } else {
                request.abort('accessdenied');
            }
        } catch (err) {
            // Request may have already been handled
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

    const tmpFile = path.resolve(__dirname, `tmp_${safeRunId}.html`);

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

        let safeCode = autoRepairJS(code);
        // Heal accidental brush re-declarations (e.g. let brush; or const brush = new p5.Brush())
        safeCode = safeCode.replace(
            /^[ \t]*(?:const|let|var)\s+brush\s*(?:=\s*(?:new\s+(?:p5\.)?Brush\([^)]*\)|brush|window\.brush|[^;\n]+))?;?/gmi,
            '// [auto-healed brush declaration]\n'
        );
        safeCode = safeCode.replace(
            /^[ \t]*brush\s*=\s*new\s+(?:p5\.)?Brush\([^)]*\);?/gmi,
            '// [auto-healed brush assignment]\n'
        );

        // Replace external loadImage calls with instant procedural dummy images (prevents preload hangs)
        safeCode = safeCode.replace(/loadImage\s*\([^)]*\)/g, 'createImage(100, 100)');

        // Inline preload() body into setup() so variables (e.g. mountains = [], clouds = []) are initialized
        // without risking premature brush.load() failure before createCanvas()
        const preloadMatch = safeCode.match(/function\s+preload\s*\(\)\s*\{([\s\S]*?)\}/);
        if (preloadMatch) {
            const preloadBody = preloadMatch[1]
                .replace(/brush\s*=\s*new[^;\n]+;?/g, '')
                .replace(/brush\.load\(\);?/g, '');
            safeCode = safeCode.replace(/function\s+preload\s*\(\)\s*\{[\s\S]*?\}/, '// [inlined preload]');
            safeCode = safeCode.replace(/(function\s+setup\s*\(\)\s*\{)/, `$1\n  ${preloadBody}\n`);
        }

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
        window.Brush = function() { return window.brush || {}; };
        window.p5 = window.p5 || {};
        window.p5.Brush = function() { return window.brush || {}; };
        window.loadImage = function(p, s, f) {
            try {
                const img = (typeof createImage === 'function') ? createImage(100, 100) : { width: 100, height: 100 };
                if (typeof s === 'function') s(img);
                return img;
            } catch(e) { return { width: 100, height: 100 }; }
        };
        
        // Fallback variables so unquoted parameter identifiers never throw ReferenceError
        window.strength = 0.2;
        window.strenght = 0.2;
        window.bleed = 0.2;
        window.opacity = 160;
        window.alpha = 160;
        window.density = 1;
        window.angle = 0;
        window.radius = 50;
        window.size = 20;
        window.speed = 1;
        window.weight = 2;
        window.brushSize = 2;
        window.color = '#1a759f';
        window.col = '#1a759f';
        window.x = 300;
        window.y = 300;
        window.w = 600;
        window.h = 600;
    }
    if (typeof window.p5 !== 'undefined' && window.p5.prototype) {
        window.p5.prototype.Brush = function() { return window.brush || {}; };
        // Safe loadImage fallback to prevent preload hangs
        window.p5.prototype.loadImage = function(path, success, failure) {
            try {
                const img = this.createImage(100, 100);
                if (typeof success === 'function') success(img);
                return img;
            } catch(e) {
                return {};
            }
        };
        // Safe image() drawer that ignores null/undefined
        const _origImage = window.p5.prototype.image;
        window.p5.prototype.image = function(img, ...args) {
            if (!img) return;
            try {
                return _origImage.apply(this, [img, ...args]);
            } catch(e) {}
        };
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

// Auto-signal completion after draw() or setup() has completed painting the canvas
(function() {
    if (typeof window !== 'undefined') {
        if (typeof window.draw === 'function') {
            const _userDraw = window.draw;
            window.draw = function(...args) {
                const res = _userDraw.apply(this, args);
                setTimeout(function() {
                    window.renderComplete = true;
                }, 400);
                return res;
            };
        } else if (typeof window.setup === 'function') {
            const _userSetup = window.setup;
            window.setup = function(...args) {
                const res = _userSetup.apply(this, args);
                setTimeout(function() {
                    window.renderComplete = true;
                }, 400);
                return res;
            };
        }
    }
})();

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
        
        // Ensure canvas exists and is visible
        const canvas = await page.$('canvas');
        if (!canvas) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'NO_CANVAS';
            }
            throw new Error("NO_CANVAS");
        }
        
        let outPath = null;
        let image_base64 = null;
        
        if (options.return_base64) {
            // In-memory base64 capture — avoids disk I/O during RL training
            try {
                image_base64 = await canvas.screenshot({ encoding: 'base64', timeout: timeouts.screenshot });
            } catch(e) {
                if (e.message && e.message.includes('Node is either not visible')) {
                    error_classification = 'NO_CANVAS';
                    throw new Error("NO_CANVAS");
                } else {
                    throw e;
                }
            }
        } else {
            // Disk-backed capture — for tests, showcase, and backward compatibility
            const outDir = path.resolve(__dirname, '../artifacts/renders');
            if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
            outPath = path.join(outDir, `${safeRunId}.png`);
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
        }
        
        const result = {
            success: error_classification === 'SUCCESS',
            image_path: outPath,
            error_classification: error_classification,
            runtime_error: runtime_error,
            console_logs: console_logs
        };
        if (image_base64) {
            result.image_base64 = image_base64;
        }
        return result;

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
