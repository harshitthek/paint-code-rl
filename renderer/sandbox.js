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
        '--use-gl=angle',
        '--use-angle=swiftshader-webgl'
    ];
    if (process.platform === 'linux') {
        args.push('--use-gl=egl');
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

    try {
        const templatePath = path.resolve(__dirname, 'template.html');
        let htmlContent = fs.readFileSync(templatePath, 'utf8');
        
        let safeCode = code;
        if (seed !== undefined && seed !== null) {
            safeCode = safeCode.replace(/function\s+setup\s*\(\)\s*\{/, "function setup() { randomSeed(" + seed + "); noiseSeed(" + seed + "); ");
        }
        
        // Automated Lifecycle & WebGL Guard Hook
        const autoSignalWrapper = `
// Intercept createCanvas to setup brush automatically
(function() {
    if (typeof window.p5 !== 'undefined') {
        const _origCreateCanvas2 = window.p5.prototype.createCanvas;
        window.p5.prototype.createCanvas = function(...args) {
            const result = _origCreateCanvas2.apply(this, args);
            if (typeof brush !== 'undefined') {
                if (typeof brush.load === 'function') {
                    try { brush.load(); } catch(e) {}
                }
                if (typeof brush.scaleBrushes === 'function') {
                    try { brush.scaleBrushes(3); } catch(e) {}
                }
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
        
        const tmpFile = path.resolve(__dirname, `tmp_${runId}.html`);
        fs.writeFileSync(tmpFile, htmlContent);

        const fileUrl = 'file://' + tmpFile;
        await page.goto(fileUrl, { waitUntil: 'load' });
        
        try {
            await page.waitForFunction('window.renderComplete === true', { timeout: timeouts.code_execution + 500 });
        } catch(e) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'TIMEOUT';
            }
            // we don't throw here so we can still try to grab screenshot
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
            if (e.message.includes('Node is either not visible')) {
                error_classification = 'NO_CANVAS';
                throw new Error("NO_CANVAS");
            } else {
                throw e;
            }
        }
        
        try { fs.unlinkSync(tmpFile); } catch(e) {}
        await page.close();
        
        return {
            success: error_classification === 'SUCCESS',
            image_path: outPath,
            error_classification: error_classification,
            runtime_error: runtime_error,
            console_logs: console_logs
        };

    } catch (e) {
        await page.close().catch(()=>{});
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
    }
}

async function closeBrowser() {
    if (browser) {
        await browser.close().catch(()=>{});
        browser = null;
    }
}

module.exports = { initBrowser, renderCode, closeBrowser };
