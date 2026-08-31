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
        '--allow-file-access-from-files'
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

async function renderCode(code, seed, runId) {
    const b = await initBrowser();
    const page = await b.newPage();
    page.setDefaultTimeout(4000);
    
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

    page.on('pageerror', err => {
        runtime_error = err.toString();
        if (runtime_error.includes('SyntaxError')) {
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
        
        // Wrap code with automated render completion signal
        const autoSignalWrapper = `
${safeCode}

// Automated Lifecycle Hook
(function() {
    const origSetup = typeof window.setup === 'function' ? window.setup : null;
    window.setup = function() {
        if (origSetup) {
            origSetup();
        }
        setTimeout(function() {
            window.renderComplete = true;
        }, 200);
    };
    setTimeout(function() {
        window.renderComplete = true;
    }, 1500);
})();
`;
        
        htmlContent = htmlContent.replace('// INJECT_CODE_HERE', autoSignalWrapper);
        
        const tmpFile = path.resolve(__dirname, `tmp_${runId}.html`);
        fs.writeFileSync(tmpFile, htmlContent);

        const fileUrl = 'file://' + tmpFile;
        await page.goto(fileUrl, { waitUntil: 'load' });
        
        try {
            await page.waitForFunction('window.renderComplete === true', { timeout: 3000 });
        } catch(e) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'TIMEOUT';
            }
            throw new Error('TIMEOUT');
        }
        
        const outDir = path.resolve(__dirname, '../artifacts/renders');
        if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
        
        const outPath = path.join(outDir, `${runId}.png`);
        
        // Ensure canvas exists and is visible
        const canvas = await page.$('canvas');
        if (!canvas) {
            error_classification = 'NO_CANVAS';
            throw new Error("NO_CANVAS");
        }
        
        try {
            await canvas.screenshot({ path: outPath });
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
            runtime_error: runtime_error
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
            runtime_error: e.toString() + (runtime_error ? " | " + runtime_error : "")
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
