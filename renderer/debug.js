const puppeteer = require('puppeteer');
async function run() {
    const browser = await puppeteer.launch({headless: true, args: ['--use-gl=egl', '--disable-dev-shm-usage']});
    const page = await browser.newPage();
    page.on('console', msg => console.log('PAGE LOG:', msg.text()));
    page.on('pageerror', error => console.log('PAGE ERROR:', error.message));
    page.on('requestfailed', request => console.log('REQUEST FAILED:', request.url(), request.failure().errorText));
    
    await page.goto('file://C:/Users/user/.gemini/antigravity/brain/eabfab2e-f626-4128-9da1-6868c5d0f842/project/renderer/template.html');
    console.log("Loaded template.");
    await browser.close();
}
run();
