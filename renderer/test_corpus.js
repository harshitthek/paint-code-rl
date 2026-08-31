const { renderCode, closeBrowser } = require('./sandbox');

const tests = [
    {
        name: "1. Plain p5 canvas with background color",
        code: `
function setup() {
    createCanvas(400, 400);
    background(220);
    signalRenderComplete();
}
function draw() {}`
    },
    {
        name: "2. Colored rectangle",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    fill(255, 0, 0);
    rect(100, 100, 200, 200);
    signalRenderComplete();
}`
    },
    {
        name: "3. Multiple circles with colors",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    noStroke();
    fill(255, 0, 0); circle(100, 200, 50);
    fill(0, 255, 0); circle(200, 200, 50);
    fill(0, 0, 255); circle(300, 200, 50);
    signalRenderComplete();
}`
    },
    {
        name: "4. Lines and paths",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    stroke(0);
    strokeWeight(5);
    line(50, 50, 350, 350);
    signalRenderComplete();
}`
    },
    {
        name: "5. Transforms (translate, rotate)",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    translate(200, 200);
    rotate(PI / 4);
    rect(-50, -50, 100, 100);
    signalRenderComplete();
}`
    },
    {
        name: "6. Text rendering",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    fill(0);
    textSize(32);
    text('Hello World', 10, 50);
    signalRenderComplete();
}`
    },
    {
        name: "7. WebGL primitive (box, sphere)",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    background(200);
    rotateX(frameCount * 0.01);
    rotateY(frameCount * 0.01);
    box(100);
    signalRenderComplete();
}`
    },
    {
        name: "8. p5.brush primitive",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    brush.set('HB', '#000000', 1);
    brush.line(50, 50, 350, 350);
    signalRenderComplete();
}`
    },
    {
        name: "9. Multi-operation p5.brush drawing with fill",
        code: `
function setup() {
    createCanvas(400, 400);
    background(255);
    brush.fill('#ff0000', 100);
    brush.rect(100, 100, 200, 200);
    brush.set('pen', '#000000', 1);
    brush.circle(200, 200, 50);
    signalRenderComplete();
}`
    },
    {
        name: "10. Intentionally broken code",
        code: `
function setup() {
    createCanvas(400, 400);
    nonExistentFunction();
}`
    }
];

async function runTests() {
    console.log("Starting test corpus...");
    let passed = 0;
    let failed = 0;
    
    for (let i = 0; i < tests.length; i++) {
        const test = tests[i];
        console.log(`\nRunning: ${test.name}`);
        
        const startTime = Date.now();
        const res = await renderCode(test.code, 42, `test_${i}`);
        const latency = Date.now() - startTime;
        
        if (res.success || test.name.includes("broken")) {
            console.log(`[PASS] Latency: ${latency}ms`);
            passed++;
        } else {
            console.log(`[FAIL] Latency: ${latency}ms`);
            failed++;
        }
        console.log(`Error Class: ${res.error_classification}`);
        if (res.runtime_error) console.log(`Runtime Error: ${res.runtime_error}`);
        if (res.console_logs && res.console_logs.length > 0) {
            console.log(`Console Logs:`, res.console_logs);
        }
    }
    
    console.log(`\nDone! Passed: ${passed}, Failed: ${failed}`);
    await closeBrowser();
}

runTests().catch(console.error);
