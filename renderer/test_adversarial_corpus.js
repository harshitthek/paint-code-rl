/**
 * Adversarial and Stress Test Corpus for WebGL Sandboxed Renderer.
 * Tests 10 rough, adversarial edge cases discovered during real training rollouts.
 */
const { renderCode, closeBrowser } = require('./sandbox');

const adversarialTests = [
    {
        name: "1. Redeclared brush with fake constructor (const brush = new p5.Brush())",
        code: `
const brush = new p5.Brush();
function setup() {
    createCanvas(400, 400, WEBGL);
    background(245, 243, 238);
    brush.load();
    brush.scaleBrushes(3);
    brush.set('charcoal', '#3a6c73', 2);
    brush.line(50, 50, 350, 350);
}
`,
        expectSuccess: true
    },
    {
        name: "2. Shadowed let brush and inlined preload arrays (mountains.push)",
        code: `
let brush;
let mountains;
let clouds;

function preload() {
    brush = new p5.Brush();
    brush.load();
    mountains = [];
    clouds = [];
}

function setup() {
    createCanvas(400, 400, WEBGL);
    background(245, 243, 238);
    brush.scaleBrushes(3);
    mountains.push({ x: 50, y: 150 });
    clouds.push({ x: 100, y: 50 });
}

function draw() {
    for (let m of mountains) {
        brush.line(m.x, m.y, m.x + 100, m.y + 100);
    }
}
`,
        expectSuccess: true
    },
    {
        name: "3. Hallucinated external textures (loadImage & image drawer)",
        code: `
const TEXTURE_PATHS = ['missing_trunk.png', 'missing_leaves.jpg'];
function setup() {
    createCanvas(400, 400, WEBGL);
    background(240);
    for (let p of TEXTURE_PATHS) {
        let img = loadImage(p);
        image(img, 0, 0);
    }
    brush.set('pen', '#112233', 1);
    brush.line(20, 20, 200, 200);
}
`,
        expectSuccess: true
    },
    {
        name: "4. Unquoted parameter identifiers (bleed, out, opacity, color)",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    background(250);
    brush.bleed(bleed, out);
    brush.fill(color, opacity);
    brush.rect(x, y, w, h);
    brush.set(charcoal, '#223344', weight);
    brush.line(50, 50, 350, 350);
}
`,
        expectSuccess: true
    },
    {
        name: "5. Undeclared 3D camera methods (camera.position / lookAt)",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    camera.position(0, 0, 500);
    camera.lookAt(0, 0, 0);
    background(230);
    brush.circle(0, 0, 50);
}
`,
        expectSuccess: true
    },
    {
        name: "6. Missing createCanvas call (auto-repaired to WEBGL)",
        code: `
function setup() {
    background(245);
    brush.set('HB', '#000000', 2);
    brush.line(10, 10, 300, 300);
}
`,
        expectSuccess: true
    },
    {
        name: "7. Separate draw loop with noLoop() (lifecycle timing guard)",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    background(255);
    noLoop();
}
function draw() {
    translate(-width/2, -height/2);
    brush.set('pen', '#cc0000', 3);
    brush.line(100, 100, 300, 300);
}
`,
        expectSuccess: true
    },
    {
        name: "8. Hallucinated brush methods handled gracefully via Proxy",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    background(240);
    brush.setColor('#ff0000');
    brush.pushMatrix();
    brush.setPalette('sunset');
    brush.nonExistentAIBrushFunction(123, 'extra');
    brush.popMatrix();
    brush.circle(0, 0, 60);
}
`,
        expectSuccess: true
    },
    {
        name: "9. Complex multi-layered watercolor with bleed and hatch",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    background(248, 246, 240);
    brush.load();
    brush.scaleBrushes(3);
    
    // Layer 1: watercolor fill
    brush.fill('#2a9d8f', 120);
    brush.bleed(0.25, 'out');
    brush.rect(-100, -100, 200, 200);
    
    // Layer 2: hatching
    brush.hatch(10, 45, { rand: true });
    brush.circle(0, 0, 80);
    
    // Layer 3: outline
    brush.set('charcoal', '#264653', 2);
    brush.circle(0, 0, 80);
}
`,
        expectSuccess: true
    },
    {
        name: "10. Forced runtime error (gracefully reports RUNTIME_ERROR without crashing daemon)",
        code: `
function setup() {
    createCanvas(400, 400, WEBGL);
    throw new Error("Intentional sandbox test error");
}
`,
        expectSuccess: false,
        expectedErrorClass: "RUNTIME_ERROR"
    }
];

async function runAdversarialTests() {
    console.log("============================================================");
    console.log("   PAINT-CODE-RL: ADVERSARIAL SANDBOX STRESS TEST SUITE");
    console.log("============================================================");
    let passed = 0;
    let failed = 0;

    for (let i = 0; i < adversarialTests.length; i++) {
        const test = adversarialTests[i];
        console.log(`\n[Test ${i+1}/${adversarialTests.length}] ${test.name}`);
        
        const startTime = Date.now();
        const res = await renderCode(test.code, 100 + i, `adv_test_${i}`);
        const latency = Date.now() - startTime;

        let testPassed = false;
        if (test.expectSuccess) {
            testPassed = res.success === true;
        } else {
            testPassed = !res.success && (!test.expectedErrorClass || res.error_classification === test.expectedErrorClass);
        }

        if (testPassed) {
            console.log(`  -> PASSED (${latency}ms) [Class: ${res.error_classification}]`);
            passed++;
        } else {
            console.log(`  -> FAILED (${latency}ms) [Class: ${res.error_classification}]`);
            if (res.runtime_error) console.log(`     Error: ${res.runtime_error}`);
            failed++;
        }
    }

    console.log("\n============================================================");
    console.log(`RESULTS: ${passed}/${adversarialTests.length} Passed, ${failed} Failed`);
    console.log("============================================================");
    
    await closeBrowser();
    process.exit(failed === 0 ? 0 : 1);
}

runAdversarialTests().catch((err) => {
    console.error("Test runner crash:", err);
    process.exit(1);
});
