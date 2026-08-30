const { renderCode, closeBrowser } = require('./sandbox');
const fs = require('fs');
const crypto = require('crypto');

const code = `
function setup() {
  createCanvas(400, 400, WEBGL);
  background(255);
  fill(200, 50, 50, 255);
  rect(-50, -50, 100, 100);
  window.signalRenderComplete();
}
function draw() {}
`;

function getHash(filePath) {
    const fileBuffer = fs.readFileSync(filePath);
    return crypto.createHash('sha256').update(fileBuffer).digest('hex');
}

async function run() {
    console.log("Running basic WebGL determinism test...");
    const hashes = new Set();
    for (let i = 1; i <= 3; i++) {
        const result = await renderCode(code, 42, `det2_test_${i}`);
        hashes.add(getHash(result.image_path));
    }
    console.log("Basic WebGL Hashes:", hashes);
    await closeBrowser();
}
run();
