const { renderCode, closeBrowser } = require('./sandbox');
const fs = require('fs');
const crypto = require('crypto');

const code = `
function setup() {
  createCanvas(400, 400, WEBGL);
  background(255);
  brush.load();
  brush.noStroke();
  brush.fill(200, 50, 50, 255);
  // Drawing with noise/randomness
  for(let i=0; i<10; i++) {
    brush.rect(random(-50, 50), random(-50, 50), 20, 20);
  }
  window.signalRenderComplete();
}
function draw() {}
`;

function getHash(filePath) {
    const fileBuffer = fs.readFileSync(filePath);
    const hashSum = crypto.createHash('sha256');
    hashSum.update(fileBuffer);
    return hashSum.digest('hex');
}

async function run() {
    console.log("Running determinism test...");
    const hashes = new Set();
    const SEED = 42;
    
    for (let i = 1; i <= 3; i++) {
        const result = await renderCode(code, SEED, `det_test_${i}`);
        if (!result.success) {
            console.error(`Run ${i} failed!`, result);
            await closeBrowser();
            process.exit(1);
        }
        const hash = getHash(result.image_path);
        console.log(`Run ${i} completed. Hash: ${hash}`);
        hashes.add(hash);
    }
    
    if (hashes.size === 1) {
        console.log("SUCCESS: Gate B (Determinism) Passed. All renders identical.");
    } else {
        console.error(`FAILED: Gate B (Determinism). Expected 1 unique hash, got ${hashes.size}.`);
    }
    await closeBrowser();
}

run();
