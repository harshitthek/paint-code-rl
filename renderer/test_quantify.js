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
  brush.rect(-50, -50, 100, 100);
  window.signalRenderComplete();
}
function draw() {}
`;

function getHash(filePath) {
    return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

async function run() {
    console.log("Quantifying p5.brush nondeterminism (20 runs)...");
    const hashes = new Set();
    
    for (let i = 1; i <= 20; i++) {
        const result = await renderCode(code, 42, `det_quant_${i}`);
        if(result.success) {
            hashes.add(getHash(result.image_path));
        }
    }
    
    console.log(`Ran 20 times. Unique byte hashes: ${hashes.size}`);
    
    if (hashes.size === 1) {
        console.log("Result: Byte-level deterministic.");
    } else {
        console.log("Result: NOT byte-level deterministic.");
        console.log("Earlier tests established Perceptually Significant variance.");
        console.log("Source: p5.brush internal shader timings or unseeded JS Math.random().");
    }
    await closeBrowser();
}
run();
