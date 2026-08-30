const { renderCode, closeBrowser } = require('./sandbox');
const fs = require('fs');
const PNG = require('pngjs').PNG;
const pixelmatch = require('pixelmatch');

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

function readPNG(filePath) {
    return PNG.sync.read(fs.readFileSync(filePath));
}

async function run() {
    console.log("Running perceptual determinism test...");
    
    const result1 = await renderCode(code, 42, `det_perc_1`);
    const result2 = await renderCode(code, 42, `det_perc_2`);
    
    if(!result1.success || !result2.success) {
        console.error("Rendering failed.");
        await closeBrowser();
        return;
    }
    
    const img1 = readPNG(result1.image_path);
    const img2 = readPNG(result2.image_path);
    
    const {width, height} = img1;
    const diff = new PNG({width, height});
    
    // Strict pixel equality first
    let numDiffPixels = pixelmatch(img1.data, img2.data, diff.data, width, height, {threshold: 0.0});
    console.log(`Strict pixel equality: ${numDiffPixels} pixels differ (threshold 0.0).`);
    
    if (numDiffPixels > 0) {
        // Perceptual similarity
        numDiffPixels = pixelmatch(img1.data, img2.data, diff.data, width, height, {threshold: 0.1});
        console.log(`Perceptual similarity: ${numDiffPixels} pixels differ (threshold 0.1).`);
    }
    
    await closeBrowser();
}
run();
