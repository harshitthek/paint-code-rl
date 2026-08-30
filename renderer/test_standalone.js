const { renderCode, closeBrowser } = require('./sandbox');
const fs = require('fs');

const validCode = `
function setup() {
  createCanvas(400, 400, WEBGL);
  background(255);
  window.signalRenderComplete();
}
function draw() { }
`;

async function run() {
    console.log("Testing standalone renderer...");
    const start = Date.now();
    const result = await renderCode(validCode, 42, 'test_run_1');
    const elapsed = Date.now() - start;
    
    console.log(result);
    console.log("Rendered in " + elapsed + " ms");
    
    if (result.success && result.image_path && fs.existsSync(result.image_path)) {
        console.log("SUCCESS: Image generated at " + result.image_path);
    } else {
        console.error("FAILED to generate image.");
    }
    
    await closeBrowser();
}

run();
