const fs = require('fs');

const code = `
function setup() {
  createCanvas(100, 100, WEBGL);
  background(255);
  window.signalRenderComplete();
}
function draw() {}
`;

async function testSequential() {
    console.log("Testing 20 sequential renders...");
    const latencies = [];
    for(let i=0; i<20; i++) {
        const start = Date.now();
        const res = await fetch('http://localhost:3000/render', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt: "test", code: code, seed: 42})
        });
        const data = await res.json();
        latencies.push(Date.now() - start);
        if(!data.success) {
            console.error(`Failed on run ${i}: `, data);
            process.exit(1);
        }
    }
    const avg = latencies.reduce((a,b)=>a+b)/latencies.length;
    console.log(`20 sequential passed. Avg latency: ${avg.toFixed(2)} ms`);
}

async function testConcurrent() {
    console.log("Testing 5 concurrent renders...");
    const promises = [];
    for(let i=0; i<5; i++) {
        promises.push(fetch('http://localhost:3000/render', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({prompt: "test", code: code, seed: 42})
        }).then(r => r.json()));
    }
    const results = await Promise.all(promises);
    const failed = results.filter(r => !r.success);
    if(failed.length > 0) {
        console.error(`${failed.length} concurrent runs failed.`, failed);
        process.exit(1);
    }
    console.log("5 concurrent passed.");
}

async function run() {
    try {
        await testSequential();
        await testConcurrent();
    } catch(e) {
        console.error(e);
        process.exit(1);
    }
}
run();
