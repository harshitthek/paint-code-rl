const { renderCode, closeBrowser } = require('./sandbox');

async function run() {
    console.log("Running Security Smoke Test...");
    
    const cases = [
        {
            name: "Infinite Loop",
            code: "function setup() { while(true){} }",
            expect: "TIMEOUT"
        },
        {
            name: "Invalid JavaScript",
            code: "function setup() { let x = ; }",
            expect: "PARSE_ERROR"
        },
        {
            name: "Missing Canvas",
            code: "function setup() { let c = document.querySelector('canvas'); if(c) c.remove(); window.signalRenderComplete(); }",
            expect: "NO_CANVAS"
        },
        {
            name: "Runtime Exception",
            code: "function setup() { createCanvas(100,100,WEBGL); throw new Error('crash'); window.signalRenderComplete(); }",
            expect: "RUNTIME_ERROR" // Modified because it catches the error and exits with RUNTIME_ERROR now.
        }
    ];
    
    let allPassed = true;
    for(const c of cases) {
        console.log(`Testing: ${c.name}`);
        const result = await renderCode(c.code, 42, 'sec_test');
        if(result.error_classification !== c.expect) {
            console.error(`  FAIL: Expected ${c.expect}, got ${result.error_classification}`, result);
            allPassed = false;
        } else {
            console.log(`  PASS: Got expected error ${c.expect}`);
        }
    }
    
    if(allPassed) {
        console.log("SUCCESS: All security smoke tests passed.");
    }
    await closeBrowser();
}
run();
