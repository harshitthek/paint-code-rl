const puppeteer = require('puppeteer');
const path = require('path');
const fs = require('fs');
const vm = require('vm');

function balanceTokens(jsCode) {
    if (!jsCode) return "";
    let openP = (jsCode.match(/\(/g) || []).length;
    let closeP = (jsCode.match(/\)/g) || []).length;
    if (openP > closeP) jsCode += ")".repeat(openP - closeP) + ";";

    let openB = (jsCode.match(/\{/g) || []).length;
    let closeB = (jsCode.match(/\}/g) || []).length;
    if (openB > closeB) jsCode += "\n" + "}".repeat(openB - closeB);

    return jsCode;
}

function ensureCanvasAndSetup(code) {
    if (!code) return code;
    if (!code.includes("setup") && code.includes("draw")) {
        return "function setup() {\n  createCanvas(600, 600, WEBGL);\n}\n" + code;
    }
    if (!code.includes("setup") && !code.includes("draw")) {
        return "function setup() {\n  createCanvas(600, 600, WEBGL);\n" + code + "\n}";
    }
    if (!code.includes("createCanvas") && code.includes("setup")) {
        return code.replace(/(function\s+setup\s*\(\)\s*\{)/, '$1\n  createCanvas(600, 600, WEBGL);\n');
    }
    return code;
}

function autoRepairJS(jsCode) {
    if (!jsCode) return jsCode;
    let candidate = ensureCanvasAndSetup(balanceTokens(jsCode));
    try {
        new vm.Script(candidate);
        return candidate;
    } catch (e) {}

    let lines = jsCode.split("\n");
    while (lines.length > 2) {
        lines.pop();
        let cand = ensureCanvasAndSetup(balanceTokens(lines.join("\n")));
        try {
            new vm.Script(cand);
            return cand;
        } catch (e) {}
    }
    return candidate;
}

function extractAndRemoveFunction(code, funcName) {
    if (!code) return { code, body: null };
    const regex = new RegExp(`function\\s+${funcName}\\s*\\([^)]*\\)\\s*\\{`);
    const match = regex.exec(code);
    if (!match) return { code, body: null };

    const startIndex = match.index;
    const bodyStartIndex = startIndex + match[0].length;
    let depth = 1;
    let inString = null;
    let inSingleComment = false;
    let inMultiComment = false;
    let i = bodyStartIndex;

    while (i < code.length && depth > 0) {
        const char = code[i];
        const nextChar = code[i + 1] || '';

        if (inSingleComment) {
            if (char === '\n') inSingleComment = false;
        } else if (inMultiComment) {
            if (char === '*' && nextChar === '/') {
                inMultiComment = false;
                i++;
            }
        } else if (inString) {
            if (char === '\\') {
                i++; // Skip escaped character
            } else if (char === inString) {
                inString = null;
            }
        } else {
            if (char === '/' && nextChar === '/') {
                inSingleComment = true;
                i++;
            } else if (char === '/' && nextChar === '*') {
                inMultiComment = true;
                i++;
            } else if (char === '"' || char === "'" || char === '`') {
                inString = char;
            } else if (char === '{') {
                depth++;
            } else if (char === '}') {
                depth--;
            }
        }
        i++;
    }

    if (depth === 0) {
        const body = code.substring(bodyStartIndex, i - 1);
        const newCode = code.substring(0, startIndex) + `// [inlined ${funcName}]\n` + code.substring(i);
        return { code: newCode, body };
    }
    return { code, body: null };
}

function hoistSetupVariables(jsCode) {
    if (!jsCode) return jsCode;
    const setupMatch = jsCode.match(/(function\s+setup\s*\(\)\s*\{)([\s\S]*?)(\n\})/);
    if (!setupMatch) return jsCode;

    let setupHeader = setupMatch[1];
    let setupBody = setupMatch[2];
    let setupFooter = setupMatch[3];

    // Collect declared variable names in setup
    let varNames = [];
    let declRegex = /^[ \t]*(?:let|const|var)\s+([a-zA-Z0-9_$]+)\s*=/gm;
    let m;
    while ((m = declRegex.exec(setupBody)) !== null) {
        if (!['width', 'height', 'brush', 'p5', 'window', 'camera', 'setup', 'draw', 'preload'].includes(m[1])) {
            varNames.push(m[1]);
        }
    }

    if (varNames.length === 0) return jsCode;

    // Convert `let x =` or `const x =` to `window.x = x =`
    let transformedBody = setupBody.replace(/^[ \t]*(?:let|const|var)\s+([a-zA-Z0-9_$]+)\s*=/gm, (match, varName) => {
        if (!['width', 'height', 'brush', 'p5', 'window', 'camera', 'setup', 'draw', 'preload'].includes(varName)) {
            return `  window.${varName} = ${varName} =`;
        }
        return match;
    });

    let topDeclarations = `var ${varNames.join(', ')};\n`;
    return topDeclarations + jsCode.replace(setupMatch[0], setupHeader + transformedBody + setupFooter);
}

let browser;

async function initBrowser() {
    if (browser) return browser;
    const args = [
        '--no-sandbox',
        '--disable-setuid-sandbox',
        '--disable-dev-shm-usage',
        '--enable-webgl',
        '--allow-file-access-from-files',
    ];

    // Platform-specific WebGL backend selection
    if (process.platform === 'darwin' && process.arch === 'arm64') {
        // macOS ARM64 (Apple Silicon): use native Metal GPU via ANGLE
        args.push('--use-gl=angle');
        args.push('--use-angle=metal');
        args.push('--disable-gpu-sandbox');
    } else if (process.platform === 'linux') {
        // Linux (Kaggle/Docker): use SwiftShader software rasterizer
        args.push('--use-gl=angle');
        args.push('--use-angle=swiftshader-webgl');
    } else {
        // Windows / other: use SwiftShader as safe fallback
        args.push('--use-gl=angle');
        args.push('--use-angle=swiftshader-webgl');
    }

    const launchOptions = {
        headless: 'new',
        args: args
    };

    if (process.env.PUPPETEER_EXECUTABLE_PATH && fs.existsSync(process.env.PUPPETEER_EXECUTABLE_PATH)) {
        launchOptions.executablePath = process.env.PUPPETEER_EXECUTABLE_PATH;
    } else if (fs.existsSync('/usr/bin/google-chrome')) {
        launchOptions.executablePath = '/usr/bin/google-chrome';
    } else if (fs.existsSync('/usr/bin/google-chrome-stable')) {
        launchOptions.executablePath = '/usr/bin/google-chrome-stable';
    }
    // Note: Never use /usr/bin/chromium-browser on Linux as Ubuntu packages it as an unusable snap stub

    browser = await puppeteer.launch(launchOptions);
    return browser;
}

async function renderCode(code, seed, runId, options = {}) {
    // Security: sanitize runId to prevent directory traversal
    const safeRunId = String(runId || 'render_' + Date.now()).replace(/[^a-zA-Z0-9_-]/g, '_');
    
    const timeouts = {
        browser_startup: options.browser_startup_timeout_ms || 10000,
        page_load: options.page_load_timeout_ms || 5000,
        code_execution: options.code_execution_timeout_ms || 4000,
        screenshot: options.screenshot_timeout_ms || 3000
    };

    const b = await initBrowser();
    const page = await b.newPage();
    page.setDefaultTimeout(timeouts.page_load);
    
    await page.setRequestInterception(true);
    const rendererDir = path.resolve(__dirname);
    page.on('request', request => {
        try {
            const url = request.url();
            if (url.startsWith('data:')) {
                request.continue();
            } else if (url.startsWith('file://')) {
                try {
                    let fileUrlPath = new URL(url).pathname;
                    if (process.platform === 'win32' && fileUrlPath.startsWith('/')) {
                        fileUrlPath = fileUrlPath.slice(1);
                    }
                    const resolvedFile = path.resolve(decodeURIComponent(fileUrlPath));
                    const rel = path.relative(rendererDir, resolvedFile);
                    if (!rel.startsWith('..') && !path.isAbsolute(rel)) {
                        request.continue();
                    } else {
                        request.abort('accessdenied');
                    }
                } catch (e) {
                    request.abort('accessdenied');
                }
            } else {
                request.abort('accessdenied');
            }
        } catch (err) {
            // Request may have already been handled
        }
    });

    let error_classification = 'SUCCESS';
    let runtime_error = null;
    let console_logs = [];

    page.on('console', msg => {
        console_logs.push(`[${msg.type()}] ${msg.text()}`);
    });

    page.on('pageerror', err => {
        const errStr = err.toString();
        if (runtime_error) {
            runtime_error += "\n" + errStr;
        } else {
            runtime_error = errStr;
        }
        if (errStr.includes('SyntaxError')) {
            error_classification = 'PARSE_ERROR';
        } else {
            error_classification = 'RUNTIME_ERROR';
        }
    });

    const tmpFile = path.resolve(__dirname, `tmp_${safeRunId}.html`);

    try {
        const templatePath = path.resolve(__dirname, 'template.html');
        let htmlContent = fs.readFileSync(templatePath, 'utf8');
        
        let seedHook = '';
        if (seed !== undefined && seed !== null) {
            seedHook = `
(function(s) {
    let _seed = s >>> 0;
    Math.random = function() {
        _seed |= 0; _seed = _seed + 0x6D2B79F5 | 0;
        let t = Math.imul(_seed ^ _seed >>> 15, 1 | _seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
})(${seed});
`;
        }
        htmlContent = htmlContent.replace('// SEED_INJECTION_HOOK', seedHook);

        // Strip ES6 module imports and CommonJS requires so vm/eval never throws Cannot use import statement outside a module
        let preprocessedCode = code
            .replace(/^[ \t]*import\s+[\s\S]*?from\s+['"][^'"]+['"];?/gm, '// [stripped import]\n')
            .replace(/^[ \t]*import\s+['"][^'"]+['"];?/gm, '// [stripped import]\n')
            .replace(/^[ \t]*import\s+[^;\n]+;?/gm, '// [stripped import]\n')
            .replace(/^[ \t]*(?:const|let|var)\s+[^=]+=\s*require\([^)]*\);?/gm, '// [stripped require]\n')
            .replace(/^[ \t]*export\s+(?:default\s+)?/gm, '// [stripped export] ');

        let safeCode = autoRepairJS(preprocessedCode);
        // Heal accidental brush re-declarations (e.g. let brush; or const brush = new p5.Brush())
        safeCode = safeCode.replace(
            /^[ \t]*(?:const|let|var)\s+brush\s*(?:=\s*(?:new\s+(?:p5\.)?Brush\([^)]*\)|brush|window\.brush|[^;\n]+))?;?/gmi,
            '// [auto-healed brush declaration]\n'
        );
        safeCode = safeCode.replace(
            /^[ \t]*brush\s*=\s*new\s+(?:p5\.)?Brush\([^)]*\);?/gmi,
            '// [auto-healed brush assignment]\n'
        );

        // Replace external asset loaders with instant procedural dummy values (prevents preload hangs)
        safeCode = safeCode.replace(/loadImage\s*\([^)]*\)/g, 'createImage(100, 100)');
        safeCode = safeCode.replace(/loadFont\s*\([^)]*\)/g, '""');
        safeCode = safeCode.replace(/loadJSON\s*\([^)]*\)/g, '{}');
        safeCode = safeCode.replace(/loadStrings\s*\([^)]*\)/g, '[]');
        safeCode = safeCode.replace(/loadSound\s*\([^)]*\)/g, '{}');

        // Inline preload() body into setup() so variables (e.g. mountains = [], clouds = []) are initialized
        // without risking premature brush.load() failure before createCanvas()
        const extractedPreload = extractAndRemoveFunction(safeCode, 'preload');
        if (extractedPreload.body) {
            safeCode = extractedPreload.code;
            const cleanedPreload = extractedPreload.body
                .replace(/brush\s*=\s*new[^;\n]+;?/g, '')
                .replace(/brush\.load\(\);?/g, '')
                .replace(/brush\.scaleBrushes\([^)]*\);?/g, '');
            if (safeCode.includes("createCanvas")) {
                safeCode = safeCode.replace(
                    /(createCanvas\s*\([^)]*\)\s*;?)/,
                    `$1\n  ${cleanedPreload}\n`
                );
            } else {
                safeCode = safeCode.replace(/(function\s+setup\s*\(\)\s*\{)/, `$1\n  ${cleanedPreload}\n`);
            }
        }

        // Hoist setup-scoped variables so they are accessible in draw()
        safeCode = hoistSetupVariables(safeCode);

        if (seed !== undefined && seed !== null) {
            safeCode = safeCode.replace(
                /function\s+setup\s*\(\)\s*\{/,
                "function setup() { try { randomSeed(" + seed + "); noiseSeed(" + seed + "); } catch(e){} "
            );
        }
        
        // Automated Lifecycle & WebGL Guard Hook
        const autoSignalWrapper = `
// Intercept createCanvas to setup brush automatically
(function() {
    function makeOmniProxy(fn) {
        const handler = {
            get(target, prop) {
                if (prop in target) {
                    const v = target[prop];
                    return typeof v === 'function' ? v.bind(target) : v;
                }
                return makeOmniProxy(function() {});
            },
            apply(target, thisArg, args) {
                try {
                    if (typeof target === 'function') return target.apply(thisArg, args);
                } catch(e) {}
                return makeOmniProxy(function() {});
            }
        };
        return new Proxy(fn || function() {}, handler);
    }

    const EXACT_CASE_MAP = {
        'hb': 'HB',
        '2b': '2B',
        '2h': '2H',
        'cpencil': 'cpencil',
        'pen': 'pen',
        'rotring': 'rotring',
        'spray': 'spray',
        'marker': 'marker',
        'marker2': 'marker2',
        'charcoal': 'charcoal',
        'hatch_brush': 'hatch_brush',
    };
    const BRUSH_TYPO_MAP = {
        'charcol': 'charcoal',
        'charcole': 'charcoal',
        'pencil': 'cpencil',
        'colorpencil': 'cpencil',
        'color_pencil': 'cpencil',
        'water_color': 'marker',
        'watercolor': 'marker',
        'water': 'marker',
        'ink': 'pen',
        'brush': 'charcoal',
        'default': 'HB',
        'hatch': 'hatch_brush',
        'hatching': 'hatch_brush',
        'airbrush': 'spray',
        'spraypaint': 'spray',
    };
    function normalizeBrushName(name) {
        if (!name || typeof name !== 'string') return 'charcoal';
        const lower = name.toLowerCase().trim();
        const mapped = BRUSH_TYPO_MAP[lower] || lower;
        return EXACT_CASE_MAP[mapped.toLowerCase()] || 'charcoal';
    }

    function makeBrushProxy(target) {
        if (!target) return target;
        if (typeof target.set === 'function' && !target.__set_patched) {
            const _origSet = target.set;
            target.set = function(name, c, ...rest) {
                if (typeof name === 'function') name = 'charcoal';
                name = normalizeBrushName(name);
                if (typeof c === 'function' || !c) c = '#3a6c73';
                return _origSet.apply(this, [name, c, ...rest]);
            };
            target.__set_patched = true;
        }
        if (typeof target.fill === 'function' && !target.__fill_patched) {
            const _origFill = target.fill;
            target.fill = function(c, ...rest) {
                if (typeof c === 'function' || !c) c = '#1a759f';
                return _origFill.apply(this, [c, ...rest]);
            };
            target.__fill_patched = true;
        }
        if (typeof target.bleed === 'function' && typeof target.fillBleed === 'undefined') {
            target.fillBleed = target.bleed;
        }
        if (typeof target.strokeWeight === 'undefined') {
            target.strokeWeight = function(w) { target.w = w; };
        }
        if (typeof target.noFill === 'undefined') {
            target.noFill = function() {};
        }
        return new Proxy(target, {
            get(t, prop, receiver) {
                if (prop in t) {
                    const val = Reflect.get(t, prop, receiver);
                    return typeof val === 'function' ? val.bind(t) : val;
                }
                return function(...args) { return t; };
            }
        });
    }

    if (typeof window !== 'undefined') {
        let _activeCamera = makeOmniProxy(window.camera || function() {});
        try {
            Object.defineProperty(window, 'camera', {
                get() { return _activeCamera; },
                set(val) { _activeCamera = makeOmniProxy(val); },
                configurable: true,
                enumerable: true
            });
        } catch(e) { window.camera = _activeCamera; }

        let _activeBrush = (typeof window.brush !== 'undefined') ? makeBrushProxy(window.brush) : null;
        try {
            Object.defineProperty(window, 'brush', {
                get() { return _activeBrush || {}; },
                set(val) { _activeBrush = makeBrushProxy(val); },
                configurable: true,
                enumerable: true
            });
        } catch(e) { window.brush = _activeBrush || {}; }

        window.Brush = function() { return window.brush || {}; };
        window.p5 = window.p5 || {};
        window.p5.Brush = function() { return window.brush || {}; };
        window.loadImage = function(p, s, f) {
            try {
                const img = (typeof createImage === 'function') ? createImage(100, 100) : { width: 100, height: 100 };
                if (typeof s === 'function') s(img);
                return img;
            } catch(e) { return { width: 100, height: 100 }; }
        };
        
        // Math and casing constants
        const mathConsts = {
            Two_PI: Math.PI * 2,
            TwoPi: Math.PI * 2,
            two_pi: Math.PI * 2,
            Two_Pi: Math.PI * 2,
            TWO_PI: Math.PI * 2,
            Half_PI: Math.PI / 2,
            HalfPi: Math.PI / 2,
            half_pi: Math.PI / 2,
            Half_Pi: Math.PI / 2,
            HALF_PI: Math.PI / 2,
            Quarter_PI: Math.PI / 4,
            QuarterPi: Math.PI / 4,
            quarter_pi: Math.PI / 4,
            Quarter_Pi: Math.PI / 4,
            QUARTER_PI: Math.PI / 4,
            TAU: Math.PI * 2,
            Tau: Math.PI * 2,
            tau: Math.PI * 2,
            PI: Math.PI,
            Pi: Math.PI,
            pi: Math.PI,
        };
        for (const [k, v] of Object.entries(mathConsts)) {
            window[k] = v;
        }

        // Lighting stubs
        ['ambientLight', 'pointLight', 'directionalLight', 'spotLight', 'lights', 'noLights'].forEach(fn => {
            if (typeof window[fn] === 'undefined') window[fn] = function() {};
        });

        // Function aliases for casing hallucinations (e.g. lERPColor -> lerpColor)
        const _dummyLerpColor = function(c1, c2, amt) {
            try {
                if (typeof window.lerpColor === 'function') return window.lerpColor(c1, c2, amt);
                if (typeof p5 !== 'undefined' && p5.prototype && typeof p5.prototype.lerpColor === 'function') {
                    return p5.prototype.lerpColor.call(this, c1, c2, amt);
                }
            } catch(e) {}
            return c1 || '#1a759f';
        };
        window.lERPColor = _dummyLerpColor;
        window.LerpColor = _dummyLerpColor;
        window.lerp_color = _dummyLerpColor;
        window.Lerp = typeof window.lerp === 'function' ? window.lerp : function(a, b, t) { return a + (b - a) * t; };

        // Numerical constants so unquoted word numbers never throw ReferenceError
        window.ZERO = 0; window.Zero = 0; window.zero = 0;
        window.ONE = 1; window.One = 1; window.one = 1;
        window.TWO = 2; window.Two = 2; window.two = 2;
        window.THREE = 3; window.Three = 3; window.three = 3;
        window.FOUR = 4; window.Four = 4; window.four = 4;
        window.FIVE = 5; window.Five = 5; window.five = 5;
        window.SIX = 6; window.Six = 6; window.six = 6;
        window.SEVEN = 7; window.Seven = 7; window.seven = 7;
        window.EIGHT = 8; window.Eight = 8; window.eight = 8;
        window.NINE = 9; window.Nine = 9; window.nine = 9;
        window.TEN = 10; window.Ten = 10; window.ten = 10;

        // Coordinate, delta, and geometric fallbacks
        window.dx = 0; window.dy = 0; window.dz = 0;
        window.vx = 0; window.vy = 0; window.vz = 0;
        window.ax = 0; window.ay = 0; window.az = 0;
        window.step = 10; window.steps = 10;
        window.count = 10; window.num = 10; window.total = 10;
        window.scale = 1; window.scaleFactor = 1;
        window.rotation = 0; window.rot = 0;
        window.offset = 0; window.spacing = 10;
        window.padding = 10; window.margin = 10;
        window.depth = 100; window.len = 100; window.length = 100;

        // Fallback variables so unquoted parameter identifiers never throw ReferenceError
        window.strength = 0.2;
        window.strenght = 0.2;
        window.bleed = 0.2;
        window.opacity = 160;
        window.alpha = 160;
        window.density = 1;
        window.angle = 0;
        window.radius = 50;
        window.size = 20;
        window.speed = 1;
        window.weight = 2;
        window.brushSize = 2;
        window.color = '#1a759f';
        window.col = '#1a759f';
        window.x = 300;
        window.y = 300;
        window.w = 600;
        window.h = 600;
        window.out = 'out';
        window.in = 'in';
        window.inside = 'in';
        window.outside = 'out';
        window.charcoal = 'charcoal';
        window.pen = 'pen';
        window.rotring = 'rotring';
        window.spray = 'spray';
        window.marker = 'marker';
        window.marker2 = 'marker2';
        window.HB = 'HB';
        window.cpencil = 'cpencil';
        window.hatch_brush = 'hatch_brush';
        window.center = 'center';
        window.CENTER = 'center';
    }

    if (typeof window.p5 !== 'undefined' && window.p5.prototype) {
        window.p5.prototype.Brush = function() { return window.brush || {}; };
        
        // Robust color parser that intercepts functions, nulls, and arguments objects
        const _origP5Color = window.p5.prototype.color;
        window.p5.prototype.color = function(...args) {
            if (args.length === 0 || args[0] === undefined || args[0] === null || typeof args[0] === 'function' || (typeof args[0] === 'object' && args[0].toString() === '[object Arguments]')) {
                return _origP5Color.call(this, '#1a759f');
            }
            try {
                return _origP5Color.apply(this, args);
            } catch(e) {
                return _origP5Color.call(this, '#1a759f');
            }
        };

        // Math constants on p5 prototype
        const mathConsts = {
            Two_PI: Math.PI * 2,
            TwoPi: Math.PI * 2,
            two_pi: Math.PI * 2,
            Two_Pi: Math.PI * 2,
            TWO_PI: Math.PI * 2,
            Half_PI: Math.PI / 2,
            HalfPi: Math.PI / 2,
            half_pi: Math.PI / 2,
            Half_Pi: Math.PI / 2,
            HALF_PI: Math.PI / 2,
            Quarter_PI: Math.PI / 4,
            QuarterPi: Math.PI / 4,
            quarter_pi: Math.PI / 4,
            Quarter_Pi: Math.PI / 4,
            QUARTER_PI: Math.PI / 4,
            TAU: Math.PI * 2,
            Tau: Math.PI * 2,
            tau: Math.PI * 2,
            PI: Math.PI,
            Pi: Math.PI,
            pi: Math.PI,
        };
        for (const [k, v] of Object.entries(mathConsts)) {
            window.p5.prototype[k] = v;
        }

        // Numerical constants on p5 prototype
        const numConsts = {
            ZERO: 0, Zero: 0, zero: 0,
            ONE: 1, One: 1, one: 1,
            TWO: 2, Two: 2, two: 2,
            THREE: 3, Three: 3, three: 3,
            FOUR: 4, Four: 4, four: 4,
            FIVE: 5, Five: 5, five: 5,
            SIX: 6, Six: 6, six: 6,
            SEVEN: 7, Seven: 7, seven: 7,
            EIGHT: 8, Eight: 8, eight: 8,
            NINE: 9, Nine: 9, nine: 9,
            TEN: 10, Ten: 10, ten: 10
        };
        for (const [k, v] of Object.entries(numConsts)) {
            window.p5.prototype[k] = v;
        }

        // Lighting stubs on p5 prototype
        ['ambientLight', 'pointLight', 'directionalLight', 'spotLight', 'lights', 'noLights'].forEach(fn => {
            if (typeof window.p5.prototype[fn] === 'undefined') window.p5.prototype[fn] = function() {};
        });

        // Function aliases on p5 prototype
        window.p5.prototype.lERPColor = window.p5.prototype.lerpColor;
        window.p5.prototype.LerpColor = window.p5.prototype.lerpColor;
        window.p5.prototype.lerp_color = window.p5.prototype.lerpColor;
        window.p5.prototype.Lerp = window.p5.prototype.lerp;

        // Safe loadImage fallback to prevent preload hangs
        window.p5.prototype.loadImage = function(path, success, failure) {
            try {
                const img = this.createImage(100, 100);
                if (typeof success === 'function') success(img);
                return img;
            } catch(e) {
                return {};
            }
        };
        // Safe image() drawer that ignores null/undefined
        const _origImage = window.p5.prototype.image;
        window.p5.prototype.image = function(img, ...args) {
            if (!img) return;
            try {
                return _origImage.apply(this, [img, ...args]);
            } catch(e) {}
        };
        const _origCreateCanvas2 = window.p5.prototype.createCanvas;
        window.p5.prototype.createCanvas = function(...args) {
            const result = _origCreateCanvas2.apply(this, args);
            if (typeof window.brush !== 'undefined') {
                if (typeof window.brush.load === 'function') {
                    try { window.brush.load(); } catch(e) {}
                }
                if (typeof window.brush.scaleBrushes === 'function') {
                    try { window.brush.scaleBrushes(3); } catch(e) {}
                }
            }
            return result;
        };
    }
})();
// Omnipresent Lexical Proxy: intercepts ANY undeclared identifier (e.g. mist_size, TWO, etc.)
(function() {
    if (typeof window === 'undefined') return;
    var _sandboxProxy = new Proxy(window, {
        has(target, prop) {
            if (prop === Symbol.unscopables) return false;
            return true;
        },
        get(target, prop) {
            if (prop in target) {
                const v = target[prop];
                if (prop === 'camera' || prop === 'brush' || prop === 'p5') return v;
                return typeof v === 'function' ? v.bind(target) : v;
            }
            if (typeof prop === 'string') {
                if (prop.endsWith('_size') || prop.endsWith('Size') || prop.includes('size')) return 20;
                if (prop.endsWith('_color') || prop.endsWith('Color') || prop.includes('color') || prop.includes('col')) return '#1a759f';
                if (prop.endsWith('_alpha') || prop.endsWith('alpha') || prop.endsWith('opacity')) return 160;
                if (prop.endsWith('_count') || prop.endsWith('Count') || prop.includes('num')) return 10;
                if (prop.endsWith('_width') || prop.endsWith('Width') || prop.includes('width')) return 100;
                if (prop.endsWith('_height') || prop.endsWith('Height') || prop.includes('height')) return 100;
                if (prop.endsWith('_radius') || prop.endsWith('Radius') || prop.includes('radius')) return 50;
                if (prop.endsWith('_speed') || prop.endsWith('Speed')) return 1;
                if (prop.endsWith('_angle') || prop.endsWith('Angle')) return 0;
                if (prop.endsWith('_spacing') || prop.endsWith('Spacing')) return 10;
            }
            return 10;
        },
        set(target, prop, value) {
            target[prop] = value;
            return true;
        }
    });

    with (_sandboxProxy) {
${safeCode}
        if (typeof setup === 'function') window.setup = setup;
        if (typeof draw === 'function') window.draw = draw;
        if (typeof preload === 'function') window.preload = preload;
    }
})();

// Auto-signal completion after draw() or setup() has completed painting the canvas
(function() {
    if (typeof window !== 'undefined') {
        if (typeof window.draw === 'function') {
            const _userDraw = window.draw;
            window.draw = function(...args) {
                const res = _userDraw.apply(this, args);
                setTimeout(function() {
                    window.renderComplete = true;
                }, 400);
                return res;
            };
        } else if (typeof window.setup === 'function') {
            const _userSetup = window.setup;
            window.setup = function(...args) {
                const res = _userSetup.apply(this, args);
                setTimeout(function() {
                    window.renderComplete = true;
                }, 400);
                return res;
            };
        }
    }
})();

// Fallback timeout to complete rendering
setTimeout(function() {
    if (!window.renderComplete) {
        window.renderComplete = true;
    }
}, ${timeouts.code_execution});
`;
        
        htmlContent = htmlContent.replace('// INJECT_CODE_HERE', autoSignalWrapper);
        
        fs.writeFileSync(tmpFile, htmlContent);

        const fileUrl = 'file://' + tmpFile;
        await page.goto(fileUrl, { waitUntil: 'load' });
        
        try {
            await page.waitForFunction('window.renderComplete === true', { timeout: timeouts.code_execution + 500 });
        } catch(e) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'TIMEOUT';
            }
        }
        
        // Ensure canvas exists and is visible
        const canvas = await page.$('canvas');
        if (!canvas) {
            if (error_classification === 'SUCCESS') {
                error_classification = 'NO_CANVAS';
            }
            throw new Error("NO_CANVAS");
        }
        
        let outPath = null;
        let image_base64 = null;
        
        if (options.return_base64) {
            // In-memory base64 capture — avoids disk I/O during RL training
            try {
                image_base64 = await canvas.screenshot({ encoding: 'base64', timeout: timeouts.screenshot });
            } catch(e) {
                if (e.message && e.message.includes('Node is either not visible')) {
                    error_classification = 'NO_CANVAS';
                    throw new Error("NO_CANVAS");
                } else {
                    throw e;
                }
            }
        } else {
            // Disk-backed capture — for tests, showcase, and backward compatibility
            const outDir = path.resolve(__dirname, '../artifacts/renders');
            if (!fs.existsSync(outDir)) fs.mkdirSync(outDir, { recursive: true });
            outPath = path.join(outDir, `${safeRunId}.png`);
            try {
                await canvas.screenshot({ path: outPath, timeout: timeouts.screenshot });
            } catch(e) {
                if (e.message && e.message.includes('Node is either not visible')) {
                    error_classification = 'NO_CANVAS';
                    throw new Error("NO_CANVAS");
                } else {
                    throw e;
                }
            }
        }
        
        const result = {
            success: error_classification === 'SUCCESS',
            image_path: outPath,
            error_classification: error_classification,
            runtime_error: runtime_error,
            console_logs: console_logs
        };
        if (image_base64) {
            result.image_base64 = image_base64;
        }
        return result;

    } catch (e) {
        if (error_classification === 'SUCCESS') {
            if (e.toString().includes('Navigation timeout') || e.toString().includes('TIMEOUT')) error_classification = 'TIMEOUT';
            else error_classification = 'BROWSER_ERROR';
        }
        return {
            success: false,
            image_path: null,
            error_classification: error_classification,
            runtime_error: e.toString() + (runtime_error ? " | " + runtime_error : ""),
            console_logs: typeof console_logs !== 'undefined' ? console_logs : []
        };
    } finally {
        // Guaranteed disk and memory resource cleanup
        try {
            if (fs.existsSync(tmpFile)) {
                fs.unlinkSync(tmpFile);
            }
        } catch (e) {}
        await page.close().catch(() => {});
    }
}

async function closeBrowser() {
    if (browser) {
        await browser.close().catch(() => {});
        browser = null;
    }
}

module.exports = { initBrowser, renderCode, closeBrowser };
