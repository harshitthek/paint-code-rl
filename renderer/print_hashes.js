const fs = require('fs');
const crypto = require('crypto');
const path = require('path');

function getHash(filePath) {
    if (!fs.existsSync(filePath)) return "MISSING";
    return crypto.createHash('sha256').update(fs.readFileSync(filePath)).digest('hex');
}

console.log("=== Environment Asset Hashes ===");
console.log("p5.min.js:      ", getHash(path.join(__dirname, 'assets/p5.min.js')));
console.log("p5.brush.min.js:", getHash(path.join(__dirname, 'assets/p5.brush.min.js')));
console.log("template.html:  ", getHash(path.join(__dirname, 'template.html')));
console.log("================================");
