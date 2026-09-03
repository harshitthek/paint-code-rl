import os
import re
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent.parent)
secret_patterns = [
    re.compile(r'\bKDAT_[a-zA-Z0-9]{32}\b'),
    re.compile(r'\bhf_[a-zA-Z0-9]{34,}\b'),
    re.compile(r'\bsk-[a-zA-Z0-9]{20,}\b'),
    re.compile(r'\bghp_[a-zA-Z0-9]{36}\b'),
    re.compile(r'\bgho_[a-zA-Z0-9]{36}\b'),
]

SCAN_EXTENSIONS = ('.json', '.yaml', '.yml', '.py', '.md', '.ipynb', '.sh', '.js', '.html', '.toml', '.txt')

found = False
for r, d, files in os.walk(root):
    if '.git' in r or '.venv' in r or '__pycache__' in r or 'node_modules' in r:
        continue
    for f in files:
        if f.endswith(SCAN_EXTENSIONS):
            path = os.path.join(r, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                for pattern in secret_patterns:
                    match = pattern.search(content)
                    if match:
                        print(f"SECRET FOUND IN {path} (Pattern: {match.group(0)[:8]}...)")
                        found = True
                        break

if found:
    print("Secret scan failed: Exposed secrets detected.")
    sys.exit(1)
else:
    print("No secrets found.")
    sys.exit(0)
