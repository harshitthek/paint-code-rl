import os
import re
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent.parent)
secret_pattern = re.compile(r'KDAT_[a-zA-Z0-9]{32}')

found = False
for r, d, files in os.walk(root):
    if '.git' in r or '.venv' in r or '__pycache__' in r:
        continue
    for f in files:
        if f.endswith('.json') or f.endswith('.yaml') or f.endswith('.py') or f.endswith('.md') or f.endswith('.ipynb'):
            path = os.path.join(r, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if secret_pattern.search(content):
                    print(f"SECRET FOUND IN {path}")
                    found = True

if found:
    print("Secret scan failed: Exposed secrets detected.")
    sys.exit(1)
else:
    print("No secrets found.")
    sys.exit(0)
