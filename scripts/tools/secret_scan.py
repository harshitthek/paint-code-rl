import os
import re

root = r"C:\Users\user\.gemini\antigravity\brain\eabfab2e-f626-4128-9da1-6868c5d0f842\paint-code-rl"
secret_pattern = re.compile(r'KDAT_[a-zA-Z0-9]{32}')

found = False
for r, d, files in os.walk(root):
    if '.git' in r:
        continue
    for f in files:
        if f.endswith('.json') or f.endswith('.yaml') or f.endswith('.py') or f.endswith('.md') or f.endswith('.ipynb'):
            path = os.path.join(r, f)
            with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                content = file.read()
                if secret_pattern.search(content):
                    print(f"SECRET FOUND IN {path}")
                    found = True

if not found:
    print("No secrets found.")
