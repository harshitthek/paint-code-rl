import os
import re
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent.parent)

for r, d, files in os.walk(root):
    if '.git' in r or '.venv' in r or '__pycache__' in r:
        continue
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(r, f)
            rel = os.path.relpath(path, root)
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            for i, line in enumerate(lines):
                if 'import config' in line or 'from config' in line:
                    print(f"Broken config import: {rel}:{i+1} -> {line.strip()}")
                if 'pydantic' in line:
                    print(f"pydantic import: {rel}:{i+1} -> {line.strip()}")
                if 'safetensors' in line:
                    print(f"safetensors import: {rel}:{i+1} -> {line.strip()}")
