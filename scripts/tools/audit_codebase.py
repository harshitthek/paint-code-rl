import os
import re
import ast
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent.parent)

print("=== 1. CHECKING SYNTAX OF ALL PYTHON FILES ===")
py_files = []
for r, d, files in os.walk(root):
    if '.git' in r or '.venv' in r or '__pycache__' in r:
        continue
    for f in files:
        if f.endswith('.py'):
            py_files.append(os.path.join(r, f))

syntax_errors = 0
for path in py_files:
    rel = os.path.relpath(path, root)
    try:
        with open(path, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        print(f"[OK] Syntax OK: {rel}")
    except Exception as e:
        print(f"[ERROR] SYNTAX ERROR in {rel}: {e}")
        syntax_errors += 1

print("\n=== 2. CHECKING CUDA-HARDCODED CHECKS ===")
cuda_pattern = re.compile(r'torch\.cuda\.(is_available|device_count|get_device_properties)')
for path in py_files:
    rel = os.path.relpath(path, root)
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if cuda_pattern.search(line) and 'device.type' not in line and 'get_compute_capabilities' not in line and 'kaggle' not in rel:
            print(f"[WARN] {rel}:{i+1} -> {line.strip()}")

print("\n=== 3. CHECKING IMPORTS IN PYTHON FILES ===")
import_pattern = re.compile(r'^\s*(?:from|import)\s+([a-zA-Z0-9_\.]+)')
found_imports = set()
for path in py_files:
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            m = import_pattern.match(line)
            if m:
                top_pkg = m.group(1).split('.')[0]
                found_imports.add(top_pkg)

import sys
print("Detected top-level imports across codebase:")
for imp in sorted(found_imports):
    print(f" - {imp}")

if syntax_errors > 0:
    print(f"\n[FAIL] Codebase audit failed: {syntax_errors} syntax errors found.")
    sys.exit(1)
else:
    print("\n[OK] Codebase audit passed!")
