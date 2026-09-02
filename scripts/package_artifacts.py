#!/usr/bin/env python3
"""Package training artifacts (checkpoints, renders, logs, dashboard) for easy transfer.

Usage:
    python scripts/package_artifacts.py [--format zip|tar.gz] [--output <path>]
"""
import os
import sys
import shutil
import tarfile
import zipfile
import argparse
from datetime import datetime

# Find repo root
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))


def main():
    parser = argparse.ArgumentParser(description="Package training artifacts for transfer")
    parser.add_argument(
        "--format",
        choices=["zip", "tar.gz"],
        default="zip",
        help="Archive format (default: zip)",
    )
    parser.add_argument(
        "--output-dir",
        default=REPO_ROOT,
        help="Directory to save the archive (default: repo root)",
    )
    parser.add_argument(
        "--checkpoints-only",
        action="store_true",
        help="Only package checkpoints (exclude render images and logs)",
    )
    args = parser.parse_args()

    artifacts_dir = os.path.join(REPO_ROOT, "artifacts")
    if not os.path.exists(artifacts_dir):
        print(f"[ERROR] Artifacts directory not found at: {artifacts_dir}")
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_base = f"paint_rl_artifacts_{timestamp}"
    os.makedirs(args.output_dir, exist_ok=True)

    items_to_pack = []
    
    # Checkpoints
    checkpoints_dir = os.path.join(artifacts_dir, "checkpoints")
    if os.path.exists(checkpoints_dir):
        items_to_pack.append(("checkpoints", checkpoints_dir))

    if not args.checkpoints_only:
        # Renders
        renders_dir = os.path.join(artifacts_dir, "renders")
        if os.path.exists(renders_dir):
            items_to_pack.append(("renders", renders_dir))
        
        # Logs
        logs_dir = os.path.join(artifacts_dir, "logs")
        if os.path.exists(logs_dir):
            items_to_pack.append(("logs", logs_dir))
            
        # Dashboard
        dash_file = os.path.join(artifacts_dir, "dashboard.html")
        if os.path.exists(dash_file):
            items_to_pack.append(("dashboard.html", dash_file))

    if not items_to_pack:
        print("[WARN] No artifacts found to package.")
        sys.exit(0)

    print("=" * 60)
    print("   PAINT-CODE-RL: PACKAGING ARTIFACTS")
    print("=" * 60)
    for name, path in items_to_pack:
        if os.path.isdir(path):
            count = len(os.listdir(path))
            print(f"  + {name}/ ({count} files)")
        else:
            size_kb = os.path.getsize(path) / 1024
            print(f"  + {name} ({size_kb:.1f} KB)")
    print("=" * 60)

    if args.format == "zip":
        archive_path = os.path.join(args.output_dir, f"{archive_base}.zip")
        with zipfile.ZipFile(archive_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for name, path in items_to_pack:
                if os.path.isdir(path):
                    for root, _, files in os.walk(path):
                        for file in files:
                            full_path = os.path.join(root, file)
                            rel_path = os.path.join("artifacts", name, os.path.relpath(full_path, path))
                            zipf.write(full_path, rel_path)
                else:
                    zipf.write(path, os.path.join("artifacts", name))
    else:
        archive_path = os.path.join(args.output_dir, f"{archive_base}.tar.gz")
        with tarfile.open(archive_path, "w:gz") as tar:
            for name, path in items_to_pack:
                arcname = os.path.join("artifacts", name)
                tar.add(path, arcname=arcname)

    size_mb = os.path.getsize(archive_path) / (1024 * 1024)
    print(f"\n[OK] Package created successfully:")
    print(f"  File: {archive_path}")
    print(f"  Size: {size_mb:.2f} MB")
    print("\nHow to transfer:")
    print(f"  1. From remote Mac/Kaggle to your machine:")
    print(f"     scp user@remote-ip:{archive_path} ./")
    print(f"  2. To extract on your machine:")
    print(f"     unzip {os.path.basename(archive_path)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
