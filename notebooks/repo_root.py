import sys
from pathlib import Path

def find_repo_root(marker=".git"):
    path = Path.cwd()
    for parent in [path] + list(path.parents):
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find repo root (no {marker} found)")

repo_root = find_repo_root()
sys.path.append(str(repo_root))