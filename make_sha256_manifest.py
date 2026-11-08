"""Write relative SHA-256 entries for every release file except this manifest."""
from __future__ import annotations
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
target = ROOT / "sha256_manifest.txt"
entries = []
for path in sorted(p for p in ROOT.rglob("*") if p.is_file() and p != target and ".git" not in p.parts):
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    entries.append(f"{digest}  {path.relative_to(ROOT).as_posix()}")
target.write_text(
    "# SHA-256 manifest. Self-excluded because a file cannot contain its own stable hash.\n"
    + "\n".join(entries) + "\n", encoding="utf-8"
)
print(f"Hashed {len(entries)} release files (manifest excludes itself).")
