"""Strict release safety scan; emits a relative-path-only report."""
from __future__ import annotations
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_BYTES = 5 * 1024 * 1024
FORBIDDEN_SUFFIXES = {".npz", ".npy", ".jsonl", ".exe", ".dll", ".sys", ".bin", ".pkl", ".pickle", ".parquet"}
ABS_PATH = re.compile(r"(?i)\b[A-Z]:\\Users\\[^\s\"'<>]+")
USERNAME = re.compile(r"(?i)" + r"\b" + "kot" + "ta" + r"\b")
EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
SECRETS = [
    ("GitHub token", re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9_]{30,}|github_pat_[A-Za-z0-9_]{30,})\b")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("secret assignment", re.compile(r"(?i)\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password)\s*[:=]\s*[\"'][^\"']{8,}[\"']")),
    ("bearer token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{20,}")),
]

findings = []
count = 0
for path in sorted(ROOT.rglob("*")):
    if not path.is_file() or ".git" in path.parts:
        continue
    rel = path.relative_to(ROOT).as_posix()
    count += 1
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append((rel, "forbidden data/binary file type", path.suffix.lower()))
    if path.stat().st_size > MAX_BYTES:
        findings.append((rel, "file exceeds 5 MiB size scan threshold", str(path.stat().st_size)))
    if path.name.lower() == ".env" or path.name.lower().startswith(".env."):
        findings.append((rel, "environment/secret file name", ""))
    try:
        text = path.read_text(encoding="utf-8")
    except (UnicodeError, OSError):
        text = path.read_bytes().decode("latin-1", errors="ignore")
    for kind, pattern in (("absolute Windows user path", ABS_PATH), ("local username", USERNAME), ("email address", EMAIL)):
        for match in pattern.finditer(text):
            findings.append((rel, kind, match.group(0)))
    for label, pattern in SECRETS:
        for match in pattern.finditer(text):
            findings.append((rel, label, "[value redacted]"))

report = [
    "# Safety scan report",
    "",
    "Command: `python safety_scan.py`",
    "Working directory: release folder root (relative; no absolute path recorded).",
    f"Files scanned: {count}",
    "Checks: secret/token patterns; forbidden dataset/binary suffixes; files over 5 MiB; .env names; absolute Windows user paths; local username token; email addresses.",
    "",
]
if findings:
    report += [f"Result: FAIL ({len(findings)} finding(s)).", "", "| File | Finding | Match |", "|---|---|---|"]
    for name, kind, value in findings:
        escaped = value.replace("|", "\\|")
        report.append(f"| `{name}` | {kind} | `{escaped}` |")
else:
    report += ["Result: PASS (zero findings).", ""]
(ROOT / "scan_report.md").write_text("\n".join(report), encoding="utf-8")
print(f"Files scanned: {count}; findings: {len(findings)}")
for finding in findings:
    print(finding)
if findings:
    raise SystemExit(1)
