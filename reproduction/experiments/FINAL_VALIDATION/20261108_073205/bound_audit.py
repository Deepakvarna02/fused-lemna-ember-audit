"""Compute split-gap TV bounds from the 16 persisted failed-subproblem NPZs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "experiments" / "results" / "raw"
MANIFEST = RAW / "LEMNA_FAILED_INPUTS_MANIFEST.json"
REPORT_JSON = HERE / "split_gap_bound_audit.json"
REPORT_CSV = HERE / "split_gap_bound_audit.csv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    paths = [Path(item) for item in manifest["files"]]
    if len(paths) != 16:
        raise RuntimeError(f"Expected 16 inputs, found {len(paths)}")
    rows = []
    for path in paths:
        with np.load(path, allow_pickle=False) as data:
            required = {"difference", "beta", "z"}
            missing = required - set(data.files)
            if missing:
                raise RuntimeError(f"{path} is missing {sorted(missing)}")
            D = np.asarray(data["difference"], dtype=np.float64)
            beta = np.asarray(data["beta"], dtype=np.float64)
            z = np.asarray(data["z"], dtype=np.float64)
        if D.shape != (605, 2381) or beta.shape != (2381,) or z.shape != (605,):
            raise RuntimeError(f"Unexpected dimensions in {path.name}: {D.shape}, {beta.shape}, {z.shape}")
        d_beta = D @ beta
        split_gap = float(np.linalg.norm(d_beta - z, ord=2))
        bound = float(math.sqrt(605) * split_gap)
        raw_beta_tv = float(np.linalg.norm(d_beta, ord=1))
        z_tv = float(np.linalg.norm(z, ord=1))
        excess = raw_beta_tv - 1.0
        rows.append({
            "file": path.name,
            "sha256_file": sha256(path),
            "z_tv": z_tv,
            "raw_beta_tv": raw_beta_tv,
            "S": 1.0,
            "raw_beta_tv_minus_S": excess,
            "split_gap_l2": split_gap,
            "bound_sqrt605_times_gap": bound,
            "check_excess_le_bound": bool(excess <= bound),
            "bound_minus_excess": bound - excess,
        })
    result = {
        "description": "Per-input split-gap TV bound audit on persisted beta,z arrays; S=1.0.",
        "formula": "sqrt(605) * ||D*beta - z||_2",
        "check": "(||D*beta||_1 - S) <= bound",
        "input_count": len(rows),
        "satisfying_count": sum(row["check_excess_le_bound"] for row in rows),
        "failed_count": sum(not row["check_excess_le_bound"] for row in rows),
        "all_checks_satisfied": all(row["check_excess_le_bound"] for row in rows),
        "inputs": rows,
    }
    REPORT_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    with REPORT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({k: result[k] for k in ("input_count", "satisfying_count", "failed_count", "all_checks_satisfied")}, indent=2))
    for row in rows:
        print(f"{row['file']}: excess={row['raw_beta_tv_minus_S']:.17g}, bound={row['bound_sqrt605_times_gap']:.17g}, check={row['check_excess_le_bound']}")


if __name__ == "__main__":
    main()
