"""Create the available, sanitized local-release staging files."""
from __future__ import annotations
import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EXP = ROOT / "experiments"
RUN_A = EXP / "FINAL_VALIDATION" / "20261108_073205"
RUN_B = EXP / "FINAL_VALIDATION_B" / "20261108_073544"
RAW = EXP / "results" / "raw"

for rel in ("results/run_a", "results/run_b", "reproduction/experiments/scripts",
            "reproduction/experiments/src", "reproduction/experiments/results/raw/failed_subproblems",
            "reproduction/experiments/FINAL_VALIDATION/20261108_073205",
            "reproduction/experiments/FINAL_VALIDATION_B/20261108_073544"):
    (HERE / rel).mkdir(parents=True, exist_ok=True)

def copy(src: Path, rel: str):
    dst = HERE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

# Run A bounded split-gap audit.
for filename in ("split_gap_bound_audit.json", "split_gap_bound_audit.csv"):
    copy(RUN_A / filename, f"results/run_a/{filename}")

# Compact the prior 16-input, three-method validation without its iteration histories.
source = json.loads((RAW / "LEMNA_RANK_AWARE_SOLVER_VALIDATION.json").read_text(encoding="utf-8"))
rows = []
for sub in source["subproblems"]:
    for method, record in sub["methods"].items():
        final = record["final"]
        primal_ok = final["primal_residual"] <= final["primal_tolerance"]
        dual_ok = final["dual_residual"] <= final["dual_tolerance"]
        rows.append({
            "file": sub["file"], "record_index": sub["record_index"], "method": method,
            "outer_iterations": record["iterations"], "residual_certificate": bool(primal_ok and dual_ok),
            "primal_residual": final["primal_residual"], "dual_residual": final["dual_residual"],
            "primal_tolerance": final["primal_tolerance"], "dual_tolerance": final["dual_tolerance"],
            "objective": final["weighted_regression_objective"], "raw_beta_tv": record["raw_tv"],
            "excess_tv": record["raw_tv"] - 1.0,
            "raw_tv_le_S_exact": bool(record["raw_tv"] <= 1.0),
            "residual_and_tv": bool(primal_ok and dual_ok and record["raw_tv"] <= 1.0),
            "runtime_seconds": record["runtime_seconds"],
            "final_inner_stop_code": record["beta_step"].get("istop"),
            "final_inner_iterations": record["beta_step"].get("iterations"),
            "inner_solver_all_steps_converged": record["beta_step_converged_all"],
        })
summary = {
    "source_artifact": "LEMNA_RANK_AWARE_SOLVER_VALIDATION.json",
    "training_only": source["training_only"], "final_test_accessed": source["final_test_accessed"],
    "configuration": source["configuration"], "fusion_graph": source["fusion_graph"],
    "input_count": source["verified_input_count"],
    "method_counts": {}, "entries": rows,
}
for method in ("cholesky_jitter", "lsmr", "lsqr"):
    values = [r for r in rows if r["method"] == method]
    summary["method_counts"][method] = {
        "entries": len(values),
        "residual_certificate_count": sum(r["residual_certificate"] for r in values),
        "raw_tv_le_S_exact_count": sum(r["raw_tv_le_S_exact"] for r in values),
        "combined_count": sum(r["residual_and_tv"] for r in values),
        "inner_final_stop_code_7_count": sum(r["final_inner_stop_code"] == 7 for r in values),
        "runtime_median_seconds": statistics.median(r["runtime_seconds"] for r in values),
        "runtime_min_seconds": min(r["runtime_seconds"] for r in values),
        "runtime_max_seconds": max(r["runtime_seconds"] for r in values),
    }
(HERE / "results/run_a/solver_validation_aggregate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
with (HERE / "results/run_a/solver_validation_aggregate.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)

# Run B artifacts (no raw inputs or sample matrices).
for stem in ("test4_larger_neighbourhood", "test5_ablation", "test6_sensitivity"):
    for ext in ("json", "csv"):
        copy(RUN_B / f"{stem}.{ext}", f"results/run_b/{stem}.{ext}")
copy(RUN_B / "run_metadata.json", "results/run_b/run_metadata.json")

# Hash-only manifest: preserve the identifiers and hashes, never the arrays or absolute source paths.
input_manifest = json.loads((RAW / "LEMNA_FAILED_INPUTS_MANIFEST.json").read_text(encoding="utf-8"))
hash_entries = []
for item in input_manifest["files"]:
    p = Path(item)
    hash_entries.append({"file": p.name, "sha256": sha(p)})
safe_manifest = {"count": len(hash_entries), "training_only": True, "final_test_accessed": False,
                 "files": hash_entries, "note": "Hashes and basenames only; NPZ data files are intentionally excluded."}
(HERE / "results/run_a/failed_inputs_hash_manifest.json").write_text(json.dumps(safe_manifest, indent=2), encoding="utf-8")

# Pinned Python dependencies and reproducibility source files.
copy(EXP / "requirements-lock.txt", "reproduction/requirements-lock.txt")
for name in ("run_pilot2.py", "rank_deficiency_diagnostic.py", "rank_aware_solver_validation.py"):
    copy(EXP / "scripts" / name, f"reproduction/experiments/scripts/{name}")
for name in ("__init__.py", "lemna.py", "protocol.py"):
    copy(EXP / "src" / name, f"reproduction/experiments/src/{name}")
synthetic_source = EXP / "scripts" / "synthetic"
if synthetic_source.exists():
    shutil.copytree(synthetic_source, HERE / "reproduction/experiments/scripts/synthetic", dirs_exist_ok=True)
for name in ("run_b.py", "run_b_test4_interrupted.py", "run_b_test5_interrupted.py",
             "finalize_run_b.py", "record_test6_interruption.py"):
    copy(RUN_B / name, f"reproduction/experiments/FINAL_VALIDATION_B/20261108_073544/{name}")
copy(RUN_A / "bound_audit.py", "reproduction/experiments/FINAL_VALIDATION/20261108_073205/bound_audit.py")
for name in ("test4_larger_neighbourhood.json", "test4_larger_neighbourhood.csv",
             "test5_ablation.json", "test5_ablation.csv", "test6_sensitivity.json",
             "test6_sensitivity.csv", "run_metadata.json"):
    copy(RUN_B / name, f"reproduction/experiments/FINAL_VALIDATION_B/20261108_073544/{name}")
copy(RUN_A / "split_gap_bound_audit.json", "reproduction/experiments/FINAL_VALIDATION/20261108_073205/split_gap_bound_audit.json")
copy(RUN_A / "split_gap_bound_audit.csv", "reproduction/experiments/FINAL_VALIDATION/20261108_073205/split_gap_bound_audit.csv")

# Sanitized, relocatable manifest consumed by the validation script after inputs are regenerated locally.
rel_manifest = {"files": [f"experiments/results/raw/failed_subproblems/{e['file']}" for e in hash_entries],
                "count": len(hash_entries), "training_only": True, "final_test_accessed": False}
(HERE / "reproduction/experiments/results/raw/LEMNA_FAILED_INPUTS_MANIFEST.json").write_text(json.dumps(rel_manifest, indent=2), encoding="utf-8")

metadata = json.loads((RUN_B / "run_metadata.json").read_text(encoding="utf-8"))
(HERE / "environment.json").write_text(json.dumps({
    "run_a_environment": {"python": "3.12.4", "numpy": "2.2.6", "scipy": "1.15.3",
                         "pinned_requirements_file": "reproduction/requirements-lock.txt"},
    "run_b_environment": metadata,
}, indent=2), encoding="utf-8")

# Citation audit is intentionally blocked until the user supplies FINAL_PAPER/*.bib.
(HERE / "reference_audit.md").write_text("""# Reference audit — blocked pending paper files

Audit date: 2026-11-08.

**Status: BLOCKED.** `experiments/FINAL_PAPER/` was absent when this release preparation began. No `.bib` file was available, so the complete bibliography could not be enumerated or checked. The user-specified cross-check list is not assumed to be the complete bibliography. No citation details are inferred here. Once the `.bib` file is supplied, verify every entry against DOI/Crossref, publisher, or arXiv records, then replace this note with the verified metadata, source links, date checked, and discrepancies.
""", encoding="utf-8")

(HERE / ".gitignore").write_text("""# Exclude EMBER and all sample-level data
data/
datasets/
train_features_*.jsonl
test_features*.jsonl
*.npz
*.npy
*.pkl
*.pickle
*.parquet
*.bin
*.exe
*.dll
*.sys
*.malware
raw/
results/raw/

# Credentials, logs, and local runtime artifacts
.env
.env.*
*.log
logs/
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
""", encoding="utf-8")

(HERE / ".zenodo.json").write_text(json.dumps({
    "title": "EMBER Fused-LEMNA: Bounded Validation Release",
    "upload_type": "software",
    "description": "Local reproducibility materials and aggregate training-only validation results for a bounded EMBER Fused-LEMNA study. Dataset files, malware binaries, sample-level NPZ inputs, and credentials are not included.",
    "creators": [{"name": "Kottapalli Deepak Varma", "affiliation": "R.M.D. Engineering College", "orcid": "AUTHOR TO CONFIRM"}],
    "keywords": ["EMBER", "LEMNA", "fused lasso", "explainable AI", "malware classification", "reproducibility"],
    "version": "0.1.0",
    "license": "AUTHOR TO CONFIRM",
    "related_identifiers": [{"identifier": "AUTHOR TO CONFIRM", "relation": "isSupplementTo", "scheme": "url"}],
}, indent=2), encoding="utf-8")
(HERE / "arxiv_checklist.md").write_text("""# arXiv local preparation checklist

Proposed primary category: `cs.CR`; secondary category: `cs.LG`.

## Files to prepare

- Final `.tex` source and `.bib` bibliography.
- All figures and required `.sty` / `.cls` files that are not part of standard TeX distributions.
- Compiled final PDF and source ZIP containing the TeX sources, bibliography, figures, and required styles.
- A clean local safety-scan report and verified page-render review.

## Author confirmation

- Confirm final title, author order/spelling, affiliation, contact address, funding, and conflict-of-interest statements; do not infer missing statements.
- Confirm category selection and all bibliography metadata.
- The author must start the arXiv submission flow to see whether an endorsement is required for `cs.CR` or `cs.LG`. This package does not submit or request endorsement.
- Verify license, ORCID, and related GitHub URL before any later deposit.

No submission was made.
""", encoding="utf-8")
(HERE / "endorsement_request_email.txt").write_text("""Subject: Request for advice on arXiv endorsement

Dear Professor [Name],

I am preparing a manuscript in computer security and plan to submit it to arXiv under cs.CR, with cs.LG as a secondary category. I will begin the submission flow to confirm whether endorsement is required. If it is, would you be willing to advise me on the appropriate process?

I can share the manuscript and a short summary for your review. There is no urgency; please feel free to decline.

Best regards,
Kottapalli Deepak Varma
""", encoding="utf-8")
(HERE / "faculty_review_request_email.txt").write_text("""Subject: Request for a brief manuscript review

Dear Professor [Name],

Would you be available to review a draft manuscript on a training-only, bounded validation study of Fused-LEMNA using EMBER features? I would especially value feedback on the experimental scope, numerical reporting, and citation accuracy.

I can send the final draft and supporting aggregate results if you are available. Please feel free to decline if your schedule does not permit it.

Best regards,
Kottapalli Deepak Varma
""", encoding="utf-8")

readme = """# Local release staging — EMBER Fused-LEMNA

This folder contains curated reproduction scripts, aggregate validation results, and local metadata templates. It excludes EMBER records, PE/malware binaries, raw neighborhood matrices, NPZ inputs, secrets, local absolute paths, and private email addresses. **The NPZ inputs are not included.**

## Current completion state

- Run A bound audit and compact solver aggregates are present.
- Run B result records and runtime metadata are present, including stopped branches and NOT RUN statuses.
- `experiments/FINAL_PAPER/` was not present at staging time. The complete `.bib` audit, paper source/PDF/figures copy, source ZIP, TeX compilation, PDF rendering, and page-count review remain blocked until those files are supplied.
- `synth.py` was not found in the project file inventory. It is not fabricated or substituted.
- Local Git initialization and commit are deferred until the release contents are complete and the final safety scan passes.

## Reproduction setup

Obtain EMBER through the official [Elastic EMBER repository](https://github.com/elastic/ember) and follow the project’s download and usage terms. The official project documents the 2018 feature archive there and cites the [EMBER arXiv paper](https://arxiv.org/abs/1804.04637). Place only the training feature JSONL files in `reproduction/`; do not place the test JSONL file in the reproduction tree. The release bundle does not redistribute the dataset.

From `reproduction/`, after installing the pinned packages with `python -m pip install -r requirements-lock.txt` and installing the EMBER feature extractor as documented by its official project:

```powershell
python experiments/scripts/rank_deficiency_diagnostic.py
python experiments/scripts/rank_aware_solver_validation.py
```

The first command reconstructs the two record-specific inputs and writes local NPZ intermediates to `experiments/results/raw/failed_subproblems/`. The second command validates the persisted inputs. These local NPZ intermediates are intentionally absent from this release. Run B scripts and the Run A bound audit are retained under their original relative paths in the reproduction tree; they require locally regenerated matching inputs and can be computationally expensive, especially dense SVD branches.

## Environment

Pinned Python packages are listed in `reproduction/requirements-lock.txt`. Captured runtime environment details are in `environment.json`. The EMBER Python feature extractor is not included in the pinned lock file; install it from the official project instructions and record its version locally before attempting reproduction.

## Limitations

- All bundled results are training-only and do not include the official EMBER test records.
- This release has aggregate results and hashes, but omits the 16 NPZ inputs and all sample-level arrays.
- Run B Tests 4 and 5 were stopped during full-spectrum SVD; Test 6 includes only reused cap-200 baselines, with higher caps and the looser setting not run. See `results/run_b/`.
- The reference audit and paper archive are pending the final `.bib`, TeX sources, figures, and PDF.
- `synth.py` was not located and is not included.

## License and publication metadata

The `.zenodo.json` template leaves license, ORCID, and related GitHub URL for author confirmation. No GitHub repository, Zenodo record, or arXiv submission has been created.
"""
(HERE / "README.md").write_text(readme, encoding="utf-8")

# Exact source names and hashes for the inputs, plus inventory note; arrays remain excluded.
(HERE / "stored_hashes_manifest.json").write_text(json.dumps(safe_manifest, indent=2), encoding="utf-8")
print(json.dumps({"release_folder": str(HERE), "aggregated_entries": len(rows),
                  "stored_npz_count_not_copied": len(hash_entries),
                  "paper_folder_present": (EXP / "FINAL_PAPER").exists()}, indent=2))
