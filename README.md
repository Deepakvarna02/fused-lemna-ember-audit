# Local release staging — EMBER Fused-LEMNA

This folder contains curated reproduction scripts, aggregate validation results, and local metadata templates. It excludes EMBER records, PE/malware binaries, raw neighborhood matrices, NPZ inputs, secrets, local absolute paths, and private email addresses. **The NPZ inputs are not included.**

## Current completion state

- Run A bound audit and compact solver aggregates are present.
- Run B result records and runtime metadata are present, including stopped branches and NOT RUN statuses.
- `experiments/FINAL_PAPER/` was not present at staging time. The complete `.bib` audit, paper source/PDF/figures copy, source ZIP, TeX compilation, PDF rendering, and page-count review remain blocked until those files are supplied.
- `synth.py` was not found in the project file inventory. It is not fabricated or substituted.
- This repository contains the curated Run A and Run B code and aggregate results. It does not include the paper or raw sample inputs.

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

The `.zenodo.json` template leaves license, ORCID, and related GitHub URL for author confirmation. The source repository is [fused-lemna-ember-audit](https://github.com/Deepakvarna02/fused-lemna-ember-audit). No Zenodo record or arXiv submission has been created. This code publication does not indicate that the paper is ready for submission.
