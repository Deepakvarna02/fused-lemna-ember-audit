"""Dataset-only protocol helpers; never reads the official test file."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from ember.features import PEFeatureExtractor


def canonical_layout() -> dict:
    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    offset = 0
    groups = {}
    for feature in extractor.features:
        groups[feature.name] = {"offset": offset, "width": feature.dim}
        offset += feature.dim
    if offset != 2381:
        raise RuntimeError(f"Unexpected EMBER v2 dimension: {offset}")
    return {"dimension": offset, "groups": groups}


def ember_fusion_edges(layout: dict | None = None) -> tuple[tuple[int, int], ...]:
    """Return the predeclared EMBER adaptation of LEMNA's fusion adjacency.

    Only byte histogram bins, byte-entropy bins, and printable-character
    distribution bins have an intrinsic ordered axis. All other EMBER groups
    are excluded because their flattened adjacency is hash, categorical,
    scalar-summary, variable-section, import/export, or directory structure
    rather than a meaningful sequential neighborhood.
    """
    layout = layout or canonical_layout()
    edges: list[tuple[int, int]] = []
    for group in ("histogram", "byteentropy"):
        start = layout["groups"][group]["offset"]
        width = layout["groups"][group]["width"]
        edges.extend((index, index + 1) for index in range(start, start + width - 1))
    strings = layout["groups"]["strings"]
    start = strings["offset"]
    # StringExtractor's first scalar fields precede printabledist (96 bins).
    printable_start = start + strings["width"] - 96
    edges.extend(
        (index, index + 1)
        for index in range(printable_start, printable_start + 95)
    )
    return tuple(edges)


def training_record_layout(root: Path, limit: int = 1) -> dict:
    layout = canonical_layout()
    extractor = PEFeatureExtractor(feature_version=2, print_feature_warning=False)
    count = 0
    with (root / "train_features_1.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record["label"] not in (0, 1):
                continue
            vector = extractor.process_raw_features(record)
            if vector.shape != (2381,):
                raise RuntimeError(f"Unexpected vector shape: {vector.shape}")
            count += 1
            if count == limit:
                break
    if count != limit:
        raise RuntimeError("Training layout check found too few records")
    return layout


def stratified_indices(labels: np.ndarray, seed: int = 20261106) -> dict[str, np.ndarray]:
    labels = np.asarray(labels)
    if not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Only supervised labels 0 and 1 may be split")
    rng = np.random.default_rng(seed)
    by_label = {label: rng.permutation(np.flatnonzero(labels == label)) for label in (0, 1)}
    splits = {"train": [], "validation": [], "reserved": []}
    for label in (0, 1):
        values = by_label[label]
        n_train = int(round(0.70 * len(values)))
        n_validation = int(round(0.15 * len(values)))
        splits["train"].extend(values[:n_train])
        splits["validation"].extend(values[n_train:n_train + n_validation])
        splits["reserved"].extend(values[n_train + n_validation:])
    return {name: np.sort(values) for name, values in splits.items()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
