from __future__ import annotations

import pandas as pd

from .detection import DetectionRule, is_detected


def sample_identification_counts(
    abundance: pd.DataFrame,
    rule: DetectionRule = "nonzero",
    sample_group_map: pd.DataFrame | None = None,
) -> pd.DataFrame:
    mask = is_detected(abundance, rule=rule)
    out = pd.DataFrame(
        {
            "sample_id": abundance.columns.astype(str),
            "detected_n": mask.sum(axis=0).astype(int).to_numpy(),
        }
    )
    if sample_group_map is None:
        out["group"] = "ALL"
    else:
        out = out.merge(sample_group_map[["sample_id", "group"]], on="sample_id", how="left")
        out["group"] = out["group"].fillna("ALL")
    return out


def group_identification_counts(
    abundance: pd.DataFrame,
    sample_group_map: pd.DataFrame,
    rule: DetectionRule = "nonzero",
    include_total: bool = True,
) -> pd.DataFrame:
    mask = is_detected(abundance, rule=rule)
    rows: list[dict[str, object]] = []

    for group in sample_group_map["group"].drop_duplicates().astype(str):
        samples = sample_group_map.loc[sample_group_map["group"].astype(str).eq(group), "sample_id"].tolist()
        samples = [s for s in samples if s in mask.columns]
        if not samples:
            continue
        rows.append({"group": group, "detected_n": int(mask[samples].any(axis=1).sum())})

    if include_total:
        rows.append({"group": "Total", "detected_n": int(mask.any(axis=1).sum())})

    return pd.DataFrame(rows)


def detected_sets_by_group(
    abundance: pd.DataFrame,
    feature_ids: pd.Series,
    sample_group_map: pd.DataFrame,
    rule: DetectionRule = "nonzero",
) -> dict[str, set[str]]:
    mask = is_detected(abundance, rule=rule)
    ids = feature_ids.astype(str).tolist()
    sets: dict[str, set[str]] = {}

    for group in sample_group_map["group"].drop_duplicates().astype(str):
        samples = sample_group_map.loc[sample_group_map["group"].astype(str).eq(group), "sample_id"].tolist()
        samples = [s for s in samples if s in mask.columns]
        if not samples:
            continue
        detected = mask[samples].any(axis=1).to_numpy()
        sets[group] = {feature for feature, keep in zip(ids, detected, strict=True) if keep}
    return sets
