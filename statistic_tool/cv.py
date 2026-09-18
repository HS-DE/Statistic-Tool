from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np
import pandas as pd

from .detection import DetectionRule, coerce_numeric, is_detected


@dataclass
class CVResult:
    long: pd.DataFrame
    wide: pd.DataFrame
    group_status: pd.DataFrame


def calculate_cv_by_group(
    abundance: pd.DataFrame,
    feature_ids: pd.Series,
    sample_group_map: pd.DataFrame,
    rule: DetectionRule = "nonzero",
    min_group_size: int = 3,
    min_detect_rate: float = 0.70,
    min_mean: float = 1e-8,
) -> CVResult:
    if min_group_size < 2:
        raise ValueError("每组最少样本数不能小于 2。")
    if not 0 < min_detect_rate <= 1:
        raise ValueError("min_detect_rate 必须在 (0, 1]。")

    numeric = coerce_numeric(abundance)
    mask = is_detected(abundance, rule=rule)
    ids = feature_ids.astype(str).reset_index(drop=True)

    long_parts: list[pd.DataFrame] = []
    wide: dict[str, pd.Series] = {}
    status_rows: list[dict[str, object]] = []

    for group in sample_group_map["group"].drop_duplicates().astype(str):
        samples = sample_group_map.loc[sample_group_map["group"].astype(str).eq(group), "sample_id"].tolist()
        samples = [s for s in samples if s in abundance.columns]
        n_group = len(samples)

        if n_group < min_group_size:
            status_rows.append(
                {
                    "group": group,
                    "n_samples": n_group,
                    "status": "skipped",
                    "reason": f"样本数 < {min_group_size}",
                    "proteins_with_cv": 0,
                }
            )
            continue

        required = max(min_group_size, ceil(n_group * min_detect_rate))
        values = numeric[samples].to_numpy(dtype=float)
        detected = mask[samples].to_numpy(dtype=bool)

        detected_n = detected.sum(axis=1)
        masked_values = np.where(detected, values, np.nan)
        masked_df = pd.DataFrame(masked_values)
        means = masked_df.mean(axis=1, skipna=True).to_numpy(dtype=float)
        sds = masked_df.std(axis=1, skipna=True, ddof=1).to_numpy(dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            cvs = (sds / means) * 100.0

        invalid = (
            (detected_n < required)
            | ~np.isfinite(means)
            | ~np.isfinite(sds)
            | (np.abs(means) < min_mean)
            | ~np.isfinite(cvs)
        )
        cvs[invalid] = np.nan

        series = pd.Series(cvs, index=ids, name=group)
        wide[group] = series
        part = pd.DataFrame({"Protein": ids, "Group": group, "CV": cvs}).dropna(subset=["CV"])
        long_parts.append(part)
        status_rows.append(
            {
                "group": group,
                "n_samples": n_group,
                "status": "calculated",
                "reason": "",
                "proteins_with_cv": int(part.shape[0]),
            }
        )

    long_df = pd.concat(long_parts, ignore_index=True) if long_parts else pd.DataFrame(columns=["Protein", "Group", "CV"])
    wide_df = pd.DataFrame(wide)
    if not wide_df.empty:
        wide_df.index.name = "Protein"
        wide_df = wide_df.reset_index()
    else:
        wide_df = pd.DataFrame({"Protein": ids})

    return CVResult(
        long=long_df,
        wide=wide_df,
        group_status=pd.DataFrame(status_rows),
    )
