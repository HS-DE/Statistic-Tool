from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

DetectionRule = Literal["nonzero", "positive", "non_na"]


@dataclass(frozen=True)
class DetectionRuleInfo:
    key: DetectionRule
    label: str
    description: str


DETECTION_RULES: tuple[DetectionRuleInfo, ...] = (
    DetectionRuleInfo(
        "nonzero",
        "非 0、非 NA、非空（默认）",
        "数值可为正数或负数；0、NA、空值及无法解析为数值的内容视为未检出。",
    ),
    DetectionRuleInfo(
        "positive",
        "> 0",
        "仅严格大于 0 的数值视为检出。",
    ),
    DetectionRuleInfo(
        "non_na",
        "非 NA、非空",
        "只要存在可解析的数值即视为检出，0 也算检出。",
    ),
)


def coerce_numeric(values: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Convert an abundance object to numeric without mutating the input."""
    def clean_series(series: pd.Series) -> pd.Series:
        text = series.astype("string").str.strip()
        cleaned = series.mask(text.eq(""))
        return pd.to_numeric(cleaned, errors="coerce")

    if isinstance(values, pd.Series):
        return clean_series(values)
    return values.apply(clean_series)


def is_detected(
    values: pd.DataFrame | pd.Series,
    rule: DetectionRule = "nonzero",
) -> pd.DataFrame | pd.Series:
    """Return a boolean mask using the shared detection rule for the whole app."""
    numeric = coerce_numeric(values)
    finite = numeric.notna() & np.isfinite(numeric)

    if rule == "nonzero":
        return finite & numeric.ne(0)
    if rule == "positive":
        return finite & numeric.gt(0)
    if rule == "non_na":
        return finite
    raise ValueError(f"Unsupported detection rule: {rule}")
