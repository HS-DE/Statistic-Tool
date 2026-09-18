from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping

import pandas as pd


def clean_items(values: Iterable[object]) -> set[str]:
    cleaned: set[str] = set()
    for value in values:
        if pd.isna(value):
            continue
        item = str(value).strip()
        if item:
            cleaned.add(item)
    return cleaned


def normalize_sets(sets: Mapping[str, Iterable[object]]) -> dict[str, set[str]]:
    result: dict[str, set[str]] = {}
    for i, (raw_name, values) in enumerate(sets.items(), start=1):
        name = str(raw_name).strip() or f"list{i}"
        if name in result:
            raise ValueError(f"List 名称必须唯一：{name}")
        result[name] = clean_items(values)
    if len(result) < 2:
        raise ValueError("至少需要 2 个 list。")
    return result


def inclusive_intersections(sets: Mapping[str, set[str]]) -> dict[str, set[str]]:
    names = list(sets)
    out: dict[str, set[str]] = {}
    for size in range(2, len(names) + 1):
        for combo in combinations(names, size):
            values = set.intersection(*(sets[name] for name in combo))
            out[_intersection_name(combo)] = values
    return out


def exclusive_regions(sets: Mapping[str, set[str]]) -> dict[str, set[str]]:
    """Return every non-empty membership region, including single-list regions."""
    names = list(sets)
    universe = set().union(*sets.values()) if sets else set()
    out: dict[str, set[str]] = {}

    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            included = set.intersection(*(sets[name] for name in combo))
            excluded_names = [name for name in names if name not in combo]
            excluded = set().union(*(sets[name] for name in excluded_names)) if excluded_names else set()
            values = included - excluded
            out[_exclusive_name(combo)] = values

    if not universe:
        return out
    return out


def intersection_summary(sets: Mapping[str, set[str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for name, values in inclusive_intersections(sets).items():
        rows.append({"intersection": name, "type": "inclusive", "n": len(values)})
    for name, values in exclusive_regions(sets).items():
        rows.append({"intersection": name, "type": "exclusive", "n": len(values)})
    return pd.DataFrame(rows)


def membership_counts(sets: Mapping[str, set[str]]) -> dict[tuple[bool, ...], int]:
    names = list(sets)
    universe = sorted(set().union(*sets.values()) if sets else set())
    counts: dict[tuple[bool, ...], int] = {}
    for item in universe:
        membership = tuple(item in sets[name] for name in names)
        counts[membership] = counts.get(membership, 0) + 1
    return counts


def region_counts_for_venn(sets: Mapping[str, set[str]]) -> list[int]:
    """Return exclusive region counts in the plotting order for 2, 3, or 4 sets."""
    names = list(sets)
    if len(names) not in (2, 3, 4):
        raise ValueError("Venn 仅支持 2–4 个集合。")

    counts = membership_counts(sets)
    if len(names) == 2:
        order = [(1, 0), (0, 1), (1, 1)]
    elif len(names) == 3:
        order = [
            (1, 0, 0), (0, 1, 0), (0, 0, 1),
            (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1),
        ]
    else:
        order = [
            (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1),
            (1, 1, 0, 0), (1, 0, 1, 0), (1, 0, 0, 1),
            (0, 1, 1, 0), (0, 1, 0, 1), (0, 0, 1, 1),
            (1, 1, 1, 0), (1, 1, 0, 1), (1, 0, 1, 1), (0, 1, 1, 1),
            (1, 1, 1, 1),
        ]
    return [counts.get(tuple(bool(x) for x in pattern), 0) for pattern in order]


def sets_to_sheets(sets: Mapping[str, set[str]]) -> dict[str, pd.DataFrame]:
    sheets: dict[str, pd.DataFrame] = {}
    for name, values in sets.items():
        sheets[name] = pd.DataFrame({"ID": sorted(values)})
    return sheets


def _intersection_name(names: tuple[str, ...]) -> str:
    return f"intersect({', '.join(names)})"


def _exclusive_name(names: tuple[str, ...]) -> str:
    return f"exclusive({', '.join(names)})"
