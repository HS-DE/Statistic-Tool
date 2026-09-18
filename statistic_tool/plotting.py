from __future__ import annotations

from io import BytesIO
from typing import Mapping

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
import pandas as pd

from .overlap import membership_counts, region_counts_for_venn

matplotlib.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
matplotlib.rcParams["axes.unicode_minus"] = False

DEFAULT_COLORS = [
    "#5A8BF9",
    "#39D88C",
    "#FFD400",
    "#E5843C",
    "#9B72CF",
    "#41B6C4",
    "#E76F8A",
    "#7A8B99",
]


def color_map(keys: list[str], custom: Mapping[str, str] | None = None) -> dict[str, str]:
    custom = custom or {}
    return {key: custom.get(key, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) for i, key in enumerate(keys)}


def sample_bar_plot(df: pd.DataFrame, colors: Mapping[str, str] | None = None) -> plt.Figure:
    groups = df["group"].astype(str).drop_duplicates().tolist()
    cmap = color_map(groups, colors)
    fig, ax = plt.subplots(figsize=(max(7, len(df) * 0.45), 5.2))
    bars = ax.bar(df["sample_id"].astype(str), df["detected_n"], color=[cmap[g] for g in df["group"].astype(str)])
    ax.set_ylabel("Identified proteins (N)")
    ax.set_title("Protein identification counts by sample")
    ax.tick_params(axis="x", rotation=45)
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, padding=3, fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cmap[g]) for g in groups]
    if groups != ["ALL"]:
        ax.legend(handles, groups, title="Group", frameon=False)
    fig.tight_layout()
    return fig


def group_bar_plot(df: pd.DataFrame, colors: Mapping[str, str] | None = None) -> plt.Figure:
    groups = [g for g in df["group"].astype(str).tolist() if g != "Total"]
    cmap = color_map(groups, colors)
    bar_colors = ["black" if g == "Total" else cmap[g] for g in df["group"].astype(str)]
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.7), 5.2))
    bars = ax.bar(df["group"].astype(str), df["detected_n"], color=bar_colors)
    ax.set_ylabel("Identified proteins (N)")
    ax.set_title("Protein identification counts by group")
    ax.spines[["top", "right"]].set_visible(False)
    ax.bar_label(bars, padding=3, fontsize=9)
    fig.tight_layout()
    return fig


def cv_plot(cv_long: pd.DataFrame, colors: Mapping[str, str] | None = None, add_points: bool = False) -> plt.Figure:
    groups = cv_long["Group"].astype(str).drop_duplicates().tolist()
    cmap = color_map(groups, colors)
    fig, ax = plt.subplots(figsize=(max(6, len(groups) * 1.25), 5.5))

    arrays = [cv_long.loc[cv_long["Group"].astype(str).eq(g), "CV"].to_numpy() for g in groups]
    if arrays:
        parts = ax.violinplot(arrays, positions=np.arange(1, len(groups) + 1), showextrema=False, widths=0.85)
        for body, group in zip(parts["bodies"], groups, strict=True):
            body.set_facecolor(cmap[group])
            body.set_alpha(0.55)
            body.set_edgecolor("none")

        bp = ax.boxplot(arrays, positions=np.arange(1, len(groups) + 1), widths=0.18, patch_artist=True, showfliers=False)
        for patch, group in zip(bp["boxes"], groups, strict=True):
            patch.set_facecolor(cmap[group])
            patch.set_alpha(0.35)

        if add_points:
            rng = np.random.default_rng(0)
            for i, (group, values) in enumerate(zip(groups, arrays, strict=True), start=1):
                jitter = rng.normal(i, 0.055, size=len(values))
                ax.scatter(jitter, values, s=10, alpha=0.45, color=cmap[group], edgecolors="none")

        for i, values in enumerate(arrays, start=1):
            if len(values):
                median = float(np.median(values))
                ymax = float(np.max(values))
                ax.text(i, ymax, f"Median={median:.2f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(np.arange(1, len(groups) + 1), groups)
    ax.set_ylabel("Coefficient of Variation (%)")
    ax.set_title("CV by group")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def venn_plot(sets: Mapping[str, set[str]], colors: Mapping[str, str] | None = None) -> plt.Figure:
    names = list(sets)
    if len(names) not in (2, 3, 4):
        raise ValueError("Venn 图仅用于 2–4 个 list。")
    cmap = color_map(names, colors)
    counts = region_counts_for_venn(sets)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    if len(names) == 2:
        ellipses = [((0.38, 0.5), 0.62, 0.62, 0), ((0.62, 0.5), 0.62, 0.62, 0)]
        label_pos = [(0.18, 0.82), (0.82, 0.82)]
        count_pos = [(0.22, 0.5), (0.78, 0.5), (0.5, 0.5)]
    elif len(names) == 3:
        ellipses = [((0.50, 0.63), 0.58, 0.58, 0), ((0.63, 0.38), 0.58, 0.58, 0), ((0.37, 0.38), 0.58, 0.58, 0)]
        label_pos = [(0.50, 0.95), (0.88, 0.13), (0.12, 0.13)]
        count_pos = [(0.5, 0.78), (0.78, 0.31), (0.22, 0.31), (0.70, 0.55), (0.30, 0.55), (0.50, 0.23), (0.50, 0.47)]
    else:
        ellipses = [
            ((0.50, 0.60), 0.74, 0.48, -45),
            ((0.50, 0.60), 0.74, 0.48, 45),
            ((0.65, 0.40), 0.74, 0.48, 45),
            ((0.35, 0.40), 0.74, 0.48, -45),
        ]
        label_pos = [(0.20, 0.91), (0.80, 0.91), (0.10, 0.20), (0.90, 0.20)]
        count_pos = [
            (0.30, 0.83), (0.70, 0.83), (0.18, 0.30), (0.83, 0.30),
            (0.50, 0.77), (0.22, 0.68), (0.75, 0.40), (0.25, 0.40),
            (0.78, 0.68), (0.50, 0.18), (0.33, 0.58), (0.66, 0.58),
            (0.60, 0.32), (0.40, 0.32), (0.50, 0.45),
        ]

    for name, (center, width, height, angle) in zip(names, ellipses, strict=True):
        ax.add_patch(Ellipse(center, width, height, angle=angle, facecolor=cmap[name], edgecolor="none", alpha=0.38))
    for name, pos in zip(names, label_pos, strict=True):
        ax.text(*pos, name, ha="center", va="center", fontsize=12, fontweight="bold")
    for count, pos in zip(counts, count_pos, strict=True):
        ax.text(*pos, str(count), ha="center", va="center", fontsize=10)
    ax.set_title("Venn plot")
    fig.tight_layout()
    return fig


def upset_plot(sets: Mapping[str, set[str]], max_intersections: int = 30) -> plt.Figure:
    names = list(sets)
    counts = membership_counts(sets)
    rows = [(membership, n) for membership, n in counts.items() if n > 0]
    rows.sort(key=lambda item: item[1], reverse=True)
    rows = rows[:max_intersections]

    fig = plt.figure(figsize=(max(9, len(rows) * 0.48), 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.2, 1], hspace=0.05)
    ax_bar = fig.add_subplot(gs[0])
    ax_matrix = fig.add_subplot(gs[1], sharex=ax_bar)

    x = np.arange(len(rows))
    values = [n for _, n in rows]
    bars = ax_bar.bar(x, values)
    ax_bar.bar_label(bars, padding=2, fontsize=8)
    ax_bar.set_ylabel("Intersection size")
    ax_bar.spines[["top", "right"]].set_visible(False)
    ax_bar.tick_params(axis="x", bottom=False, labelbottom=False)

    for xi, (membership, _) in enumerate(rows):
        active = [i for i, flag in enumerate(membership) if flag]
        for yi, flag in enumerate(membership):
            ax_matrix.scatter(xi, yi, s=30, color="black" if flag else "#D9D9D9")
        if len(active) >= 2:
            ax_matrix.plot([xi, xi], [min(active), max(active)], color="black", linewidth=1.2)

    ax_matrix.set_yticks(np.arange(len(names)), names)
    ax_matrix.invert_yaxis()
    ax_matrix.set_xlabel("Intersections (largest first)")
    ax_matrix.spines[["top", "right", "bottom"]].set_visible(False)
    ax_matrix.tick_params(axis="x", bottom=False, labelbottom=False)
    fig.suptitle("UpSet plot")
    return fig


def figure_bytes(fig: plt.Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    buffer = BytesIO()
    fig.savefig(buffer, format=fmt, dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()
