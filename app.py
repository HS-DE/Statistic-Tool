from __future__ import annotations

import re
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from statistic_tool.cv import calculate_cv_by_group
from statistic_tool.detection import DETECTION_RULES, coerce_numeric
from statistic_tool.identification import (
    detected_sets_by_group,
    group_identification_counts,
    sample_identification_counts,
)
from statistic_tool.io import (
    build_sample_group_map,
    ensure_unique_feature_ids,
    read_table,
    to_excel_bytes,
)
from statistic_tool.overlap import (
    exclusive_regions,
    inclusive_intersections,
    intersection_summary,
    normalize_sets,
    sets_to_sheets,
)
from statistic_tool.plotting import (
    DEFAULT_COLORS,
    cv_plot,
    figure_bytes,
    group_bar_plot,
    sample_bar_plot,
    upset_plot,
    venn_plot,
)

st.set_page_config(page_title="Statistic Tool", page_icon="🧬", layout="wide")


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def numeric_candidate_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    candidates: list[str] = []
    for col in df.columns:
        if str(col) in exclude:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().mean() >= 0.5:
            candidates.append(str(col))
    return candidates


def read_list_upload(uploaded) -> list[str]:
    if uploaded is None:
        return []
    suffix = Path(uploaded.name).suffix.lower()
    uploaded.seek(0)
    if suffix == ".txt":
        text = uploaded.getvalue().decode("utf-8-sig", errors="replace")
        return [x.strip() for x in re.split(r"[\n\r\t,]+", text) if x.strip()]
    df = read_table(uploaded)
    if df.empty or df.shape[1] == 0:
        return []
    return df.iloc[:, 0].dropna().astype(str).tolist()


def render_figure_downloads(fig, prefix: str) -> None:
    cols = st.columns(3)
    for col, fmt, mime in zip(
        cols,
        ("png", "pdf", "svg"),
        ("image/png", "application/pdf", "image/svg+xml"),
        strict=True,
    ):
        with col:
            st.download_button(
                f"下载 {fmt.upper()}",
                data=figure_bytes(fig, fmt=fmt),
                file_name=f"{prefix}.{fmt}",
                mime=mime,
                use_container_width=True,
            )


def group_color_controls(groups: list[str]) -> dict[str, str]:
    colors: dict[str, str] = {}
    for i, group in enumerate(groups):
        key = f"group_color::{group}"
        default = st.session_state.get(key, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        colors[group] = st.sidebar.color_picker(str(group), default, key=key)
    return colors


st.title("🧬 Statistic Tool")
st.caption("Proteomics identification · overlap · CV")

with st.sidebar:
    st.header("数据 / Data")
    abundance_file = st.file_uploader(
        "丰度矩阵",
        type=["csv", "tsv", "txt", "xlsx"],
        help="支持 CSV、TSV、TXT、XLSX。",
    )
    metadata_file = st.file_uploader(
        "样本信息 metadata（可选）",
        type=["csv", "tsv", "txt", "xlsx"],
        help="样本鉴定量可以不提供 metadata；组鉴定量和 CV 需要 metadata。",
    )

abundance_df: pd.DataFrame | None = None
metadata_df: pd.DataFrame | None = None
feature_col: str | None = None
sample_group_map: pd.DataFrame | None = None
sample_columns: list[str] = []
rule = "nonzero"
group_colors: dict[str, str] = {}

if abundance_file is None:
    st.info("先在左侧上传丰度矩阵。上传后可以自己指定 Protein/Gene ID、sample_id 和 group 列。")
    st.stop()

try:
    abundance_df = read_table(abundance_file)
except Exception as exc:
    st.error(f"丰度矩阵读取失败：{exc}")
    st.stop()

if abundance_df.empty:
    st.error("丰度矩阵为空。")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("字段映射")
    feature_col = st.selectbox("Protein / Gene ID 列", [str(c) for c in abundance_df.columns])

try:
    ensure_unique_feature_ids(abundance_df, feature_col)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

if metadata_file is not None:
    try:
        metadata_df = read_table(metadata_file)
    except Exception as exc:
        st.error(f"metadata 读取失败：{exc}")
        st.stop()

    with st.sidebar:
        metadata_cols = [str(c) for c in metadata_df.columns]
        sample_id_col = st.selectbox("metadata: sample_id 列", metadata_cols)
        default_group_idx = metadata_cols.index("group") if "group" in metadata_cols else 0
        group_col = st.selectbox("metadata: group 列", metadata_cols, index=default_group_idx)

    try:
        sample_group_map, matrix_only, metadata_only = build_sample_group_map(
            metadata_df,
            sample_id_col,
            group_col,
            [str(c) for c in abundance_df.columns if str(c) != feature_col],
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    sample_columns = sample_group_map["sample_id"].astype(str).tolist()
    with st.sidebar:
        st.success(f"匹配样本：{len(sample_columns)}")
        if metadata_only:
            st.warning(f"metadata 中有 {len(metadata_only)} 个样本未匹配矩阵。")
        if matrix_only:
            st.caption(f"矩阵中另有 {len(matrix_only)} 列未被 metadata 匹配；不会自动作为样本使用。")
else:
    candidates = numeric_candidate_columns(abundance_df, {feature_col})
    with st.sidebar:
        sample_columns = st.multiselect(
            "样本列",
            [str(c) for c in abundance_df.columns if str(c) != feature_col],
            default=candidates,
            help="没有 metadata 时请确认这里仅选择真正的样本丰度列。",
        )

if not sample_columns:
    st.error("没有可用样本列。")
    st.stop()

with st.sidebar:
    st.divider()
    st.subheader("检出规则")
    rule_labels = {info.label: info for info in DETECTION_RULES}
    selected_rule_label = st.selectbox("Detection rule", list(rule_labels), index=0)
    rule_info = rule_labels[selected_rule_label]
    rule = rule_info.key
    st.caption(rule_info.description)

    st.divider()
    st.subheader("颜色")
    if sample_group_map is not None:
        groups = sample_group_map["group"].astype(str).drop_duplicates().tolist()
        group_colors = group_color_controls(groups)
    else:
        all_color = st.color_picker("全部样本", "#5A8BF9", key="all_sample_color")
        group_colors = {"ALL": all_color}

feature_ids = abundance_df[feature_col].astype(str).str.strip().reset_index(drop=True)
abundance = abundance_df[sample_columns].copy()
abundance.columns = [str(c) for c in abundance.columns]

with st.expander("数据检查", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Features", len(feature_ids))
    c2.metric("Samples", len(sample_columns))
    c3.metric("Groups", sample_group_map["group"].nunique() if sample_group_map is not None else 1)
    st.dataframe(abundance_df.head(20), use_container_width=True, hide_index=True)
    if sample_group_map is not None:
        st.markdown("**实际参与分析的 sample → group 映射**")
        st.dataframe(sample_group_map, use_container_width=True, hide_index=True)

sample_tab, group_tab, overlap_tab, cv_tab = st.tabs(
    ["样本鉴定量", "组鉴定量", "Overlap / Venn / UpSet", "CV"]
)

with sample_tab:
    st.subheader("每个样本的鉴定量")
    sample_counts = sample_identification_counts(abundance, rule=rule, sample_group_map=sample_group_map)
    fig = sample_bar_plot(sample_counts, group_colors)
    st.pyplot(fig, use_container_width=True)
    render_figure_downloads(fig, "sample_identification")
    plt.close(fig)

    st.dataframe(sample_counts, use_container_width=True, hide_index=True)
    d1, d2 = st.columns(2)
    d1.download_button(
        "下载 CSV",
        csv_bytes(sample_counts),
        "sample_identification.csv",
        "text/csv",
        use_container_width=True,
    )
    d2.download_button(
        "下载 XLSX",
        to_excel_bytes({"sample_identification": sample_counts}),
        "sample_identification.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
    )

with group_tab:
    st.subheader("每个组的鉴定量")
    if sample_group_map is None:
        st.info("该功能需要 metadata，用于 sample_id → group 映射。")
    else:
        include_total = st.checkbox("增加 Total（全部样本 unique）", value=True, key="include_total")
        group_counts = group_identification_counts(
            abundance,
            sample_group_map,
            rule=rule,
            include_total=include_total,
        )
        fig = group_bar_plot(group_counts, group_colors)
        st.pyplot(fig, use_container_width=True)
        render_figure_downloads(fig, "group_identification")
        plt.close(fig)
        st.dataframe(group_counts, use_container_width=True, hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("下载 CSV", csv_bytes(group_counts), "group_identification.csv", "text/csv", use_container_width=True)
        d2.download_button(
            "下载 XLSX",
            to_excel_bytes({"group_identification": group_counts}),
            "group_identification.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

with overlap_tab:
    st.subheader("Protein / Gene list overlap")
    overlap_mode_options = ["手动粘贴 / 上传多个 List"]
    if sample_group_map is not None:
        overlap_mode_options.append("从当前丰度矩阵按 group 生成 List")
    overlap_mode = st.radio("输入方式", overlap_mode_options, horizontal=True)

    raw_sets: dict[str, list[str] | set[str]] = {}
    overlap_colors: dict[str, str] = {}

    if overlap_mode.startswith("手动"):
        n_lists = st.number_input("List 数量", min_value=2, max_value=20, value=2, step=1)
        for i in range(1, int(n_lists) + 1):
            with st.expander(f"List {i}", expanded=i <= 3):
                c1, c2 = st.columns([3, 1])
                custom_name = c1.text_input("名称（留空则自动命名）", key=f"list_name_{i}")
                color = c2.color_picker("颜色", DEFAULT_COLORS[(i - 1) % len(DEFAULT_COLORS)], key=f"list_color_{i}")
                pasted = st.text_area("粘贴 ID（每行一个，也兼容逗号/Tab）", key=f"list_text_{i}", height=120)
                uploaded = st.file_uploader(
                    "或上传该 List（TXT / CSV / TSV / XLSX；表格读取第一列）",
                    type=["txt", "csv", "tsv", "xlsx"],
                    key=f"list_file_{i}",
                )
                pasted_items = [x.strip() for x in re.split(r"[\n\r\t,]+", pasted) if x.strip()]
                uploaded_items = read_list_upload(uploaded) if uploaded is not None else []
                name = custom_name.strip() or f"list{i}"
                raw_sets[name] = pasted_items + uploaded_items
                overlap_colors[name] = color
    else:
        raw_sets = detected_sets_by_group(abundance, feature_ids, sample_group_map, rule=rule)
        overlap_colors = {name: group_colors.get(name, DEFAULT_COLORS[i % len(DEFAULT_COLORS)]) for i, name in enumerate(raw_sets)}
        st.caption("按组使用 union：组内任一样本检出，该 Protein/Gene 即进入该组 list。")

    try:
        sets = normalize_sets(raw_sets)
    except ValueError as exc:
        st.error(str(exc))
        sets = {}

    if sets:
        set_sizes = pd.DataFrame({"list": list(sets), "n": [len(v) for v in sets.values()]})
        st.dataframe(set_sizes, use_container_width=True, hide_index=True)

        if len(sets) <= 4:
            fig = venn_plot(sets, overlap_colors)
            plot_prefix = "venn"
        else:
            st.info("检测到 5 个及以上 List：交集照常计算，图已自动切换为 UpSet plot。")
            fig = upset_plot(sets)
            plot_prefix = "upset"
        st.pyplot(fig, use_container_width=True)
        render_figure_downloads(fig, plot_prefix)
        plt.close(fig)

        summary = intersection_summary(sets)
        st.markdown("**交集概览**")
        st.dataframe(summary, use_container_width=True, hide_index=True)

        inclusive = inclusive_intersections(sets)
        exclusive = exclusive_regions(sets)
        sheets = {"set_sizes": set_sizes, "intersection_summary": summary}
        sheets.update({f"I_{name}": df for name, df in sets_to_sheets(inclusive).items()})
        sheets.update({f"E_{name}": df for name, df in sets_to_sheets(exclusive).items()})
        st.download_button(
            "下载全部交集 List（XLSX）",
            to_excel_bytes(sheets),
            "overlap_intersections.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        intersection_names = list(inclusive)
        if intersection_names:
            selected_intersection = st.selectbox("查看某个 inclusive intersection", intersection_names)
            selected_df = pd.DataFrame({"ID": sorted(inclusive[selected_intersection])})
            st.dataframe(selected_df, use_container_width=True, hide_index=True)
            st.download_button(
                "下载当前交集 CSV",
                csv_bytes(selected_df),
                f"{selected_intersection}.csv",
                "text/csv",
            )

with cv_tab:
    st.subheader("CV")
    st.warning("CV 建议使用未 log 转换的线性丰度数据。若输入 log2/log10 数据，CV 的解释会发生变化。")
    if sample_group_map is None:
        st.info("CV 需要 metadata，用于 sample_id → group 映射。")
    else:
        c1, c2, c3 = st.columns(3)
        min_group_size = int(c1.number_input("每组最少样本数", min_value=2, max_value=50, value=3, step=1))
        min_detect_rate = float(c2.slider("蛋白组内最低检出率", min_value=0.0, max_value=1.0, value=0.70, step=0.05))
        add_points = c3.checkbox("显示散点", value=False)

        numeric_abundance = coerce_numeric(abundance)
        finite_values = numeric_abundance.to_numpy(dtype=float)
        finite_values = finite_values[np.isfinite(finite_values)]
        if len(finite_values) and (finite_values < 0).mean() > 0.05:
            st.warning("检测到较多负值。请确认是否为 log 转换数据；软件不会自动反转换。")

        cv_result = calculate_cv_by_group(
            abundance,
            feature_ids,
            sample_group_map,
            rule=rule,
            min_group_size=min_group_size,
            min_detect_rate=min_detect_rate,
        )
        st.markdown("**分组计算状态**")
        st.dataframe(cv_result.group_status, use_container_width=True, hide_index=True)

        if cv_result.long.empty:
            st.warning("当前参数下没有可绘制的 CV。请检查组样本数、检出规则和最低检出率。")
        else:
            fig = cv_plot(cv_result.long, group_colors, add_points=add_points)
            st.pyplot(fig, use_container_width=True)
            render_figure_downloads(fig, "cv_by_group")
            plt.close(fig)

        sheets = {
            "CV_long": cv_result.long,
            "CV_wide": cv_result.wide,
            "group_status": cv_result.group_status,
        }
        st.download_button(
            "下载 CV 结果 XLSX",
            to_excel_bytes(sheets),
            "cv_results.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )
