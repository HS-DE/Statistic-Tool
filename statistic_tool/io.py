from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import BinaryIO, Iterable, Mapping

import pandas as pd

SUPPORTED_EXTENSIONS = {".csv", ".tsv", ".txt", ".xlsx"}


def _file_name(file: object) -> str:
    name = getattr(file, "name", None)
    if not name:
        raise ValueError("无法识别文件名。")
    return str(name)


def read_table(file: BinaryIO | object) -> pd.DataFrame:
    """Read CSV/TSV/TXT/XLSX from a Streamlit UploadedFile or file-like object."""
    name = _file_name(file)
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError("仅支持 CSV、TSV、TXT、XLSX。")

    if hasattr(file, "seek"):
        file.seek(0)

    if suffix == ".xlsx":
        return pd.read_excel(file, engine="openpyxl")
    if suffix == ".tsv":
        return pd.read_csv(file, sep="\t")
    if suffix == ".txt":
        return pd.read_csv(file, sep=None, engine="python")
    return pd.read_csv(file)


def ensure_unique_feature_ids(df: pd.DataFrame, feature_col: str) -> None:
    if feature_col not in df.columns:
        raise ValueError(f"找不到 Protein/Gene ID 列：{feature_col}")

    ids = df[feature_col].astype("string").str.strip()
    bad = ids.isna() | ids.eq("")
    if bad.any():
        raise ValueError(f"Protein/Gene ID 列存在 {int(bad.sum())} 个 NA/空值。")

    duplicated = ids[ids.duplicated(keep=False)].drop_duplicates().tolist()
    if duplicated:
        preview = ", ".join(map(str, duplicated[:10]))
        more = "..." if len(duplicated) > 10 else ""
        raise ValueError(f"Protein/Gene ID 必须唯一；发现重复 ID：{preview}{more}")


def build_sample_group_map(
    metadata: pd.DataFrame,
    sample_id_col: str,
    group_col: str,
    abundance_columns: Iterable[str],
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Validate metadata and return matched map, matrix-only samples, metadata-only samples."""
    for col in (sample_id_col, group_col):
        if col not in metadata.columns:
            raise ValueError(f"metadata 中找不到列：{col}")

    md = metadata[[sample_id_col, group_col]].copy()
    md.columns = ["sample_id", "group"]
    md["sample_id"] = md["sample_id"].astype("string").str.strip()
    md["group"] = md["group"].astype("string").str.strip()

    bad_sample = md["sample_id"].isna() | md["sample_id"].eq("")
    if bad_sample.any():
        raise ValueError("metadata 的 sample_id 列存在 NA/空值。")

    duplicates = md.loc[md["sample_id"].duplicated(keep=False), "sample_id"].drop_duplicates().tolist()
    if duplicates:
        preview = ", ".join(map(str, duplicates[:10]))
        raise ValueError(f"metadata 中 sample_id 必须一一对应；发现重复：{preview}")

    md["group"] = md["group"].fillna("ALL").replace("", "ALL")

    abundance_cols = [str(c) for c in abundance_columns]
    meta_samples = md["sample_id"].astype(str).tolist()
    matched_samples = [c for c in abundance_cols if c in set(meta_samples)]
    if not matched_samples:
        raise ValueError("metadata 的 sample_id 与丰度矩阵列名没有任何匹配。")

    mapped = md.set_index("sample_id").loc[matched_samples].reset_index()
    matrix_only = [c for c in abundance_cols if c not in set(meta_samples)]
    metadata_only = [s for s in meta_samples if s not in set(abundance_cols)]
    return mapped, matrix_only, metadata_only


def to_excel_bytes(sheets: Mapping[str, pd.DataFrame]) -> bytes:
    buffer = BytesIO()
    used: set[str] = set()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for raw_name, df in sheets.items():
            base = _safe_sheet_name(raw_name)
            sheet_name = base
            suffix = 2
            while sheet_name.lower() in used:
                tail = f"_{suffix}"
                sheet_name = f"{base[:31 - len(tail)]}{tail}"
                suffix += 1
            used.add(sheet_name.lower())
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    buffer.seek(0)
    return buffer.getvalue()


def _safe_sheet_name(name: str) -> str:
    bad = set('[]:*?/\\')
    cleaned = "".join("_" if c in bad else c for c in str(name)).strip() or "Sheet"
    return cleaned[:31]
