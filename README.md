# Statistic Tool

一个面向日常蛋白组数据 QC / summary 的轻量 Python + Streamlit 工具。

## 当前功能

- **每个样本鉴定量**：0 / NA / 空值默认视为未检出；有 metadata 时同组同色，无 metadata 时统一颜色。
- **每个组鉴定量**：组内任一样本检出即视为该组检出（union），可附加全部样本的 Total unique。
- **Overlap**：支持多个 Protein/Gene list，自定义中英文名称和颜色；2–4 个 list 使用 Venn，5 个及以上自动使用 UpSet；同时导出 inclusive intersections 和 exclusive regions。
- **CV**：按组逐 Protein/Gene 计算 `SD / Mean × 100%`；默认每组至少 3 个样本，最低可设为 2；样本数不足的组直接跳过。

## 输入文件

丰度矩阵和 metadata 均支持：

- CSV
- TSV
- TXT
- XLSX

上传后由用户自己指定：

- Protein / Gene ID 列
- metadata 的 sample_id 列
- metadata 的 group 列

metadata 的 sample_id 必须与丰度矩阵的样本列名一一对应。

## 检出规则

所有依赖“是否检出”的分析统一调用 `statistic_tool.detection.is_detected()`。第一版提供：

1. `nonzero`（默认）：非 0、非 NA、非空；负数也可以是有效值。
2. `positive`：仅 > 0。
3. `non_na`：非 NA / 非空，0 也算检出。

新增公司内部检测规则时，只需要扩展这一层，不需要在各页面重复实现。

## CV 提示

CV 建议使用**未 log 转换的线性丰度数据**。软件检测到较多负值时会提示，但不会擅自转换输入数据。

CV 中检测规则用于决定哪些值参与有效值计数和均值/标准差计算。默认最低检出率为 70%。

## 运行

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

浏览器会自动打开本地页面。示例数据见 `examples/`。

## 项目结构

```text
Statistic-Tool/
├── app.py
├── statistic_tool/
│   ├── detection.py
│   ├── io.py
│   ├── identification.py
│   ├── overlap.py
│   ├── cv.py
│   └── plotting.py
├── tests/
├── examples/
├── requirements.txt
└── README.md
```

## 验证

核心统计逻辑使用标准库 `unittest`：

```bash
python -m unittest discover -s tests -v
```

> 当前仓库的核心逻辑来源于公司现有 R 脚本定义，并在 Python 中重新实现；没有依赖 R 运行环境。
