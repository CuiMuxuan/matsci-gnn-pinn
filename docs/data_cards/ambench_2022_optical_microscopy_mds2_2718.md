# AM-Bench 2022 Optical Microscopy Data Card

## 数据定位

- Dataset id: `mds2-2718`
- DOI: https://doi.org/10.18434/mds2-2718
- NIST PDR: https://data.nist.gov/od/id/mds2-2718
- Project manifest: `configs/data/ambench_mds2_2718_sources.yaml`
- Recommended local root: `data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/`

`mds2-2718` 是 AMB2022-03 同源 optical microscopy 数据，包含 laser-scanned single tracks and pads 的截面与 top-view 图像，并带有 melt-pool cross-section measurement results。它是 Phase 17 的第一优先数据源，因为它比 `mds2-2775` 小得多、文件类型更直接，且与当前 `mds2-2716` thermography / scan strategy 主线共享 AMB2022-03 工艺上下文。

## PDR 结构摘要

当前 PDR 元数据检查结果：

| Item | Count / Size |
| --- | ---: |
| Components | 211 |
| TIFF images | 102 |
| TIFF total size | about 10.6 GB |
| SHA256 sidecars | 104 |
| README | 1 |
| Measurement XLSX | 1 |

主要子目录：

```text
Single_Track_Cross_Sections/
Pad_Cross_Sections/
Top_View_Images/
```

第一版项目 manifest 的 required files 只固定一个小子集：

```text
2718_README.txt
AMB2022-718-SH1-MeltPool_Cross-Section_Measurement_Results.xlsx
Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif
Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif.sha256
```

这样做的目的不是缩小研究对象，而是先验证真实显微图像到 graph feature 的可复现链路。

当前 optional files 进一步固定了第一版 multi-image panel，覆盖不同 `P/L/replicate` 条件和 masked/unmasked 对照：

| Image | Role |
| --- | --- |
| `AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif` | required representative masked TIFF |
| `AMB2022-718-SH1-BP1-P2-L2.1-3.tif` | same condition, unmasked counterpart |
| `AMB2022-718-SH1-BP1-P1-L3.1-3_m.tif` | additional P1/L3.1 masked condition |
| `AMB2022-718-SH1-BP1-P3-L0-2_m.tif` | additional P3/L0 masked condition |
| `AMB2022-718-SH1-BP1-P4-L0-2_m.tif` | additional P4/L0 masked condition |
| `AMB2022-718-SH1-BP1-P4-L0-2.tif` | P4/L0 unmasked counterpart |

默认下载只处理 required files；下载 multi-image panel 时必须显式使用 `--include-optional` 或逐个指定 `--file-id`，避免误拉全量 TIFF。

## 复现入口

查看计划下载内容：

```bash
python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --dry-run \
  --output outputs/data_audits/ambench_mds2_2718_download_dry_run.json
```

下载并校验第一子集：

```bash
python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_download_report.json
```

下载并校验 multi-image optional panel：

```bash
python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --include-optional \
  --verify-sha256 \
  --retries 3 \
  --timeout-seconds 300 \
  --resume-partial \
  --download-backend curl \
  --output outputs/data_audits/ambench_mds2_2718_micro_panel_download_report.json
```

只校验已下载文件：

```bash
python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_download_check.json
```

## 显微图像预处理 MVP

第一版显微图像入口使用 TIFF inspection，而不是直接训练 GNN：

```bash
python -m gnnpinn.data.loaders.ambench_microstructure \
  --image data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif \
  --sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m \
  --threshold-quantile 0.9 \
  --grid-rows 8 \
  --grid-cols 8 \
  --graph-k 4 \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json
```

输出包含：

- 图像 shape、dtype、灰度统计。
- 从文件名解析出的 `BP/P/L/replicate/view/masked` 等样品元数据。
- 阈值 mask fraction。
- coarse grid `MicrostructureGraph`。
- node features: `center_row_norm`, `center_col_norm`, `mean_intensity_norm`, `std_intensity_norm`, `mask_fraction`。
- `edge_index`: 2D kNN graph。

把 inspection JSON 聚合成训练分支可读取的 graph feature table：

```bash
python -m gnnpinn.data.loaders.ambench_microstructure \
  --mode aggregate \
  --inspection outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json \
  --jsonl-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl \
  --csv-output data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.csv \
  --output outputs/data_audits/ambench_mds2_2718_micrograph_feature_table_manifest.json
```

服务器上可用一键脚本构建 multi-image panel 的 inspection 与聚合表：

```bash
bash scripts/server/build_mds2_2718_micro_panel_a100.sh \
  > logs/ambench_mds2_2718_micro_panel_build_a100_v1.log 2>&1
```

输出的 panel feature 表为：

```text
data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl
data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.csv
```

训练入口已经支持读取该 JSONL 作为 `real_micro` graph conditioning：

```bash
python -m gnnpinn.train.macro_pinn \
  --closure-mode sparse_linear \
  --closure-graph-mode real_micro \
  --closure-graph-features data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features.jsonl \
  --closure-graph-sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m \
  --closure-graph-embedding-dim 4
```

这不是最终显微组织表征，只是 Phase 17 的低风险入口。后续应加入 melt-pool contour、区域统计、grain/phase segmentation 或 `mds2-2775`/ExaCA 生成的晶粒图作为更强 microstructure features。

## 与当前热场主线的关系

当前热场主线来自 `mds2-2716`：

```text
Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5
ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5
```

`mds2-2718` 暂时不做逐像素时空对齐，而采用样品级/区域级对齐：

1. 从 file name 解析 `BP/P/L/replicate` 等工艺标识。
2. 从 TIFF inspection 得到 sample-level / region-level micro graph features。
3. 与 `mds2-2716` 中相同或邻近工艺条件的 thermal field run 建立弱条件输入。
4. 先比较手工统计特征 vs graph features，再进入 GNN-conditioned closure。

## 风险

- TIFF 全量约 10 GB，不适合在第一步一次性下载。
- optical microscopy 与 thermography 并非天然逐像素配准，论文主张应先定位为工艺条件级或区域级 microstructure conditioning。
- `mds2-2775` 更丰富但文件数量很大，第一阶段只作为第二优先级。
