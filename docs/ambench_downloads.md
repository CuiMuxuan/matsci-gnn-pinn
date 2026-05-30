# AM-Bench 下载与校验说明

## 当前主线下载目标

第一批只下载 AMB2022-03 thermography / scan strategy：

- DOI: https://doi.org/10.18434/mds2-2716
- NIST PDR 记录页: https://data.nist.gov/od/id/mds2-2716
- README: https://data.nist.gov/od/ds/mds2-2716/2716_README.txt
- 仓库内固定清单: `configs/data/ambench_mds2_2716_sources.yaml`
- 推荐本地目录：

```text
data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/
```

最少应包含：

```text
2716_README.txt
Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5
ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5
```

`SamplePhotos/` 可选。

## 推荐下载流程

本项目已经把 AMB2022-03 / `mds2-2716` 的下载地址、文件大小和 SHA256 固定到：

```text
configs/data/ambench_mds2_2716_sources.yaml
```

后续本机或服务器复现时，优先使用项目内 Python 下载入口。它会：

- 自动创建目标目录。
- 跳过已经存在且大小正确的文件。
- 下载缺失文件或大小不匹配的文件。
- 使用 `--verify-sha256` 校验下载结果。
- 如果校验未通过，返回非零退出码，避免后续实验误用不完整数据。

Windows 本机推荐命令：

```powershell
cd C:\code\GNN-PINN

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:CONDA_NO_PLUGINS="true"

conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 `
  --download `
  --verify-sha256 `
  --output outputs/data_audits/ambench_mds2_2716_download_report.json
```

Linux/云服务器推荐命令：

```bash
cd /path/to/GNN-PINN

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2716_download_report.json
```

只想查看将下载哪些文件，不实际写入：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 `
  --download `
  --dry-run `
  --output outputs/data_audits/ambench_mds2_2716_download_dry_run.json
```

## 单文件重下

热像 HDF5 文件约 550 MB，网络不稳时可能出现下载到一半就结束的情况。若报告中出现：

```json
"mismatched_required": ["thermography_signal_hdf5"]
```

或者热像文件大小不是：

```text
549979044 bytes
```

则只重下热像大文件：

```powershell
cd C:\code\GNN-PINN

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:CONDA_NO_PLUGINS="true"

conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 `
  --download `
  --overwrite `
  --file-id thermography_signal_hdf5 `
  --verify-sha256 `
  --output outputs/data_audits/ambench_mds2_2716_thermography_redownload_report.json
```

只重下 ScanStrategy 小文件：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 `
  --download `
  --overwrite `
  --file-id scan_strategy_hdf5 `
  --verify-sha256 `
  --output outputs/data_audits/ambench_mds2_2716_scan_strategy_redownload_report.json
```

成功时，输出 JSON 中应包含：

```json
"ready_for_hdf5_adapter": true
```

并且 `mismatched_required` 应为空列表。

## DOI 或网页不可用时的下载

如果 `https://doi.org/10.18434/mds2-2716` 返回 502，或 `https://www.nist.gov/el/ammt/datasets` 页面找不到展开文件，不代表数据消失。当前可用的机器可读入口是：

```text
https://data.nist.gov/od/id/mds2-2716
```

该 PDR 元数据记录返回了 HDF5 文件、大小和 SHA256。当前第一批真实实验只需要两个 HDF5：

| 文件 | 大小 | SHA256 |
| --- | ---: | --- |
| `Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5` | 549979044 | `f6fe21ec911707f72e7efda2932c77eae2b75d84765848878fe5beb6b728cd43` |
| `ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5` | 406992 | `7b7004753e150bc26632e9ce356e0440429160fa92cbff8fc8559202fdce2103` |

如果项目内 Python 下载入口不可用，也可以使用 PowerShell + `curl.exe` 直接下载：

```powershell
cd C:\code\GNN-PINN
New-Item -ItemType Directory -Force -Path data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/Thermography,data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy

curl.exe -L "https://data.nist.gov/od/ds/ark:/88434/mds2-2716/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5" -o "data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/Thermography/AMB2022-03-718-AMMT-StaringCamera_Signal.h5"

curl.exe -L "https://data.nist.gov/od/ds/ark:/88434/mds2-2716/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5" -o "data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716/ScanStrategy/AMB2022-03-AMMT-718-Pad_XYPT.h5"
```

## 下载后验证

手动下载或复制文件后，在项目根目录运行：

```powershell
cd C:\code\GNN-PINN

$env:PYTHONUTF8="1"
$env:PYTHONIOENCODING="utf-8"
$env:CONDA_NO_PLUGINS="true"

conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 `
  --verify-sha256 `
  --output outputs/data_audits/ambench_mds2_2716_download_check.json
```

验证通过后，报告中的关键字段应为：

```json
{
  "ready_for_hdf5_adapter": true,
  "missing_required": [],
  "mismatched_required": []
}
```

若只下载了部分文件，报告会列出 `missing_required`；若文件大小或 SHA256 不匹配，报告会列出 `mismatched_required`。这两种情况都不要继续做真实数据转换或训练。

## 验证通过后的第一步转换

当 `ready_for_hdf5_adapter` 为 `true` 后，可先转换一个轻量真实子集。默认选择：

```text
ThermalData/Line_0_1/Signal
```

输出字段表包含 `x,y,z,t,signal`，其中 `x/y` 是相机像素索引，`t` 来自 HDF5 的 `frame_rate`，`signal` 是 raw digital level。第一版真实 smoke 不把它直接命名为温度，避免在校准公式完全确认前引入物理单位误差。

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.loaders.ambench_hdf5 `
  --output data/interim/ambench/2022_single_track/AMB2022-03/line_0_1_signal_subset.csv `
  --manifest outputs/data_audits/ambench_line_0_1_hdf5_conversion_manifest.json `
  --split-manifest outputs/data_splits/ambench_line_0_1_signal_subset_split.json
```

## 常见问题

### 下载时间太短或中途断开

如果下载进程提前结束，可能得到一个能打开路径但大小不对的 HDF5。典型表现是：

```text
actual_size_bytes: 211028490
expected_size_bytes: 549979044
```

处理方式是使用“单文件重下”中的 `--overwrite --file-id thermography_signal_hdf5` 命令。不要手动改校验值，也不要继续使用这个不完整文件。

### DOI 返回 502

直接使用 `https://data.nist.gov/od/id/mds2-2716` 或本项目 manifest 中的 `download_url`。DOI 中转页异常不影响 PDR 直链可用性。

### 服务器环境没有 `gnnpinn-cu130`

云服务器环境名不必和本机一致。把命令中的 `gnnpinn-cu130` 替换为服务器上的环境名，例如：

```bash
conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2716 --download --verify-sha256
```

## Phase 17 optical microscopy 下载入口

方向三真实/半真实微观组织阶段优先使用 AMB2022-03 optical microscopy：

- DOI: https://doi.org/10.18434/mds2-2718
- NIST PDR 记录页: https://data.nist.gov/od/id/mds2-2718
- 仓库内固定清单: `configs/data/ambench_mds2_2718_sources.yaml`
- 推荐本地目录：

```text
data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/
```

该 PDR 记录全量包含 102 个 TIFF 图像，合计约 10.6 GB。项目第一版 required files 只固定一个低风险子集：README、melt-pool measurement XLSX、一个 representative single-track cross-section TIFF 及其 `.sha256`。

Phase 17b 已在 `optional_files` 中固定一个更小的 multi-image single-track panel，用于把单张显微图扩展到多工艺条件 graph feature 表。面板包含：

| ID | Purpose |
| --- | --- |
| `single_track_cross_section_p2_l2_1_r3_unmasked_tif` | required representative masked image 的 unmasked counterpart |
| `single_track_cross_section_p1_l3_1_r3_masked_tif` | P1/L3.1/replicate-3 masked condition |
| `single_track_cross_section_p3_l0_r2_masked_tif` | P3/L0/replicate-2 masked condition |
| `single_track_cross_section_p4_l0_r2_masked_tif` | P4/L0/replicate-2 masked condition |
| `single_track_cross_section_p4_l0_r2_unmasked_tif` | P4/L0/replicate-2 unmasked counterpart |

默认下载命令仍只拉 required files。只有显式传入 `--include-optional`，或用 repeated `--file-id` 指定上述 ID 时，才会下载 optional panel。

查看计划下载内容：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --dataset-id mds2-2718 `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 `
  --download `
  --dry-run `
  --output outputs/data_audits/ambench_mds2_2718_download_dry_run.json
```

下载并校验第一显微子集：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.ambench_downloads `
  --dataset-id mds2-2718 `
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 `
  --download `
  --verify-sha256 `
  --output outputs/data_audits/ambench_mds2_2718_download_report.json
```

如果某个 TIFF 下载中途断开，报告会出现 `"status": "incomplete"`，并留下 `.part` 文件。此时只重试对应 TIFF：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --overwrite \
  --file-id single_track_cross_section_representative_tif \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_tif_redownload_report.json
```

Linux/云服务器命令：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_download_report.json
```

下载并校验 optional micro panel：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --include-optional \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_micro_panel_download_report.json
```

如果只想补其中某一张图，用 `--file-id` 精确下载。`--file-id` 可以选择 required 或 optional 文件：

```bash
PYTHONUTF8=1 PYTHONIOENCODING=utf-8 conda run -n gnnpinn python -m gnnpinn.data.ambench_downloads \
  --dataset-id mds2-2718 \
  --root data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718 \
  --download \
  --overwrite \
  --file-id single_track_cross_section_p4_l0_r2_masked_tif \
  --verify-sha256 \
  --output outputs/data_audits/ambench_mds2_2718_p4_l0_r2_masked_redownload_report.json
```

下载后生成第一版显微图像 inspection / coarse micro graph：

```powershell
conda run -n gnnpinn-cu130 python -m gnnpinn.data.loaders.ambench_microstructure `
  --image data/raw/ambench/2022_single_track/AMB2022-03/mds2-2718/Single_Track_Cross_Sections/AMB2022-718-SH1-BP1-P2-L2.1-3_m.tif `
  --sample-id AMB2022-718-SH1-BP1-P2-L2.1-3_m `
  --threshold-quantile 0.9 `
  --grid-rows 8 `
  --grid-cols 8 `
  --graph-k 4 `
  --output outputs/data_audits/ambench_mds2_2718_micrograph_inspection.json
```

服务器上一键构建 multi-image micro panel：

```bash
bash scripts/server/build_mds2_2718_micro_panel_a100.sh \
  > logs/ambench_mds2_2718_micro_panel_build_a100_v1.log 2>&1
```

该脚本会生成：

```text
outputs/data_audits/ambench_mds2_2718_micro_panel_download_report.json
outputs/data_audits/mds2_2718_micro_panel/*_inspection.json
outputs/data_audits/ambench_mds2_2718_micro_panel_feature_table_manifest.json
data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.jsonl
data/processed/ambench/2022_single_track/AMB2022-03/mds2-2718/micro_graph_features_panel.csv
```

## 稍后可下载

第二阶段再下载：

- Cross-sectional microstructure, AMB2022-03: https://doi.org/10.18434/mds2-2775

当前阶段不需要手动下载电池、多孔介质、PFHub 或 ExaCA。它们是备选路线或仿真增强资源；只有当 AM-Bench optical microscopy 路线无法形成可用 micro graph 或可解释对齐时，再切换到 `mds2-2775`、ExaCA 或 PFHub。
