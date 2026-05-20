# SynthesizeBrain

*[English](README.md) | 繁體中文*

把 FlyCircuit 的單神經元 warp 體積影像，打包成成對的 **intensity + instance-label** 3D 體積資料，供**密集染色腦影像的 single-neuron auto-segmentation 訓練**使用。

## 使用方式

### 安裝

```bash
pip install -r requirements.txt
```

主要相依：`numpy` / `scipy` / `nibabel` / `matplotlib` / `Pillow` / `imageio` / `imageio-ffmpeg` / `tqdm`。Python 3.10+。

### 輸入資料

需要一個資料夾放 FlyCircuit warp volumes（每顆 neuron 一個 `.am` 檔，AmiraMesh BINARY-LITTLE-ENDIAN 2.1 格式）。

預設路徑：`C:\Users\USER\Work\Kaleido\warp\`

裡面應該有像這樣的檔名（每顆神經元獨立一檔）：

```
Tdc2-F-000000_seg001_warp_volume.am
Trh-F-000123_seg001_warp_volume.am
VGlut-F-500740_seg001_warp_volume.am
fru-F-900058_seg001_warp_volume.am
...
```

要改路徑，每個 subcommand 加 `--warp-dir <path>`。

### 首次設定（只跑一次）

```bash
python synthesize.py init
```

掃 warp 資料夾、為每顆 neuron 建 metadata 索引（lattice 尺寸、bbox、tight 非零 bbox、voxel 數量），存成 `cache/warp_index.npz`（~2 MB）。23 workers 約 18 秒。後續所有指令都從這份索引讀。

只在來源資料 `Kaleido\warp\` 變動時才需重跑。

### 日常指令

```bash
# 合成一次新 run（最常用）— 預設 N=500、隨機 seed
python synthesize.py synthesize

# 用指定 seed 重現某次 run
python synthesize.py synthesize --seed 2454876261

# 迴圈產出多組訓練樣本（每次不同 neuron 組合）
for _ in 1 2 3 4 5; do python synthesize.py synthesize; done             # bash / git-bash
1..5 | ForEach-Object { python synthesize.py synthesize }                # PowerShell
for /L %i in (1,1,5) do python synthesize.py synthesize                  # Windows cmd.exe

# 一次掃多個 N（測天花板）
python synthesize.py sweep --ns 10 50 100 500

# 對既有 output 補產 contacts.csv
python synthesize.py contacts --dir output/output_N500_K148_R150_s.../

# 對既有 output 補產 scan_video.mp4
python synthesize.py video --dir output/output_N500_K148_R150_s.../

# 對既有 output 補加 / 改 Gaussian 雜訊
python synthesize.py noise --dir output/output_N500_K148_R150_s.../ \
                           --sigma 50 --baseline 100
```

典型流程就是 `init` 一次 + `synthesize` 跑 N 次。其他 subcommand 按需求擇一使用，不用串。

### 全參數總表

#### `init` — 建快取（一次性）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--warp-dir` | `C:\Users\USER\Work\Kaleido\warp` | warp `.am` 檔資料夾 |
| `--cache` | `cache/warp_index.npz` | 快取輸出位置 |
| `--workers` | cpu_count − 1 | 平行 worker 數 |
| `--limit` | 全部 | 只掃前 N 檔（smoke test 用） |

#### `synthesize` — 合成單次配對體積

| 參數 | 預設 | 說明 |
|---|---|---|
| `--n` | `500` | 請求 neuron 數量。實際 K 可能更小（packing 飽和） |
| `--seed` | 隨機 | 隨機 seed。省略時抽新 seed、印出並嵌入資料夾名稱以利重現 |
| `--out` | 自動命名 | 覆寫輸出資料夾路徑（預設 `output/output_N{req}_K{c1}_R{total}_s{seed}/`） |
| `--cache` | `cache/warp_index.npz` | 讀取快取 |
| `--warp-dir` | 同 init | warp `.am` 檔資料夾 |
| `--rand-sigma` | `0.25` | Score 雜訊（fraction of score std）。0 = 完全 deterministic |
| `--coverage-threshold` | `0.5` | C1 bbox 覆蓋率閾值 [0, 1]。值越低 → 包裝越鬆、repair 砍越少。`0` 等於關掉 C1 |
| `--score-mode` | `density` | Greedy 排序。`density` = 從最密腦區優先（預設，視覺上聚集）。`small-first` = 從最瘦 neuron 優先（多 ~30% 顆，見「最大化 K」）。`hybrid` = score / nnz |
| `--noise-sigma` | `50` | intensity.am 加的 Gaussian 雜訊 σ（0 = 不加） |
| `--noise-baseline` | `100` | 加 noise 前的基線偏移，讓背景仍是純 Gaussian（~ sCMOS dark-current level） |
| `--no-video` | 關 | 略過 `scan_video.mp4` 產生（省 ~50 秒/次，批次跑用） |

#### `sweep` — 一次跑多個 N

| 參數 | 預設 | 說明 |
|---|---|---|
| `--ns` | `10 50 100 500` | 要跑的 N 列表 |
| `--out-root` | `output` | 各 run 輸出的根目錄 |
| `--cache` / `--warp-dir` / `--seed` / `--rand-sigma` / `--coverage-threshold` / `--score-mode` / `--noise-sigma` / `--noise-baseline` | 同 synthesize | |

#### `contacts` — 重算 F/E/V 接觸統計

| 參數 | 預設 | 說明 |
|---|---|---|
| `--dir` | 必填 | 既有 output 資料夾 |

#### `video` — 重產 scan_video.mp4

| 參數 | 預設 | 說明 |
|---|---|---|
| `--dir` | 必填 | 既有 output 資料夾 |
| `--fps` | `3` | 影片 frame rate（每顆約 0.33 秒） |

#### `noise` — 對既有 output 補加 / 改雜訊

| 參數 | 預設 | 說明 |
|---|---|---|
| `--dir` | 必填 | 既有 output 資料夾 |
| `--sigma` | `50` | Gaussian σ |
| `--baseline` | `100` | 基線偏移 |
| `--seed` | `0` | RNG seed |

---

## 輸出格式

每次 run 產生一個資料夾 `output/output_N{req}_K{c1}_R{total}_s{seed}/`，命名欄位意義：

- `N` = 命令列請求的神經元數量
- `K` = C1 保障下收的數量（greedy + repair 生還者）
- `R` = 總產出 = K + expand 階段加碼（R ≥ K；expand 放寬 C1 把還能塞的 neuron 都塞進去，提高訓練密度）
- `s` = 實際使用的隨機 seed（用 `--seed s` 即可重現）

資料夾內容：

- `intensity.am` / `intensity.nii.gz` — uint16，voxel 值 = 原始 warp 強度（神經元所在位置），背景 0 + Gaussian 雜訊。
- `labels.am` / `labels.nii.gz` — uint16，每 voxel 是 instance label ID（1..K），0 = 背景。與 `intensity.*` 1 對 1 對齊。
- `neuron_list.tsv` — `label_id, filename, driver, phase, voxel_count, bbox_coverage, origin_{ix,iy,iz}, lattice_{nx,ny,nz}`。`phase` 欄是 `greedy` / `repair` / `expand` — 見下方「Selection 階段」。
- `contacts.csv` — 兩兩 neuron 的 F/E/V 接觸 voxel 對數統計（見下方「成對接觸統計」）。
- `mip.png` — 三軸 MIP 預覽圖（上排灰階 intensity、下排亂色 label）。
- `scan_video.mp4` — 逐 label 掃描影片（新 neuron 亮白、舊的暗灰、左上角字卡 N = i）。

## 包裝限制

選出的 `N` 顆神經元必須滿足：

1. **BBox 覆蓋率 ≥ 閾值**（C1）：對每顆 neuron *n*，`|bbox(n) ∩ ⋃_{m≠n} bbox(m)| / |bbox(n)| ≥ T`，預設 `T = 0.5`，可用 `--coverage-threshold` 調整（voxel 計數一次）。防止空間分散，不然 segmentation 太簡單。設 `T = 0` 等於關閉 C1（只剩 C2 voxel 互斥要過）。
2. **Voxel 層級互斥**（C2）：入選 neuron 的非零 voxel 兩兩不重疊（可以觸碰但不可重疊）。真實密集腦組織不會重疊；warp 自不同個體會。

同一次合成允許混多個 driver（Tdc2 / Trh / VGlut / fru）。

## Selection 三階段

選樣器分三階段跑，每顆 neuron 的階段歸屬都記在 `neuron_list.tsv → phase` 欄：

1. **greedy** — 依 score 由高至低試裝，到 N 或 candidate 用完為止。只檢查 voxel 互斥（C2），**還不管 C1**。
2. **repair** — 驗證每顆 neuron 的 C1 覆蓋率；違規者丟掉（釋放其 voxel 與 bbox 貢獻），從剩餘 pool 補進不違規的。最多 5 輪。此階段結束時，**留下的都滿足 C1**。
3. **expand** — 在沒有更多候選能替換 violator 時，對剩餘 candidate 再掃一次，凡是 voxel 仍能塞進去的就收 — **忽略 C1**。這些是 training 密度的 bonus，不在 C1 保障內。

## 最大化 K（盡可能塞多顆 neuron）

預設配置下 K ≈ 140–180 飽和，原因是 greedy 階段**優先放大顆神經元**，把後續小顆的空間堵死。想塞更多用：

```bash
python synthesize.py synthesize --n 500 --score-mode small-first --coverage-threshold 0
```

`small-first` 把 greedy 改成 nnz 由小到大排序 — 瘦神經元先擠進大顆留下的縫隙。配合 `--coverage-threshold 0`（關掉 C1 repair 的 churn）通常可以**多 ~30% 顆**，代價是平均 bbox 覆蓋率稍降（如 mean 0.86 → 0.83）。在 28 620 顆的 dataset 上同 seed 對比：

| `--score-mode` | `--coverage-threshold` | K | R | 平均覆蓋率 |
|---|---|---|---|---|
| `density`（預設） | `0.5` | 147 | 148 | 0.860 |
| `small-first` | `0.5` | 191 | 195 | 0.838 |
| `small-first` | `0.0` | **193** | 193 | 0.832 |
| `hybrid` | `0.5` | 169 | 173 | 0.844 |

Trade-off：neuron 數變多，但**總非零 voxel 量變少**（小神經元每顆 voxel 少）、低覆蓋率的 long tail 變長（如 7 顆 < 0.5，相對於 `density` 的 1 顆）。若訓練重點在 instance 多樣性，選 `small-first`；要逼近真實密集染色外觀，選 `density`。

## intensity.am 的 Gaussian 雜訊

模型：`intensity' = intensity + baseline + N(0, σ)`，clip 到 `[0, 65535]`。

預設 σ = 50、baseline = 100。**baseline 偏移的用意**是讓背景落在乾淨的 N(100, 50) 而非 clip-at-0 造成的折疊 Gaussian — 模擬真實 sCMOS / CCD 相機的 dark-current offset。

實測驗證：σ = 50 / baseline = 100 時，背景 10×10×10 corner mean ≈ 97.8、std ≈ 50.2，與理論 N(100, 50) 相符。`labels.am` 完全不加雜訊 — ground truth 保持乾淨。

## 在 Avizo 裡逐顆檢查 label

用 Avizo 打開 `labels.am`（label field）+ `colormaps/bandpass_white.am`，把 colormap port 的 `MinMax` 設為：

| 目標 | MinMax |
|---|---|
| 只看 label N | `[N - 0.5, N + 0.5]` |
| 看 label A..B（含端點） | `[A - 0.5, B + 0.5]` |
| 逐顆掃描 | `[0.5, 1.5]` → `[1.5, 2.5]` → ... → `[K - 0.5, K + 0.5]` |

**為什麼是 0.5 偏移**：Avizo 會把 `[N, N]` 零寬區間當成空集合。留半個 unit 的寬度才涵蓋到整數 label N。細神經元可能還是不易察覺，這時把 Volume Rendering 的 Composition 改成 `max`（MIP 模式）、或改用 Ortho Slice 切片看。

## 成對接觸統計

`contacts.csv` 欄位：`neuron1, neuron2, N_F, N_E, N_V`。只列出有接觸的對（非零 voxel 對數 > 0）：

- **F** = 面接觸（voxel 共用一整面）
- **E** = 邊接觸（共用邊、但非面）
- **V** = 頂點接觸（共用頂點、但非邊）

## 經驗性的包裝天花板

對 `Kaleido\warp\` 的 9987 顆 FlyCircuit warp：

| N 請求 | K (under C1) | R (total) | 備註 |
|-------:|-------------:|----------:|------|
| 10     | 10           | 10        | C1 無壓力 |
| 50     | 50           | 50        | C1 OK |
| 100    | 100          | 100       | C1 OK |
| 500    | ~130–150     | ~135–160  | 隨 seed 變化；voxel 互斥導致飽和 |

Voxel 互斥下的包裝約在 **R ≈ 140–160** 飽和 — 再多就放不下了。每次換 seed 會探索到不同的 local optimum；N=500 重複跑會得到不同 K/R 與不同 neuron 組合，這正是產多組訓練資料的預期行為。

11 runs × N=500 的實測：1079 顆 unique neurons、平均 pairwise Jaccard 0.06、70% 的 neuron 只出現在單一 run — 訓練多樣性高。

## 演算法草圖

- **Phase 1 — `synthesize_brain/index.py`**：掃每一個 `.am`、記錄 lattice 尺寸、header bbox、**非零 voxel 的 tight bbox**、非零 voxel 數。Canvas = 全部 header bbox 聯集、round 到整數 voxel。
- **Phase 2 — `synthesize_brain/select.py`**：
  1. 對全部 dataset 建 canvas-voxel coverage counter；對每顆 neuron 用其 tight bbox 內的平均 coverage 當分數（中心 = 高）。
  2. Greedy 依分數由高到低試裝（tie-break：小顆優先），撞到 `occupied_mask` 就跳過。I/O 平行跑 8 個 thread + 有限 prefetch window；被拒絕的讀取保留在共享 in-memory cache、repair 階段不用重解壓。
  3. 用增量維護的 per-voxel bbox counter 驗 C1；剔除 violator、從剩餘池按分數補、最多 5 輪。
- **Phase 3 — `synthesize_brain/compose.py`**：把每顆入選 neuron 的非零 voxel 貼進 `intensity`、對應位置寫 label ID 進 `labels`。
- **Phase 4 — writers**：`amira_io.py`（raw AmiraMesh ushort + label field）、`nifti_io.py`（`.nii.gz` via nibabel）、`mip.py`（三軸 MIP PNG）、`contacts.py`（兩兩 F/E/V 接觸 voxel 統計）、`scan_video.py`（逐 label 掃描 MP4）、`noise.py`（Gaussian 雜訊）。

## 專案結構

```
synthesize.py                  # CLI 入口
synthesize_brain/
├── amira_io.py                # AmiraMesh reader（從 Kaleido 移植）+ ushort / label writer
├── compose.py                 # Phase 3
├── contacts.py                # 成對 F/E/V 接觸統計
├── index.py                   # Phase 1（init 子指令的實作）
├── mip.py                     # MIP 預覽
├── nifti_io.py                # NIfTI 寫檔
├── noise.py                   # Gaussian 雜訊
├── scan_video.py              # 逐 label 掃描 MP4
└── select.py                  # Phase 2
colormaps/bandpass_white.am    # Avizo 用的 bandpass colormap
check_saturation.py            # 診斷：最後的 K 是不是真的 voxel 飽和？
analyze_overlap.py             # 多個 run 之間的 neuron 重疊度分析
cache/                         # warp_index.npz, candidate_scores.npz
output/                        # output_N{req}_K{c1}_R{total}_s{seed}/, sweep_summary.tsv
tasks/                         # todo.md, lessons.md
docs/                          # PPTX 簡報、tutorial、build 腳本
```
