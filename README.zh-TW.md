# SynthesizeBrain

*[English](README.md) | 繁體中文*

把 FlyCircuit 的單神經元 warp 體積影像，打包成成對的 **intensity + instance-label** 3D 體積資料，供**密集染色腦影像的 single-neuron auto-segmentation 訓練**使用。

## 輸出

每次 run 產生一個資料夾 `output/output_N{req}_K{c1}_R{total}_s{seed}/`，命名欄位意義：

- `N` = 命令列請求的神經元數量
- `K` = C1 保障下收的數量（greedy + repair 生還者）
- `R` = 總產出 = K + expand 階段加碼（R ≥ K；expand 放寬 C1 把還能塞的 neuron 都塞進去，提高訓練密度）
- `s` = 實際使用的隨機 seed（用 `--seed s` 即可重現）

資料夾內容：

- `intensity.am` / `intensity.nii.gz` — uint16。voxel 值 = 原始 warp 的強度（神經元所在位置），背景 0。
- `labels.am` / `labels.nii.gz` — uint16，每 voxel 是 instance label ID（1..K），0 = 背景。與 `intensity.*` 1 對 1 對齊。
- `neuron_list.tsv` — `label_id, filename, driver, phase, voxel_count, bbox_coverage, origin_{ix,iy,iz}, lattice_{nx,ny,nz}`。`phase` 欄是 `greedy` / `repair` / `expand` — 見下方「Selection 階段」。
- `contacts.csv` — 兩兩 neuron 的 F/E/V 接觸 voxel 對數統計（見下方「成對接觸統計」）。
- `mip.png` — 三軸 MIP 預覽圖（上排灰階 intensity、下排亂色 label）。
- `scan_video.mp4` — 逐 label 掃描影片（新 neuron 亮白、舊的暗灰、左上角字卡 N = i）。

## intensity.am 的 Gaussian 雜訊

每次 synthesize 都會在 `intensity.am` 加上 `baseline + N(0, σ)` 雜訊（預設 σ = 50、baseline = 100），並 clip 到 `[0, 65535]` 保持 uint16 範圍。**baseline 偏移的用意**是讓背景落在乾淨的 N(100, 50) 而非 clip-at-0 造成的折疊 Gaussian — 模擬真實 sCMOS / CCD 相機的 dark-current offset。

實測驗證：σ = 50 / baseline = 100 時，背景 10×10×10 corner mean ≈ 97.8、std ≈ 50.2，與理論 N(100, 50) 相符。

關閉雜訊：

```bash
python synthesize.py synthesize --noise-sigma 0 --noise-baseline 0
```

對既有 output 資料夾補加雜訊（重寫 `intensity.am` / `intensity.nii.gz` / `mip.png`；`labels.am`、`contacts.csv`、`scan_video.mp4` 不動）：

```bash
python synthesize.py noise --dir output/output_N500_K139_R141_s3460920629 \
    --sigma 50 --baseline 100
```

---

## 在 Avizo 裡逐顆檢查 label

用 Avizo 打開 `labels.am`（label field）+ `colormaps/bandpass_white.am`，把 colormap port 的 `MinMax` 設為：

| 目標 | MinMax |
|---|---|
| 只看 label N | `[N - 0.5, N + 0.5]` |
| 看 label A..B（含端點） | `[A - 0.5, B + 0.5]` |
| 逐顆掃描 | `[0.5, 1.5]` → `[1.5, 2.5]` → ... → `[K - 0.5, K + 0.5]` |

**為什麼是 0.5 偏移**：Avizo 會把 `[N, N]` 零寬區間當成空集合，什麼都不顯示。留半個 unit 的寬度才涵蓋到整數 label N。細神經元可能還是不易察覺，這時把 Volume Rendering 的 Composition 改成 `max`（MIP 模式）、或改用 Ortho Slice 切片看。

---

## 成對接觸統計

`contacts.csv` 欄位：`neuron1, neuron2, N_F, N_E, N_V`。只列出有接觸的對（非零 voxel 對數 > 0）：

- **F** = 面接觸（voxel 共用一整面）
- **E** = 邊接觸（共用邊、但非面）
- **V** = 頂點接觸（共用頂點、但非邊）

## 包裝限制

選出的 `N` 顆神經元必須滿足：

1. **BBox 覆蓋率 ≥ 50 %**：對每顆 neuron *n*，`|bbox(n) ∩ ⋃_{m≠n} bbox(m)| / |bbox(n)| ≥ 0.5`（voxel 計數一次）。防止空間分散，不然 segmentation 太簡單。
2. **Voxel 層級互斥**：入選 neuron 的非零 voxel 兩兩不重疊（可以觸碰但不可重疊）。真實密集腦組織不會重疊；warp 自不同個體會。

同一次合成允許混多個 driver（Tdc2 / Trh / VGlut / fru）。

## 安裝

```bash
pip install -r requirements.txt
```

## 使用方式

### 首次設定（只跑一次）

```bash
# 掃 warp 資料夾、建 per-neuron metadata 快取。23 workers ~18 秒。
# 只在來源資料 Kaleido/warp/ 改變時才需重跑。
python synthesize.py index
```

會寫出 `cache/warp_index.npz`（~2 MB）。之後所有指令都從它讀。

### 日常指令（擇一使用）

```bash
# A. 合成一次新 run。預設 N=500、每次隨機 seed。
python synthesize.py synthesize

# B. 用指定 seed 重現某次 run。
python synthesize.py synthesize --seed 2454876261

# C. 迴圈產出多組訓練樣本（每次不同 neuron 組合）。
for i in 1 2 3 4 5; do python synthesize.py synthesize; done

# D. 一次掃多個 N（例如測天花板）。
python synthesize.py sweep --ns 10 50 100 500

# E. 對既有 output 目錄重算 contacts.csv。
python synthesize.py contacts --dir output/output_N500_K148_R150_s.../

# F. 對既有 output 目錄重產 scan_video.mp4。
python synthesize.py video --dir output/output_N500_K148_R150_s.../
```

**典型流程就是 `index` 一次 + `synthesize` 跑 N 次**。其他 subcommand 是因應特定需求（重現、批次掃、補產 contacts / video），不用串。

預設值：

- warp dir：`C:\Users\USER\Work\Kaleido\warp`
- cache：`cache/warp_index.npz`（加上首次選樣後會有 `cache/candidate_scores.npz` — 這份快取把 coverage-map 打分的 ~1 分鐘花費平攤到後續每次 run）
- output：`output/output_N{req}_K{c1}_R{total}_s{seed}/`（命名同時揭露 C1 保障數 K、expand 後總數 R；多組隨機 run 可以共存不會蓋掉）
- canvas：所有 9987 顆 bbox 聯集動態算出（目前資料集是 989 × 646 × 337，voxel spacing 1）

## Selection 三階段

選樣器分三階段跑，每顆 neuron 的階段歸屬都記在 `neuron_list.tsv → phase` 欄：

1. **greedy** — 依 score 由高至低試裝，到 N 或 candidate 用完為止。只檢查 voxel 互斥（C2），**還不管 C1**。
2. **repair** — 驗證每顆 neuron 的 C1 覆蓋率；違規者丟掉（釋放其 voxel 與 bbox 貢獻），從剩餘 pool 補進不違規的。最多 5 輪。此階段結束時，**留下的都滿足 C1**。
3. **expand** — 在沒有更多候選能替換 violator 時，對剩餘 candidate 再掃一次，凡是 voxel 仍能塞進去的就收 — **忽略 C1**。這些是 training 密度的 bonus，不在 C1 保障內。

加 `--rand-sigma 0` 可以關閉 score 雜訊、回到純 deterministic 選樣順序。

## 經驗性的包裝天花板

對 `Kaleido\warp\` 的 9987 顆 FlyCircuit warp：

| N 請求 | K (under C1) | R (total) | 備註 |
|-------:|-------------:|----------:|------|
| 10     | 10           | 10        | C1 無壓力 |
| 50     | 50           | 50        | C1 OK |
| 100    | 100          | 100       | C1 OK |
| 500    | ~130–150     | ~135–160  | 隨 seed 變化；voxel 互斥導致飽和 |

Voxel 互斥下的包裝約在 **R ≈ 140–160** 飽和 — 再多就放不下了。每次換 seed 會探索到不同的 local optimum，N=500 重複跑會得到不同 K/R 與不同 neuron 組合，這正是產多組訓練資料的預期行為。

## 演算法草圖

- **Phase 1 — `synthesize_brain/index.py`**：掃每一個 `.am`、記錄 lattice 尺寸、header bbox、**非零 voxel 的 tight bbox**、非零 voxel 數。Canvas = 全部 header bbox 聯集、round 到整數 voxel。
- **Phase 2 — `synthesize_brain/select.py`**：
  1. 對全部 dataset 建 canvas-voxel coverage counter；對每顆 neuron 用其 tight bbox 內的平均 coverage 當分數（中心 = 高）。
  2. Greedy 依分數由高到低試裝（tie-break：小顆優先），撞到 `occupied_mask` 就跳過。I/O 平行跑 8 個 thread + 有限 prefetch window；被拒絕的讀取保留在共享 in-memory cache、repair 階段不用重解壓。
  3. 用增量維護的 per-voxel bbox counter 驗 C1；剔除 violator、從剩餘池按分數補、最多 5 輪。
- **Phase 3 — `synthesize_brain/compose.py`**：把每顆入選 neuron 的非零 voxel 貼進 `intensity`、對應位置寫 label ID 進 `labels`。
- **Phase 4 — writers**：`amira_io.py`（raw AmiraMesh ushort）、`nifti_io.py`（`.nii.gz` via nibabel）、`mip.py`（三軸 MIP PNG）、`contacts.py`（兩兩 F/E/V 接觸 voxel 統計）、`scan_video.py`（逐 label 掃描 MP4）、`noise.py`（Gaussian 雜訊）。

## 專案結構

```
synthesize.py                  # CLI 入口
synthesize_brain/
├── amira_io.py                # AmiraMesh reader（從 Kaleido 移植）+ ushort / label writer
├── compose.py                 # Phase 3
├── contacts.py                # 成對 F/E/V 接觸統計
├── index.py                   # Phase 1
├── mip.py                     # MIP 預覽
├── nifti_io.py                # NIfTI 寫檔
├── noise.py                   # Gaussian 雜訊
├── scan_video.py              # 逐 label 掃描 MP4
└── select.py                  # Phase 2
colormaps/bandpass_white.am    # Avizo 用的 bandpass colormap
check_saturation.py            # 診斷：最後的 K 是不是真的 voxel 飽和？
analyze_overlap.py             # 多個 run 之間的 neuron 重疊度分析
cache/                         # warp_index.npz, candidate_scores.npz
output/                        # output_N{req}_K{ach}_R{total}_s{seed}/, sweep_summary.tsv
tasks/                         # todo.md, lessons.md
docs/                          # PPTX 簡報、tutorial、build 腳本
```
