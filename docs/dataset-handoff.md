# SynthesizeBrain Dataset — Handoff Spec for Auto-Segmentation Project

這份文件給**接手做 single-neuron auto-segmentation 的下游專案 agent** 讀。它的目的是讓你**不必看 SynthesizeBrain 的 source code 也能正確消費資料**。

來源 repo：`https://github.com/ctshih/SynthesizeBrain`（私有）
本機路徑：`C:\Users\USER\Work\SynthesizeBrain`
資料產出路徑：`<repo>/output/output_N{req}_K{c1}_R{total}_s{seed}/`

---

## 1. 你拿到的是什麼

每個 run 是一組**配對的 3D 體積**：

| 欄位 | 角色 | dtype | 形狀 |
|---|---|---|---|
| `intensity.am` / `.nii.gz` | **訓練輸入**（模擬顯微鏡影像）| uint16 | (Z=337, Y=646, X=989) |
| `labels.am` / `.nii.gz` | **訓練 ground truth**（per-voxel instance label）| uint16 | (Z=337, Y=646, X=989) |

兩者**逐 voxel 對齊**。`intensity` 是模擬的密集染色腦影像（含 Gaussian 雜訊），`labels` 是乾淨的 instance segmentation 真值（值 = neuron ID，0 = 背景）。

每個 run 還附四個輔助檔（見第 4 節）。

---

## 2. Quick Start（5 行載入）

```python
import nibabel as nib
import csv

# Pick any one run dir.
run_dir = "C:/Users/USER/Work/SynthesizeBrain/output/output_N500_K150_R156_s91136998"

# Inputs
intensity = nib.load(f"{run_dir}/intensity.nii.gz").get_fdata().astype("uint16")  # (X, Y, Z)
labels    = nib.load(f"{run_dir}/labels.nii.gz").get_fdata().astype("uint16")     # (X, Y, Z)

# Per-label metadata
with open(f"{run_dir}/neuron_list.tsv") as f:
    meta = list(csv.DictReader(f, delimiter="\t"))   # one dict per label_id 1..K
```

`intensity.shape == labels.shape` 永遠成立。Voxel spacing 是 1.0、單位任意（FlyCircuit warp 空間，量綱不重要）。

> **關於 axis order**：nibabel 預設給 `(X, Y, Z)`（Fortran-style）。AmiraMesh 原生是 `(Z, Y, X)`（C-style）。兩者都是同一份資料，只是維度排列不同 — 任選一種、整個 pipeline 一致即可。

---

## 3. 目錄結構

每跑一次 `python synthesize.py synthesize` 產一個資料夾，命名格式：

```
output/output_N{requested}_K{c1_count}_R{total_count}_s{seed}/
```

例：`output_N500_K150_R156_s91136998/`

| 欄位 | 含義 |
|---|---|
| `N=500` | CLI 請求的 neuron 數量 |
| `K=150` | C1 保障下的數量（greedy + repair 生還者，所有 C1 都滿足）|
| `R=156` | 總 instance 數量（K + expand 階段加碼，可能違反 C1）|
| `s=91136998` | 隨機 seed（可重現）|

R - K = expand 階段加碼的 neuron 數量。這些 neuron 在 `neuron_list.tsv` 的 `phase` 欄會標 `expand`，且 `bbox_coverage < 0.5`。

---

## 4. 檔案逐項

### 4.1 `intensity.am` + `intensity.nii.gz`

- **uint16**, 形狀 (989, 646, 337) 或 (337, 646, 989)（看你用 nibabel 還是 amiramesh 讀）
- Voxel 值 = 原始 FlyCircuit warp 強度 + Gaussian 雜訊
- **背景 ≠ 0**！加入了 `baseline + N(0, σ)` 雜訊，預設 baseline=100、σ=50 → 背景 voxel 平均約 100
- Neuron 內部 voxel 約 100 ~ 4000+

**重點**：**不要**用 `intensity > threshold` 判斷 neuron 位置。背景隨時可能突破 200。**一律用 `labels` 當真值**。

兩個檔案是**完全相同**的資料，只是格式不同：
- `.am` — Amira/Avizo 原生 binary（給 Avizo 視覺化用）
- `.nii.gz` — NIfTI 標準（給 Python ML pipeline 用，nibabel 直接讀）

### 4.2 `labels.am` + `labels.nii.gz`

- **uint16**（單一 run 的 K 一定 < 65535，但形式上是 uint16）
- 值的範圍：`0..K`
  - `0` = 背景（沒有 neuron）
  - `1..K` = instance ID
- **Voxel 不重疊保證**：每個 voxel 最多屬於一個 instance（packing 限制 C2 強制過）
- Avizo 看會自動辨識成 label field（含 `Materials { ... }` block 與 per-instance 顏色）

```python
import numpy as np
labels = nib.load(f"{run_dir}/labels.nii.gz").get_fdata().astype("uint16")
print(labels.max())                          # K, e.g. 156
print(np.unique(labels))                     # array([0, 1, 2, ..., K])
print((labels > 0).sum())                    # total non-zero voxel count
print((labels == 47).sum())                  # voxels of instance 47
```

### 4.3 `neuron_list.tsv`

每個 label ID 對應到一行的 metadata。Tab-separated 14 欄：

| 欄位 | 型別 | 說明 |
|---|---|---|
| `label_id` | int | 1..K，與 `labels.*` 的 voxel 值對應 |
| `filename` | str | 原始 FlyCircuit `.am` 檔名（用來 trace 來源）|
| `driver` | str | GAL4 driver（`Tdc2` / `Trh` / `VGlut` / `fru`）|
| `phase` | str | `greedy` / `repair` / `expand` — 決定 C1 是否保障 |
| `voxel_count` | int | 該 instance 的非零 voxel 數量 |
| `bbox_coverage` | float | C1 覆蓋率 [0, 1]，phase=expand 可能 < 0.5 |
| `origin_ix/iy/iz` | int | 該 instance 的 bounding-box 起點（canvas voxel 座標）|
| `lattice_nx/ny/nz` | int | 原始 warp 檔的 lattice 尺寸 |

```python
import csv
with open(f"{run_dir}/neuron_list.tsv") as f:
    meta = {int(row["label_id"]): row for row in csv.DictReader(f, delimiter="\t")}
print(meta[47]["driver"], meta[47]["voxel_count"])
```

### 4.4 `contacts.csv`

兩兩 instance 之間的**接觸 voxel 統計**。可以用來訓練 boundary detection / instance separation 之類的 auxiliary task。

| 欄位 | 含義 |
|---|---|
| `neuron1`, `neuron2` | filename 對 |
| `N_F` | **face contacts**（共用一個 voxel 面）|
| `N_E` | **edge contacts**（共用邊但非面）|
| `N_V` | **vertex contacts**（共用頂點但非邊）|

只列出**有接觸的對**（其餘都是 0）。Pair 採 canonical ordering，不重複。

### 4.5 `mip.png` 與 `scan_video.mp4`

純診斷用，**不是訓練資料**：
- `mip.png` — 三軸 MIP 預覽，2 排（intensity / labels）× 3 軸（axial / coronal / sagittal）
- `scan_video.mp4` — 逐 instance 掃描影片，新 neuron 亮白、舊的暗灰

可以拿來 sanity check 載入是否正確（你的 MIP 應該長得跟附的 PNG 一樣）。

---

## 5. Canvas / Coordinate System

| 屬性 | 值 |
|---|---|
| Canvas dims (X, Y, Z) | 989 × 646 × 337 |
| Voxel spacing | 1.0 × 1.0 × 1.0（無單位）|
| Physical bbox | x: -482.5..505.8, y: -403.3..241.8, z: -192.9..143.3 |
| Total voxels | ~215 M |
| Typical occupied (R≈150) | ~2 M voxels （≈1% of canvas）|

**Canvas 對所有 run 都一樣** — 不同 seed 只改 instance 選擇，不改 canvas 尺寸或座標。

---

## 6. 關鍵 Conventions / 陷阱

### Conventions（必須遵守）

1. **Ground truth 一律看 `labels`、不看 `intensity`**。intensity 背景非零（有雜訊），用 thresholding 會出錯。
2. **Instance ID 對應檢查**：`labels[..] == n` 對應 `neuron_list.tsv` 裡 `label_id == n` 那行。
3. **Voxel 不重疊**：兩個 instance 不會共用 voxel，但**會接觸**（共面/共邊/共頂點）。Boundary 任務的訓練樣本就在這些 voxel pairs。

### 陷阱（常見錯誤）

- ❌ `intensity > 0` 當 mask → 半個 canvas 都是「neuron」
- ❌ 假設所有 instance 都滿足 C1 → `phase=expand` 那些不滿足
- ❌ 假設不同 run 共用 instance ID → ID 在每個 run 是獨立 1..K，跨 run 不對應
- ❌ 假設 instance n 的 bounding box 等於原始 warp 檔的 lattice → `lattice_*` 是原檔尺寸，實際在 canvas 中的位置要看 `origin_*` 和**實際 voxel 分布**
- ✅ 要算 instance n 的真正 bbox：`np.where(labels == n)`，再取 min/max

### Multi-run 的正確用法

每個 `output_N{...}/` 資料夾是**一個獨立的訓練樣本**。Instance ID 在不同 run 之間**不對應**：

- run A 的 label 47 ≠ run B 的 label 47
- 用 `neuron_list.tsv` 的 `filename` 欄判斷跨 run 是否同一顆原始 neuron

---

## 7. Multi-run 多樣性

實測（11 runs × N=500）：

| 量 | 值 |
|---|---|
| Unique neurons across all runs | 1079 |
| Total instances（11 × 平均 R） | ~1626 |
| Pairwise Jaccard 平均 | 0.06 |
| Neurons 只出現在單一 run | 70 % |

**對訓練意義**：多個 run 之間幾乎獨立，不太會 overfitting 到「永遠出現的那幾顆」。建議用法：

- **每個 epoch 隨機抽幾個 run** 當 mini-batch source（簡單做法：每個 run 視為一張獨立的 training image）
- **要更多多樣性** → 跑更多 seeds（每個 ~70 秒，sequential）
- **train/val/test split** → 不同 run 分配到不同 split 即可（沒有 sample 重疊問題，因為 instance ID 跨 run 不對應）

---

## 8. Auxiliary signal: contacts.csv

很多 instance segmentation 模型（StarDist、Cellpose、PatchPerPix）需要**邊界訊號**。`contacts.csv` 提供了現成的：

- 對於有接觸的 (n1, n2) 對，可以從 `labels` 抽出 contact voxels 當 boundary positives：
  ```python
  # voxels of n1 that are face-adjacent to n2 along +z
  v1 = (labels == n1)
  v2 = (labels == n2)
  v2_shifted = np.roll(v2, 1, axis=0)
  boundary_voxels = v1 & v2_shifted
  ```
- 或單純用 `contacts.csv` 的計數當 graph regularizer

如果不需要這個訊號，**忽略整個檔案**也沒關係，主任務只用 `intensity` + `labels`。

---

## 9. 重新產生資料 / 改設定

如果這份資料集需要更新（例如想關掉 noise、改 N、加新 driver、用更小的 σ）：

1. 進去 SynthesizeBrain repo：`cd C:\Users\USER\Work\SynthesizeBrain`
2. 看 `README.md` 的「Full parameter reference」表
3. 通常的調整：
   - 不要 noise → `--noise-sigma 0 --noise-baseline 0`
   - 大 noise → `--noise-sigma 150`
   - 小 N → `--n 50`
   - 重現某次 → `--seed <n>`
4. 跑 `python synthesize.py synthesize ...`

如果要產**新 driver / 新 dataset**：要動 `synthesize_brain/index.py`（重建 cache）+ 確認新檔的格式對得上 AmiraMesh BINARY-LITTLE-ENDIAN 2.1。

---

## 10. 與本專案的接觸面 / 求助順序

如果你（auto-seg agent）發現資料有問題：

1. **先讀 `output/output_*/neuron_list.tsv` 抽樣對照原始 warp 檔**：每行 filename 對應 `Kaleido\warp\<filename>` 的原始強度。可以對 a single instance 做 roundtrip — 比對 `intensity[labels == n] vs original_warp_voxels` 應該幾乎相同（差別僅來自 noise）。
2. **跑 `python check_saturation.py output/output_xxx/`** 驗證 voxel 不重疊保證。
3. **跑 `python analyze_overlap.py`** 看跨 run 多樣性。
4. **不要直接改 `output/...`** 裡的檔案。要改設定就重跑 `synthesize.py`（這份產出設計上是 reproducible，不是 mutable）。

如果有結構性問題（檔案格式錯、shape 不對、metadata 對不上）：在 SynthesizeBrain 那邊開 issue 或聯絡作者。

---

## 11. 一頁 Summary

```
資料位置: C:\Users\USER\Work\SynthesizeBrain\output\output_N{req}_K{c1}_R{total}_s{seed}/

訓練輸入: intensity.nii.gz   uint16   (X=989, Y=646, Z=337)   含 Gaussian 雜訊
訓練 GT:  labels.nii.gz     uint16   同形狀   值 0..K   0=背景，1..K=instance ID

K (under C1) ≈ 130-150 per run
R (total)    ≈ 135-160 per run
Voxel 互斥保證、Voxel 不重疊
Instance ID 在不同 run 之間不對應 (用 neuron_list.tsv 的 filename 跨 run 對齊)

最重要的提醒:
  * intensity 背景非零 (有 noise) — 不要用 thresholding
  * 一律用 labels 當 ground truth
  * phase=expand 的 instance 不保障 C1 (bbox_coverage < 0.5)
  * 不同 run 之間的 instance ID 不對應 — 用 filename 對齊
```
