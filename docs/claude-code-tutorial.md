# Claude Code 實戰教學：SynthesizeBrain 訓練資料合成案例

> 本文件是一份給 Claude Code 初學者的參考資料，透過一個真實專案（果蠅神經元 auto-segmentation 訓練資料合成器 **SynthesizeBrain**），示範如何與 Claude Code 協作完成一個中型研究型工程任務。
>
> **閱讀對象**：會寫點 Python、可能沒有用過 AI coding assistant 的研究生或科學家。神經科學 / 影像格式 / Avizo 背景不是前提 — 下面會補齊所有看得懂的條件。
>
> 專案結果：從 0 到能從 9,987 顆 FlyCircuit 果蠅神經元中挑選、合成、驗證**可用於 single-neuron auto-segmentation 模型訓練**的 3D 體積資料；並產出中文 lab meeting 簡報、中文教學（就是你正在讀的這份）。

---

## 目錄

> 想直接看怎麼實做，可以從 [Part 2 — 完整對話時序](#part-2--完整對話時序) 開始。

- [對話框顏色說明](#對話框顏色說明)
- [名詞速查表](#名詞速查表)
- [Claude Code 快速入門](#claude-code-快速入門)
- [Git 快速入門](#git-快速入門)
- [Part 1 — 給初學者的 10 個核心教訓](#part-1--給初學者的-10-個核心教訓)
- [Part 2 — 完整對話時序](#part-2--完整對話時序)
- [Part 3 — 最佳實踐清單](#part-3--從這場對話整理的-claude-code-最佳實踐清單)
- [Part 4 — 最終成果](#part-4--最終成果)

---

## 對話框顏色說明

本文件用五種顏色的對話框代表不同類型的使用者輸入：

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
<strong>🟢 請求動作</strong>：叫 Claude 去做某件事（寫程式、跑測試、git 操作…）
</div>

<div style="background:rgba(59,130,246,0.12);border-left:4px solid #2563eb;padding:10px 14px;margin:10px 0;border-radius:4px">
<strong>🔵 問問題</strong>：詢問現況、估算、技術細節
</div>

<div style="background:rgba(168,85,247,0.14);border-left:4px solid #9333ea;padding:10px 14px;margin:10px 0;border-radius:4px">
<strong>🟣 關鍵決策</strong>：在二選一或多選一節點做決定
</div>

<div style="background:rgba(245,158,11,0.14);border-left:4px solid #d97706;padding:10px 14px;margin:10px 0;border-radius:4px">
<strong>🟡 中斷 / 澄清</strong>：打斷 Claude、修正誤會
</div>

<div style="background:rgba(239,68,68,0.12);border-left:4px solid #dc2626;padding:10px 14px;margin:10px 0;border-radius:4px">
<strong>🔴 設計回饋 / 糾錯</strong>：指出方案的根本問題
</div>

---

## 名詞速查表

### 神經科學 / 資料面

| 名詞 | 意思 |
|---|---|
| **神經元 (neuron)** | 細胞本體加上糾纏的細纖維。在 3D 影像中就是一條條細長亮線組成的結構。 |
| **driver (GAL4 driver)** | 果蠅基因工具學用語。把某特定基因的啟動子（例 VGlut、Tdc2、Trh、fru）驅動螢光蛋白表現，就能只標記出帶那個基因的神經元。本專案資料用 4 個 driver：Tdc2(264)、Trh(994)、VGlut(6001)、fru(2728)。 |
| **warp volume** | 把不同個體果蠅腦**影像對齊 (warp)** 到同一張標準腦之後的 3D 體積影像檔。每顆 FlyCircuit 神經元都是這樣一份檔案。 |
| **FlyCircuit** | 公開的果蠅單神經元資料庫，有約 10,000 筆經過標準化的 warp volume。 |
| **auto-segmentation** | 給電腦看一張充滿神經元的 3D 影像，要它**自動圈出每一顆神經元的 voxel mask**。本專案是為了產生此任務的訓練資料。 |

### 影像與檔案格式

| 名詞 | 意思 |
|---|---|
| **voxel** | 3D 像素（volume pixel）。 |
| **lattice** | 在 AmiraMesh 檔案中裝 voxel 的 3D 陣列。每顆 warp volume 的 lattice 大小不同（因為各自被**緊緻裁切**）。 |
| **bounding box (bbox)** | 能完整包住某物的最小長方體。在本專案有兩個意思：檔案頭的 *lattice bbox*（裁切邊界）與 *tight bbox*（非零 voxel 的邊界）。 |
| **canvas** | 本專案合成輸出用的共同 3D 座標系，`989 × 646 × 337` voxels。9987 顆 warp 的 bbox 聯集得出。 |
| **AmiraMesh `.am`** | Avizo / Amira 軟體原生的 3D 影像格式。header 是 ASCII（lattice shape、bbox），data 段是 binary。 |
| **label field** | AmiraMesh 檔的一種特殊形式：`Lattice { ushort Labels } @1` + `Materials { ... }` 段。Avizo 讀到會認成 instance segmentation，渲染時用 nearest-neighbour、每個 id 一個顏色。與 **scalar field**（`ushort Data`）對比。 |
| **NIfTI `.nii.gz`** | neuroimaging 通用壓縮格式。比 `.am` 方便在 Python / deep learning pipeline 裡直接讀。本專案每次輸出都有 `.am` 和 `.nii.gz` 兩份。 |
| **MIP (Maximum Intensity Projection)** | 把 3D 影像沿某軸壓成 2D 的投影法：每條射線取最大值當 2D 像素。快速肉眼驗證 3D 內容的標準做法。 |

### 演算法與工具

| 名詞 | 意思 |
|---|---|
| **Greedy packing** | 每一步取「當下看來最好」的候選加入解集，不回溯。本專案用來一顆顆往 canvas 塞 neuron。 |
| **C1 / C2** | 本專案的兩條幾何限制。**C1**：bbox 覆蓋率 ≥ 50% — 每顆入選 neuron 的 bbox 至少要有一半被其他入選者的 bbox 聯集蓋住（防止空間分散到 segmentation 變簡單）；**C2**：voxel 互斥 — 任兩顆入選 neuron 的非零 voxel 不能重疊。 |
| **Jaccard similarity** | 兩集合交集大小 ÷ 聯集大小。0 = 完全不同、1 = 完全相同。本專案用來測兩組訓練樣本多樣性。 |
| **Gaussian noise N(μ, σ)** | 常態分布雜訊。σ 是**標準差**（非變異數）— 與 numpy `.normal(loc=μ, scale=σ)` 慣例一致。 |
| **pdftotext** | poppler 工具，把 PDF 抽成純文字。本專案拿來讓 Claude 讀 Avizo 8 文件（直接 `Read` PDF 效果不好）。 |

### Claude Code / AI 工具面

| 名詞 | 意思 |
|---|---|
| **Claude Code** | 在終端機跑的 AI coding 助手，可以讀檔、跑 shell、寫程式、commit。本文件記錄就是人類與它的一次完整協作。 |
| **context / context window** | Claude 單輪對話能記住的總文字量。長檔塞進去會爆 context。 |
| **Skill** | Claude Code 外掛專業技能，用特定觸發詞啟用。本專案用過 `anthropic-skills:pptx`（做簡報）與 `claude-code-research-tutorial`（做本文件）。 |
| **Subagent** | 分派給另一個 Claude 分身處理的子任務，**context 與主對話獨立**。讀大檔、做獨立 QA 常用。 |
| **Monitor** | 一個可以 tail + grep 背景 log 並把每行發通知的工具。本專案用來追蹤 10 個並行批次的進度。 |
| **memory files** | 放在 `~/.claude/projects/<proj>/memory/*.md`，每次進專案 Claude 會自動讀。專案累積的 domain 常識寫在這裡。 |
| **auto mode** | Claude Code 的自主執行模式：不等使用者確認、直接做。方便但要盯緊。 |

---

## Claude Code 快速入門

在看對話之前，先了解幾個 Claude Code 的工作方式：

**1. Claude 會自己動手。** 它不只是回答文字——會呼叫 `Read` / `Grep` / `Bash` / `Write` / `Edit` / `git` 等工具。你看不到每一步，但它會報告做了什麼。

**2. 你可以隨時打斷。** 按 `Esc` 或輸入 `[Request interrupted by user]`。本專案打斷了數次，每次都是 Claude 開始做你沒要求的事的時候。**早點打斷比讓它做完錯的再改便宜很多。**

**3. 背景任務 + 通知。** Claude Code 可以把長時間任務丟到背景跑，用 `Monitor` 監聽 log 或 `run_in_background` 收完成通知。使用者不用盯著、主對話不被阻塞。本案 10 × N=500 批次就是這樣跑的。

**4. Subagent 處理會吃 context 的任務。** 比如 QA、讀很長的檔案、獨立搜尋。Subagent 的 context 與主對話隔離、結果簡化後再回傳。

**5. 不可逆操作要先問。** 刪除、覆蓋、`rm -rf`、推公開 repo、執行交易都屬於這類。**Auto mode 也擋得住這些。**

**6. Claude 會記錄教訓。** 本專案有 `tasks/lessons.md` 跟 `~/.claude/projects/<proj>/memory/*.md`。每當使用者糾正 Claude，就寫一條規則進去，下次他進來先讀一遍、減少重複犯錯。

---

## Git 快速入門

本案不斷提到 `git commit`、`.gitignore`，這些屬於**版本控制**。沒用過 git 的話先看這段。

**什麼是 git？** 一個追蹤檔案變更歷史的工具，可以把它想像成**程式碼的時光機**：
- 每次改完一個段落，下 `git commit`，git 會把這次改動打包成一筆「**提交 (commit)**」
- 所有 commit 連成一條鏈，任何時候都能回到過去某個版本
- 避免「改壞了又不記得原本長怎樣」的慘劇

**什麼是 repository (repo)？** 一個被 git 管理的資料夾。有個隱藏 `.git/` 子目錄存歷史。

**什麼是 `.gitignore`？** 一個檔案，列出「**不要讓 git 追蹤**」的路徑。例：暫存檔、大型中間資料、密碼。本案的 `cache/`（索引）與 `output/`（幾百 MB 合成體積）都在 `.gitignore` 裡。

**最常用的指令**：

| 指令 | 意思 |
|---|---|
| `git init` | 初始化當前資料夾為 repo |
| `git status` | 看現在改了哪些檔 |
| `git add <檔>` | 把檔案加入下一次 commit |
| `git commit -m "訊息"` | 建立新 commit |
| `git log --oneline` | 看歷史 commit（精簡版） |
| `git diff` | 看尚未 commit 的差異 |

**commit 訊息長什麼樣**（本案的真實 commit）：

```
19eb742  Add docs/SynthesizeBrain.pptx — 11-slide lab-meeting deck
09e83ee  Add docs/claude-code-tutorial.md — Claude Code beginner guide
f6d7ea2  Noise: add baseline offset so background stays a proper Gaussian
44c5548  Add Gaussian noise to intensity.am (labels.am untouched)
270496b  Add scan_video.mp4 — one frame per label, newest white, seen dim
bb38a89  Emit labels.am as a real Avizo label field; ship bandpass colormap
c39154f  Fix expand pass: include dropped-violator candidates
f17ee3a  Add random-seed selection, expand pass, pairwise F/E/V contact stats
f5b0dbd  Rename output dir to reflect achieved K, not just requested N
4d3c5d8  Initial commit: SynthesizeBrain pipeline
```

每筆前面是 7 碼 hash，後面是人寫的說明。訊息短而具體，未來查 bug 很有用。

---

## Part 1 — 給初學者的 10 個核心教訓

從這場對話可以抽取出以下 Claude Code 使用模式，對任何專案都適用。

### 1. 先看環境再動手

第一句話就是「在這裡開新專案，會用到 `C:\Users\USER\Work\Kaleido\warp` 裡的資料」。Claude 先 `ls` 兩邊資料夾、看檔名規則、讀 Kaleido 專案的 README，建立 context 後才開始推估任務範圍。

**不要急著下指令** — 先讓助手摸清楚專案結構，後續規劃會準確很多。

### 2. 要計畫、不要只要結果

對中大型任務加一句 `先說明你的計畫，別急著做`。Claude 會把思路攤開成文字，你就能在「錯誤方向」被寫成 code 前攔截。

本案第一次計畫就是這樣。Claude 先列了 5 個待確認的規格問題（C1 怎麼定義、輸出格式、driver 混合、能否重用 Kaleido 的 reader、要不要 GPU），使用者逐一回答後才切 commit 動工。省掉「先寫了才發現方向錯」的白工。

### 3. 別從一個樣本外推

最慘的錯是**看一個樣本就斷言整份資料是某個樣子**。本案 Claude 一開始讀了一個 warp `.am` 檔，就說整份資料都是 `386 × 345 × 182` lattice。

<div style="background:rgba(239,68,68,0.12);border-left:4px solid #dc2626;padding:10px 14px;margin:10px 0;border-radius:4px">
🔴 <strong>使用者</strong>：Kaleido 合成出來的腦影像 bounding box 是 958x601x328，你從哪裡得到 386×345×182？
</div>

這迫使 Claude 查證幾份樣本，發現每顆 warp 檔 lattice 彼此不同（tight crop），只有放回共同物理座標才對齊。這教訓直接進了 `tasks/lessons.md`：

> **Rule**: Never generalize from one sample in a multi-file dataset.

### 4. 端到端 timing，別只測局部加速

改效能時 Claude 常只測改動那段的時間，忽略**全函式的端到端時間**。本案經典案例：

1. N=100 跑 5.7 分鐘太慢 → 加 parallel prefetch
2. Greedy fill 從 171 秒 → 47 秒（**3.7× 加速**），Claude 歡呼
3. **但全函式總時間還是 220 秒**！
4. 使用者逼 Claude 追「消失的 170 秒」
5. 加了 5 處 timing 才發現 — repair 階段的 refill loop 會 serial 重讀 9000+ 個候選
6. 加 `read_cache` 共享兩階段 I/O → 全函式變 56 秒（真正 6× 加速）

教訓進 lessons.md：

> **Rule**: When optimizing, add wall-clock timing around every non-trivial block. If the pieces don't sum to the whole, there's hidden work.

### 5. 實測校準外部工具的行為

整合 Avizo 時我們需要 `[min, max]` 的 bandpass colormap 來逐顆檢查 label。Claude 推論是 `[N-1, N+1]`（scalar field 的邏輯）。使用者實測：

<div style="background:rgba(239,68,68,0.12);border-left:4px solid #dc2626;padding:10px 14px;margin:10px 0;border-radius:4px">
🔴 <strong>使用者</strong>：我仔細檢查，跟你說的不一樣。colormap 上下限 50, 51 會顯示第 50 個⋯但 39, 40 卻會顯示兩個
</div>

經過三回合的假設 → 使用者實測 → 再假設，最後定調為 `[N-0.5, N+0.5]`（zero-width 被 Avizo 當空集合）。

**規則**：文件和 AI 推理不一定對，拿實際資料實測才算數。

### 6. 把外部文件餵給 Claude

你有 Avizo 8 的三份 PDF 文件、但 Claude 不能直接讀 PDF 排版？

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：我在 C:\Users\USER\Work\Avizo8_Docs 放了 Avizo 8 文件，讀一下
</div>

Claude 用 `pdftotext -layout` 把 PDF 抽成純文字，用 `grep` 定位相關章節（label field 格式、Icol 格式），照著改 writer。從不懂到產出可用的 label field，花不到 30 分鐘。

**規則**：Claude 不懂的領域，先把權威文件餵過去，他能比你更快找到關鍵段落。

### 7. 每個 feature 都附一個驗證手段

專案長大的順序是：indexing → selection → compose → MIP → video → noise → randomize → expand → contacts → label field → ...。**每一步**都要求 Claude 附驗證：

| 功能 | 驗證方法 |
|---|---|
| Indexer | 跑出 sanity stats（driver 分布、canvas 尺寸、nnz 分布） |
| Selection | C1 覆蓋率平均 / 最低 / violator 數 |
| Compose | `nz(intensity) == nz(labels)` assertion |
| MIP 輸出 | 肉眼看像不像果蠅腦 |
| NIfTI 輸出 | 自己讀回比對 shape / dtype / unique labels |
| Noise | 背景 corner mean / std 對照 theoretical |
| Label field | 載入 Avizo 看 Data info 是否認成 label |
| Expand 修 bug | `check_saturation.py` 對剩餘候選逐一試裝 |

沒有驗證手段的 feature = 沒有 feature。

### 8. 長任務背景跑 + Monitor + 事先準備後續分析

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：N=500 幫我跑 10 cases，跑完後回報 10 個 cases 共用 neuron 的個數狀況
</div>

Claude 的處理流程：

1. 加 `--no-video` flag 讓批次快些（~130s → ~90s/run）
2. 開 `run_in_background` 丟 shell loop 去跑 10 次
3. 設 Monitor：`tail -F <log> | grep "written to"` 每 run 完成發一個通知
4. **趁批次還在跑**，事先寫好分析腳本 `analyze_overlap.py`
5. 批次結束自動跑分析

15 分鐘跑完、Claude 全程沒阻塞、使用者隨時能插新指令。

### 9. Auto mode 要有停止能力

Auto mode 方便，但也可能讓 Claude 做你沒要求的事。本案例：

Claude 發現 Avizo Segmentation Editor 不吃 uint16 label，自己開始改 writer 把 labels 降成 uint8 以避開 dialog：

<div style="background:rgba(245,158,11,0.14);border-left:4px solid #d97706;padding:10px 14px;margin:10px 0;border-radius:4px">
🟡 <strong>使用者</strong>：不，我還是用拉範圍的方法就好
</div>

Claude 立刻 revert、不繼續。看到 Claude 做你沒要求的事要**立刻踩煞車**，越早越好。

### 10. 活用 Skill

專業任務交給專業工具：

- **做簡報** → `anthropic-skills:pptx`（內建 pptxgenjs 範本、字型建議、自動 QA 循環）
- **寫 Claude Code 教學** → `claude-code-research-tutorial`（就是產出本文件用的 skill）
- **做 Word 文件、PDF、Excel** → 各自有 `anthropic-skills:docx` / `pdf` / `xlsx`
- **大檔 QA** → 派 subagent 隔離 context

本文件就有個反例教訓：我第一次寫 tutorial 時**手刻**，使用者糾正「應該用 `/claude-code-research-tutorial`」，才用對工具重寫。**接到特定類別任務時先搜 skill 清單。**

---

## Part 2 — 完整對話時序

以下依時序列出關鍵使用者輸入與 Claude 回應。用彩色對話框標示 prompt 類型。

---

### ① 環境探索

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：我要在這裡開新專案，會用到 C:\Users\USER\Work\Kaleido\warp 裡的資料
</div>

**Claude 回應**

- 用 `Bash ls` 掃兩邊資料夾
- `SynthesizeBrain/` 是空資料夾（新專案）
- `Kaleido/warp/` 有 **9,987 個 `.am` 檔**、4 個 driver 前綴：Tdc2 (264)、Trh (994)、VGlut (6001)、fru (2728)
- 讀 Kaleido 的 README，知道那是另一個專案（配色視覺化）
- 用 `head -c 2000` 看一個 sample `.am` 檔的 ASCII header，解出 lattice shape、BoundingBox
- 因為不知道具體任務，**停下來問使用者要做什麼**

---

### ② 規格丟過來

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：任務是從 warp 裡選 N 顆神經元，在標準腦中組成同一個影像，voxel 值直接取 intensity，目的是做 single-neuron auto-segmentation 的訓練資料。除了合成影像外還要一個 label 欄位檔。
<br><br>
合成時採用的 neuron 必須滿足：<br>
1. 每一個 neuron 的 bounding box 與其他所有 neuron 的 bounding box 的 overlap 體積至少要佔 50%（總數、不重複計算）<br>
2. neuron 之間的 voxel 不能 overlap（可以靠在一起但不能重疊）<br>
3. 程式完成後，用不同大小的 N 測試<br>
先說明一下你的計畫，別急著做
</div>

**Claude 回應**

先給出 Phase 1-5 計畫草稿：

- **Phase 1**：建索引 — 掃 9987 檔，記 tight bbox + nnz count
- **Phase 2**：選 N 顆 — greedy + validation + repair
- **Phase 3**：合成 — 貼到 canvas
- **Phase 4**：輸出 — .am + 其他格式
- **Phase 5**：跑多 N 測試

**同時列 5 個待確認問題**：C1 體積還是計數？輸出格式？混 driver 嗎？能重用 Kaleido reader？要 GPU 嗎？

---

### ③ 🔴 第一次糾錯：canvas 尺寸

<div style="background:rgba(239,68,68,0.12);border-left:4px solid #dc2626;padding:10px 14px;margin:10px 0;border-radius:4px">
🔴 <strong>使用者</strong>：等一下，Kaleido 合成出來的腦影像 bounding box 是 958x601x328，你從哪裡得到 386×345×182？
</div>

**Claude 回應**

重新查證。發現：
- `Tdc2-F-000000` 是 386×345×182
- `Trh-F-000000` 是 213×134×89
- `fru-F-900000` 是 118×69×112
- **每顆 warp 檔都是 tightly-cropped，lattice 大小彼此不同**
- BoundingBox 才是共同物理座標系的定位
- Kaleido 的 canvas `958×601×328` 是 9987 顆 bbox 聯集

更新計畫：canvas 動態從索引算出來（不硬編）。更新 memory 檔避免下次再錯。

---

### ④ 規格確認五問

<div style="background:rgba(168,85,247,0.14);border-left:4px solid #9333ea;padding:10px 14px;margin:10px 0;border-radius:4px">
🟣 <strong>使用者</strong>：1. 「自己 bbox 體積的 ≥ 50% 被其他 bbox 聯集覆蓋」是的<br>
2. .am 跟 .nii.gz 都輸出，am 習慣用<br>
3. 那是 driver 不是 marker，全部混<br>
4. 可以重用 Kaleido reader，但複製到專案來<br>
5. 先用 CPU，過程中留意需不需要改 GPU
</div>

**Claude 回應**

把每條都寫進 memory：

- `driver_vs_marker.md` — driver 術語糾錯
- `vendored_deps.md` — 「複製、不跨專案 import」
- `project_goals.md` — 任務與限制
- `project_n_ceiling.md` — 測試上限 N=500

定稿計畫：3 個 commit 切法（skeleton + Phase 1 / Phase 2+3 / Phase 4+5）。

---

### ⑤ Commit 1：skeleton + Phase 1 indexer

<div style="background:rgba(168,85,247,0.14);border-left:4px solid #9333ea;padding:10px 14px;margin:10px 0;border-radius:4px">
🟣 <strong>使用者</strong>：好
</div>

**Claude 回應**

- 建專案骨架：`synthesize_brain/` 模組、`cache/`、`output/`、`tasks/`
- 從 Kaleido 複製 `amira_io.py` 過來，加上新的 `write_ushort_amira`
- `index.py` 用 `multiprocessing.Pool` 平行掃 9987 檔（單檔算 tight bbox + nnz）
- **Smoke test**：`--limit 50` 確認能跑
- 全量：9987 檔、23 workers、**18.7 秒跑完**、cache 2 MB
- 輸出 sanity stats：canvas `989×646×337`（稍微比 Kaleido 大）、driver 分布正確、median nnz ≈ 25k、0 個 out-of-canvas、0 個 negative origin

commit `4d3c5d8` — `Initial commit: SynthesizeBrain pipeline`

---

### ⑥ Commit 2：select + compose

**Claude 回應**

- `select.py` — greedy 選樣 + repair rounds（coverage counter 增量維護）
- `compose.py` — 貼 intensity + labels
- CLI 接好
- **N=10 測試**：10/10 achieved、C1 mean 0.802、no violators、73 秒

---

### ⑦ 🔵 Debug 故事：170 秒消失在哪

N=100 第一次跑 344 秒太慢。Claude 加 `parallel prefetch`（ThreadPool 預讀）後 greedy fill 從 171s → 47s 歡呼「3.7× 加速」。

但是：

<div style="background:rgba(59,130,246,0.12);border-left:4px solid #2563eb;padding:10px 14px;margin:10px 0;border-radius:4px">
🔵 <strong>使用者</strong>：greedy fill 46.9s，但 select 總共 219.9s — 消失的 173s 在哪？
</div>

**Claude 回應**

在 5 處加 `time.perf_counter()`：
- `scores section in 0.0s`（cache hit）
- `greedy fill 46.9s`
- `initial counter 0.1s`
- `remaining_order 0.0s`
- `round 0 ratios 0.1s`
- `final ratios 0.1s`
- 各段總和 ≈ 47s，**還是差 170s 沒找到**

再加 timing 到 repair 內部 refill loop — bingo：

```
[select]   dropped 1, refilled 1, now 100/100  (refill tried 9350 in 171.4s)
```

**refill 是 serial 重讀同樣 9000+ 個候選**。greedy 的 parallel prefetch 沒傳進 refill。

修法：`read_cache` 共享兩階段的讀取結果。再測：select 從 220s → **56s**（真正 6× 加速）。

進 `lessons.md`：

> **Rule**: 全函式 timing 的各段要相加 ≈ 總時間，否則有 hidden work。

commit `f17ee3a` — `Add random-seed selection, expand pass, pairwise F/E/V contact stats`（這個 commit 同時收進隨機化 + expand + contacts，見後續）

---

### ⑧ 🔵 N=500 意外：只能裝 148 顆

<div style="background:rgba(59,130,246,0.12);border-left:4px solid #2563eb;padding:10px 14px;margin:10px 0;border-radius:4px">
🔵 <strong>使用者</strong>：你前面那個 N500_K148 的 case，是有掃完整個 9987 筆資料，確定無法再塞進去任何一個了嗎？
</div>

**Claude 回應**

寫 `check_saturation.py`：載入 K=148 的結果，逐一對剩下 9839 個候選試 voxel overlap。

結果：**還有 6 顆可以塞而沒被試到**。原因：expand 階段漏了**被 drop 的 violators**（那些 neuron 既不在 selected 也不在 remaining_order 裡）。

修法：expand 前重算 `expand_order = [i for i in order if i not in selected]`，讓 dropped violators 也有機會。

commit `c39154f` — `Fix expand pass: include dropped-violator candidates`

---

### ⑨ 三個新需求：隨機、expand、contacts

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：<br>
1. 我要生成許多不同資料集來訓練，所以希望每次選到的要不一樣<br>
2. N 都設 500，K 能到哪就到哪；上限到了就不管 C1，只要不 overlap 就塞，越多越好<br>
3. 產一個 CSV：兩兩 neuron 之間三種接觸（面 F、邊 E、頂點 V）的 voxel 數
</div>

**Claude 回應**

- **隨機化**：在 selection 的 candidate score 加 Gaussian noise `σ = 0.25 × scores.std()`。每次 run 抽隨機 seed（印出來並寫進輸出目錄名），同 seed 重現、不同 seed 不同
- **expand**：3-phase 演算法的第三階段。前兩 phase 保證 C1；expand 放寬 C1 只看 C2。phase 欄位寫進 `neuron_list.tsv`
- **contacts**：沿 26 個 offset 把 label volume shift 一次，取對齊後非零且不等的 label pair 做 histogram。整個 `989×646×337` 在 ~8 秒內跑完。寫成 `contacts.csv`，欄位 `neuron1, neuron2, N_F, N_E, N_V`

---

### ⑩ 🟢 更精確的目錄命名

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：用 N=500，輸出目錄名 output_NXXX_KXXX_RXXX_sXXX，R 代表最後完成的數量（應該 >= K）
</div>

**Claude 回應**

拆 K 跟 R 的語意：
- **K** = under C1（greedy + repair 生還者）
- **R** = 總產出 = K + expand bonus
- **seed** = 亂數種子（預設抽、也可 `--seed` 指定重現）

改所有相關程式碼 + `sweep_summary.tsv` 的欄位 + README 文件。commit `4811945`。

---

### ⑪ Avizo 8 文件整合

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：我在 C:\Users\USER\Work\Avizo8_Docs 放了 Avizo 8 文件，讀一下
</div>

**Claude 回應**

三份 PDF：CodeBook、UsersGuide、RefGuide（共 ~56 MB）。

- `Read` 工具直接讀 PDF 需要 `pdftoppm`，沒裝
- 改用 `pdftotext -layout` 抽純文字到 `C:\Users\USER\AppData\Local\Temp\avizo\`
- 用 `grep` 定位相關章節（colormap、Icol format、Materials、Segmentation Editor、TCL 腳本）
- 寫 memory 檔 `avizo_docs.md` 記錄檔案位置與各章節重點

後續問「如何增加自訂 colormap」、「bandpass colormap」、「label field 格式」都從這些抽出的文字查。

---

### ⑫ 🟢 製作 bandpass colormap

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：算了，你直接刻一個給我，重點是要可以拉那個上下限
</div>

**Claude 回應**

Avizo colormap 檔是 AmiraMesh ASCII 格式。手刻 256-entry `.am`：
- entry[0]、entry[255] alpha=0（out-of-range 透明）
- entry[1..254] alpha=1 白色
- 加 `ColormapEditorCache` 讓 GUI Editor 顯示乾淨的 knots

存進 `colormaps/bandpass_white.am`，使用者透過 `Options → Load Colormap` 載入。

---

### ⑬ 🔴 scalar vs label field 的 fade 問題

使用者試把 MinMax 設 [57, 58] 想看 label 57，**結果有朦朧淡入淡出的 ghosts**。

**Claude 回應**

因為我們的 `labels.am` 寫成 `ushort Data`，Avizo 把它當 scalar field → Volume Rendering 預設用 **trilinear interpolation** → 相鄰 label 57 / 58 界面出現插值值 57.5 落在我們的 bandpass 範圍內 → 顯示為半透明球殼。

修法：把 `labels.am` 改寫成**真正的 label field**：
- `Lattice { ushort Labels } @1`（而非 `ushort Data`）
- Parameters 加 `Materials { Exterior {...} Neuron_0001 {...} ... }` 段
- Avizo 認到後自動用 nearest-neighbour、per-label 顏色

改 writer，regen 後 Avizo 的 Properties 顯示 `Data info: label, 16-bit unsigned`。問題解決。

commit `bb38a89`。

---

### ⑭ 🟡 Auto mode 踩煞車

Avizo 8 的 Segmentation Editor 只吃 8-bit label field，開啟時會跳 dialog：「A lossless conversion from 16-bit to 8-bit is required.」

Claude 自主判斷：K 永遠 ≤ 255 不會丟資訊，**自行開始**改 writer 輸出 uint8 以避開 dialog...

<div style="background:rgba(245,158,11,0.14);border-left:4px solid #d97706;padding:10px 14px;margin:10px 0;border-radius:4px">
🟡 <strong>使用者</strong>：不，我還是用拉範圍的方法就好
</div>

**Claude 回應**

立刻 revert 改動（`git restore`）、放棄 uint8 轉換、繼續走 16-bit label field 的路線。使用者按 Yes 讓 Avizo 做 lossless 轉換即可。

**規則**：auto mode 不等於沒有剎車。使用者一句話就該停手。

---

### ⑮ 🔴 [N, N] 規則實測糾正

Claude 推論要看 label N 用 `MinMax [N-1, N+1]`（scalar field 邏輯）。使用者實測：

<div style="background:rgba(239,68,68,0.12);border-left:4px solid #dc2626;padding:10px 14px;margin:10px 0;border-radius:4px">
🔴 <strong>使用者</strong>：跟你說的不一樣。50, 51 會顯示第 50 個；39, 40 卻會顯示兩個；39, 39 一顆都沒有
</div>

**Claude 回應**

經過三回合假設：
1. 推論「左閉右開 `[A, B)`」— 和 [39, 40] 顯示兩顆不符
2. 推論「inclusive」— 但 [39, 39] 什麼都沒有
3. 意識到**原因是「zero-width 區間」被 Avizo 當空集合**（可能內部 `(value - min) / (max - min)` 分母為零）
4. 正確做法：**`[N - 0.5, N + 0.5]`**，寬度 1、只涵蓋整數 N 一個

使用者實測：「這樣對了」。進 memory `avizo_label_inspection.md`。

**規則**：推理和文件都不一定對，實測才算數。

---

### ⑯ Gaussian noise on intensity

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：在影像中適當增加 Gaussian noise，只加在 intensity.am，labels.am 不用
</div>

**Claude 回應**

寫 `noise.py`：`intensity + N(0, σ)` 後 clip 到 `[0, 65535]`。預設 σ=50。

使用者追問：intensity 不會有負值，怎麼處理？

Claude 解釋 clip-at-0 會讓背景變**折疊 Gaussian**（mean 被推到 0.4σ、分布偏斜）。列四個替代方案：
- A: clip at 0（目前）
- B: `abs(noise)`
- C: **baseline offset + Gaussian**（像真實 sCMOS camera）
- D: offset 再 clip

<div style="background:rgba(168,85,247,0.14);border-left:4px solid #9333ea;padding:10px 14px;margin:10px 0;border-radius:4px">
🟣 <strong>使用者</strong>：改成 C 好了
</div>

**Claude 回應**

改為 `intensity + baseline + N(0, σ)`, clip `[0, 65535]`。預設 baseline=100, σ=50。

驗證：背景 corner mean=97.84, std=50.17 — 正好 N(100, 50)。commit `f6d7ea2`。

---

### ⑰ scan_video.mp4 功能

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：做一個 mp4 影片，讓 neuron 依 label 一個一個出現，新的全白、之前亮度調暗，左上角字卡 N=&lt;第幾個&gt;
</div>

**Claude 回應**

寫 `scan_video.py`：

- 每顆 label 的 2D MIP footprint 事先算好（三軸各一）
- 三軸 MIP 並排 axial | coronal | sagittal
- 逐 frame：當前 label 白色 (255)、之前的暗灰 (80)
- 左上角用 Pillow 畫「N = i」黃字
- 用 `imageio` + 內建 `imageio-ffmpeg` 輸出 MP4

K=150 的 run 產 ~50 秒的 mp4、約 1 MB。commit `270496b`。

---

### ⑱ 🟢 10 × N=500 批次 + 多樣性分析

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：N=500 幫我跑 10 cases，跑完後回報 10 個 cases 共用 neuron 的個數狀況（共用越少越好）。目的是開發 auto-segmentation 演算法
</div>

**Claude 回應**

工作流：

1. 加 `--no-video` flag（批次不需要 mp4、省 ~50s/run）
2. shell loop 丟背景跑：`for i in 1..10; python synthesize.py synthesize --n 500 --no-video`
3. 開 Monitor：`tail -F <log> | grep "written to"` 每 run 完成發通知
4. **趁 15 分鐘批次還在跑，事先寫 `analyze_overlap.py`**
5. 批次結束 → 立刻跑分析

分析結果（加上一個之前的 run 共 11 runs）：

- Unique neurons 1079 / total placements 1626 → redundancy 1.51×
- Pairwise Jaccard 平均 0.059（任兩 run 只重疊 ~6%）
- 70% 的 unique neurons 只出現在 1 個 run
- 最頻繁的 neuron 也只在 7/11 runs 出現

結論：訓練資料多樣性**足夠**，11 runs 是好的起手組。

---

### ⑲ 🟡 忘記產 video

<div style="background:rgba(245,158,11,0.14);border-left:4px solid #d97706;padding:10px 14px;margin:10px 0;border-radius:4px">
🟡 <strong>使用者</strong>：剛剛跑的 10 case 你忘記做 movie 了，這個要設為 default 選項
</div>

**Claude 回應**

video 其實已經是 default（`--no-video` 是 opt-out flag）— 我當初批次跑自己加了 flag 圖快。

- 確認 `make_video=True` 在 `_run_one` 是預設
- 對 10 個沒 video 的目錄：shell loop + Monitor 跑 `python synthesize.py video --dir ...`
- ~8 分鐘補齊 10 個 mp4
- 記取教訓：**不要自行省略 feature，除非使用者明確說要快**

---

### ⑳ 做 PPT

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：做一個 PPT 說明這個專案做了什麼事（不用寫用 Claude Code 協做的部分）
</div>

**Claude 回應**

呼叫 `anthropic-skills:pptx` skill — 載入 `pptxgenjs.md` 與設計 best practice。

- 11 頁 16:9 deck
- 配色：深海軍 + 奶油 + 孔雀綠 + 珊瑚橘
- 每頁版型：左側細條、右上 section 標籤、右下頁碼
- 用實際 MIP 圖當 hero visual
- Build 流程：pptxgenjs → `soffice --headless` → PDF → `pymupdf` → JPG
- **QA 循環**：Read 每張 JPG、列問題、回去改 `build_pptx.js`、重跑
- 第一次 pass 抓到 4 個問題：標題被切兩行、chart title 擠標題、bullet spacing 鬆、MIP 超出底部
- 第二 pass 全部修好

commit `19eb742`。

---

### ㉑ 🟢 寫 tutorial（第一版手刻）

<div style="background:rgba(34,197,94,0.12);border-left:4px solid #16a34a;padding:10px 14px;margin:10px 0;border-radius:4px">
🟢 <strong>使用者</strong>：以這個專案為範例，寫說明文件給 Claude Code 初學者
</div>

**Claude 回應**

（錯誤示範）手刻一份 337 行的 `docs/claude-code-tutorial.md`，內容是實戰經驗的條列式整理。

commit `09e83ee`。

---

### ㉒ 🔴 糾正：要用 skill

<div style="background:rgba(239,68,68,0.12);border-left:4px solid #dc2626;padding:10px 14px;margin:10px 0;border-radius:4px">
🔴 <strong>使用者</strong>：你在寫 tutorial 的時候，要用 /claude-code-research-tutorial 這個 skill
</div>

**Claude 回應**

- 進 `lessons.md`：「接到 tutorial / 教學 / 案例研究任務時先掃 skills 清單，有對應觸發詞的一律先 Skill() — 不要手寫」
- 呼叫 `claude-code-research-tutorial` skill
- Skill 載入 Kaleido 案例的模板（4-part 結構、對話框顏色編碼、名詞速查表、10 核心教訓、37 段時序）
- 用這個模板**重寫**本文件，覆蓋手刻版本

（就是你現在讀的這份。）

---

## Part 3 — 從這場對話整理的 Claude Code 最佳實踐清單

| 實踐 | 對應對話階段 | 為什麼重要 |
|---|---|---|
| 先 `看一下這裡有什麼` | ① | 建立 context，避免瞎猜 |
| 要 `先說明你的計畫` | ② | 在寫 code 前攔截錯誤方向 |
| 實際用樣本驗證斷言 | ③ | 別從一個樣本外推（canvas 尺寸就是教訓） |
| Memory 檔持續累積 domain 常識 | ④ | 下次進來不用重教 |
| 拆多 commit、訊息寫清楚 | ⑤⑧⑬⑯⑰ | 每步可獨立 review + 可回退 |
| 每 feature 一個驗證手段 | 全程 | 沒驗證 = 沒 feature |
| 全函式 timing | ⑦ | 局部加速可能被全局慢掉吃掉 |
| 長任務背景跑 + Monitor | ⑱⑲ | 不阻塞對話、使用者不用盯 |
| 預先寫後續分析腳本 | ⑱ | 批次一結束就出結果 |
| 實測校準外部工具規則 | ⑮ | 推理和文件不一定對 |
| 把 PDF 文件餵給 Claude | ⑪ | Claude 找關鍵章節比你快 |
| 不可逆 / 出乎預期的操作要停下來 | ⑭ | Auto mode 要有剎車 |
| 用 Skill 處理專業任務 | ⑳㉒ | pptx、tutorial 都有專門 skill |
| 糾錯後寫進 lessons.md | ③㉒ | 下次不再犯 |

---

## Part 4 — 最終成果

- **程式碼**：`synthesize_brain/` 模組化 pipeline、CPU numpy 實作
- **效能**：
  - Indexer：18.7 秒 / 9987 檔（23 workers 平行）
  - Select + Compose：N=500 約 60 秒 / run（seed 隨機）
  - 全套輸出：~8 秒 I/O
- **輸出**（每次 run 一套）：
  - `intensity.am` / `intensity.nii.gz`（uint16 強度影像 + Gaussian 雜訊）
  - `labels.am` / `labels.nii.gz`（真 Avizo label field，per-label Materials）
  - `neuron_list.tsv`、`contacts.csv`、`mip.png`、`scan_video.mp4`
- **11 組隨機 N=500 runs**：1079 unique neurons，pairwise Jaccard 0.06，可作為 auto-seg 訓練集
- **文件**：
  - `README.md`（技術使用說明）
  - `docs/claude-code-tutorial.md`（本文）
  - `docs/SynthesizeBrain.pptx`（11 頁 lab meeting 簡報）
- **commit 數**：~13 個，訊息清晰可追溯

---

*本教學文件由 Claude Code 整理自 SynthesizeBrain 專案 2026-04 的完整對話紀錄，透過 `claude-code-research-tutorial` skill 以 Kaleido 案例為模板改寫而成。*
