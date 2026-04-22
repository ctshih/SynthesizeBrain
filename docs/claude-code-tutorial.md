# Claude Code 初學者指南：以 SynthesizeBrain 專案為例

這份文件記錄了一個真實研究專案（SynthesizeBrain）從 0 到完成的 Claude Code
協作過程，帶出新手最容易忽略的幾個技巧與陷阱。讀完你應該能帶著自己的研究
題目直接上手。

目標讀者：**有研究題目、會寫點 Python、剛開始用 Claude Code 的研究生或科學家**。

---

## 1. Claude Code 是什麼，適合做什麼

Claude Code 是一個在終端機裡跑的 AI 協作者。它能讀你的檔案、跑命令、改
code、發 git commit，跟你你來我往迭代。簡單說就是**一位專心的工程師
pair-programming 夥伴**，差別是你可以清晨三點 poke 他而他不會抱怨。

最適合的工作：

- 把研究裡**臨時性但複雜**的程式碼一次寫好（例如：「從這份 FlyCircuit
  dataset 合成訓練資料」）
- 整合陌生的**檔案格式**或**外部工具**（AmiraMesh 讀寫、Avizo 互動規則）
- **反覆實驗調參**（跑 10 個 seed、量 overlap）
- 寫研究產出的**輔助腳本**（生成影片、產 MIP、計算 contact statistics）

不太適合：

- 把整個深度學習模型的訓練流程外包給他（那是獨立工作流程，Claude Code
  比較適合 prototype 階段）
- 無法用文字精確描述的視覺化設計決策
- 需要大量 domain knowledge 而你沒有的領域 — 他會 hallucinate

---

## 2. 開始前的三個檔案

在你 project 跑 `claude` 之前，先建這三個檔案。它們比任何技巧都重要。

### 2.1 `~/.claude/CLAUDE.md`（全域規則）

放你對所有專案都適用的要求。範例（簡化版）：

```markdown
## 最高原則
1. 安全優先：高風險操作（刪除、覆寫）必須先確認
2. 簡單優先：最小改動、最小影響範圍
3. 品質優先：寧可慢，不要留下技術債

## 語言
總是用 Traditional Chinese 回覆。

## Plan 模式
非 trivial 任務（3+ 步驟）先進 plan mode、寫 tasks/todo.md、等我 OK 再動手。
```

### 2.2 `<project>/CLAUDE.md`（專案特定）

每個專案根目錄都應該有一份，寫專案特定的慣例：data 放哪、coding style、
要跑哪些 test。Claude 每次進這專案會先讀這份。

### 2.3 `<project>/tasks/todo.md` + `tasks/lessons.md`

- **todo.md**：當前計畫，checkable 項目。完成後寫 review 段落
- **lessons.md**：每次被你糾正後，Claude 應該把「教訓」寫在這裡，下次進
  來先讀一遍、避免重蹈覆轍

這兩個檔案是 Claude 的**工作記憶與長期記憶**之間的橋樑。

---

## 3. Case Study：SynthesizeBrain 從無到有

背景：把 9987 顆 FlyCircuit 果蠅神經元 warp volume 塞進共同腦空間，產出
「intensity + instance-label」配對資料供 segmentation 訓練。

### 3.1 破題：讓 Claude 問，別急著做

我丟出的第一個訊息只有大意：

> 任務是：從 warp 裡面選出 N 顆神經元，組成同一個影像...（大致規格）

Claude 第一件事不是開寫 code，而是**釐清五個細節**：

- 限制 1 的「50%」是體積還是計數？
- 輸出格式？
- 要混 driver 嗎？
- 可以重用別專案的 reader 嗎？
- 需要 GPU 嗎？

**結果**：我只花 5 分鐘回 5 條答案，避免了之後可能要砍掉重寫的 200 行。

> **新手守則 1**：規格不清楚時，逼 Claude 先問，不要讓他「用最佳猜測」開
> 寫。合格的問題都很短；爛猜測的 debug 都很長。

### 3.2 查證，不要假設

Claude 一開始看了一個 .am 樣本，**斷言**整份資料都是 386×345×182 lattice。
我隔天查了合成用的 Kaleido 腦空間是 958×601×328，立刻 push back。Claude
重新驗證、發現每顆 neuron 的 lattice 大小**彼此不同**（彼此被 tightly
crop），只有放到同一物理座標系才對齊。

```
Tdc2-F-000000: 386 × 345 × 182
Trh-F-000000:  213 × 134 ×  89
fru-F-900000: 118 ×  69 × 112  ← 完全不一樣
```

這個教訓被寫進 `tasks/lessons.md`：

> **Rule**: Never generalize from one sample in a multi-file dataset.

> **新手守則 2**：Claude 做出「所有 X 都是 Y」的斷言時，你要檢查。他看到
> 一個樣本就外推是常態。

### 3.3 切 commit，一個 commit 一個焦點

我們講好**分三個 commit**：
1. Skeleton + Phase 1 indexer
2. Phase 2 select + Phase 3 compose
3. Phase 4 writers + Phase 5 sweep

這樣做的好處：每個 commit message 都能清楚交代做了什麼、出錯可以精準
revert。完成後 `git log` 長這樣：

```
270496b Add scan_video.mp4 — one frame per label, newest white, seen dim
44c5548 Add Gaussian noise to intensity.am (labels.am untouched)
f6d7ea2 Noise: add baseline offset so background stays a proper Gaussian
bb38a89 Emit labels.am as a real Avizo label field; ship bandpass colormap
c39154f Fix expand pass: include dropped-violator candidates
f17ee3a Add random-seed selection, expand pass, pairwise F/E/V contact stats
4d3c5d8 Initial commit: SynthesizeBrain pipeline
```

每個 commit 都能獨立理解，未來查 bug 很容易。

> **新手守則 3**：Claude 一次改 5 個檔案要 commit，請他**拆 commit**。每個
> commit 訊息都要能告訴未來的你「為什麼改」，不只是「改了什麼」。

### 3.4 Debug 故事：看見的 vs 隱藏的時間

N=100 跑 5.7 分鐘太慢。我請 Claude 加 parallel prefetch，他做了，greedy
fill 從 171 秒降到 47 秒、歡呼說 3.7× 加速。

**但總時間還是 220 秒**，沒省到。

我逼他追查這「消失的 170 秒」在哪。他在 5 處加 `time.perf_counter()`、
發現 greedy 階段的 prefetch 有效，但**後面的 repair 階段會 serial 重讀
同樣的 9000+ 個候選**。加上 `read_cache` 字典共用兩階段的讀取結果 →
N=100 的 select 從 344 秒 降到 56 秒（真正的 6×）。

這次教訓被寫進 lessons.md：

> **Rule**: When optimizing, add wall-clock timing around every non-trivial
> block. If the pieces don't sum to the whole, there's hidden work.

> **新手守則 4**：Claude 宣稱的加速數字要**全函式端到端**測量才算數，只
> 測某個 sub-block 的加速會漏掉其他地方反增的時間。

### 3.5 Iterative feature 添加：慢慢長大

專案長出來的順序很重要，**每步產出可驗證的東西才往下走**：

1. 選樣算法 → 看 K/R 統計合理 ✓
2. 加 NIfTI 輸出 → 用 nibabel 對讀 roundtrip ✓
3. 加 MIP 預覽 → 肉眼看是果蠅腦 ✓
4. 加 random seed → 確認兩次 seed 出來不一樣 ✓
5. 加 expand pass → check_saturation 驗證真的飽和 ✓
6. 加 contacts.csv → N_F/N_E/N_V 比例合理 ✓
7. 加 scan_video.mp4 → 播起來順 ✓
8. 加 noise → 背景統計匹配 N(100, 50) ✓

**每次** Claude 都加 timing、sanity check，或肉眼 MIP。如果某步不 OK、
回去重做；整個後面不動。

> **新手守則 5**：每新增一個 feature，**同時**要求 Claude 產一個驗證
> 手段（數字、圖像、或 roundtrip）。沒有驗證手段的 feature = 沒有
> feature。

### 3.6 整合外部工具（Avizo）：從文件學，不是 Claude 猜

要讓輸出的 `labels.am` 在 Avizo 裡被認成 label field，Claude 一開始
不知道格式，我把 Avizo 8 文件 PDF 放進一個資料夾叫他讀。他用
`pdftotext` 抽文字、`grep` 找 label field 格式範例（`motor.labels.am`）、
按格式改 writer。一次過。

類似的 per-label inspection 規則也是實驗出來的：

```
[N, N+1]    → 顯示 labels N 和 N+1
[N, N]      → 什麼都沒有（Avizo 把零寬區間當空集合）
[N-0.5, N+0.5] → 只顯示 label N  ← 實測出來的正確公式
```

> **新手守則 6**：整合陌生工具前，把文件丟給 Claude。他比你更快找到關
> 鍵段落。實測規則要你跟他一起在 UI 裡試、互相修正 — 他沒有眼睛，你
> 有。

### 3.7 用跑批次驗證設計：10 × N=500 diversity

為了確保每次合成出來的訓練資料**彼此多樣**，我請 Claude 跑 10 個不同
seed 的 N=500 runs，然後分析神經元重疊度。

Claude 的處理：
1. 加 `--no-video` flag（避免批次跑 50 秒去產視訊 × 10 次）
2. 在 background 跑 shell loop（Monitor tool 通知每個 run 完成）
3. 事先把分析腳本 `analyze_overlap.py` 寫好（趁批次還在跑）
4. 批次結束自動跑分析、產報告

結果：11 runs 產出 1079 unique 神經元、平均 pairwise Jaccard 0.06、
70% 的 neuron 只出現在單一 run。多樣性足夠。

> **新手守則 7**：長時間批次要**背景跑** + Monitor 通知 + 事先準備
> 後續分析腳本。不要 fork shell loop 然後盯著它。

---

## 4. 核心技巧速覽

### TodoWrite — 顯示化工作流

Claude 會在長任務裡維護一個 todo list，你看得到他現在做哪一項、哪些
完成了、哪些還待辦。新手請求寫 feature 時，明確說「**拆成 todos**」。

### Memory / 記憶檔

`.claude/projects/<project>/memory/` 放一堆 `.md` 檔案，每次開新 session
自動讀。這專案累積了：

- `warp_data_reference.md` — 資料集規格
- `project_goals.md` — 任務與限制
- `project_n_ceiling.md` — K ≈ 150 天花板（實測結論）
- `feedback_terminology.md` — driver vs marker 糾正
- `feedback_vendored_deps.md` — 別跨專案 import
- `avizo_docs.md` — Avizo PDF 位置與相關章節
- `avizo_label_inspection.md` — `[N-0.5, N+0.5]` 規則

下次開 session，Claude 知道所有這些脈絡，不用再教一次。

### Background bash + Monitor

跑長任務（訓練、批次）時：

```python
# 用 run_in_background=true 丟到背景
# Monitor 事先裝個 tail-F | grep 監聽器，重要事件即時通知
```

你跟 Claude 的對話不被阻塞。新手常犯的錯是**等 subprocess 同步完成**
—— 15 分鐘對話凍結、也不能改其他東西。

### Subagents

複雜探索工作（「找出這 codebase 裡所有相關地方」）可以丟給 subagent，
它自己在獨立 context 裡跑、回報摘要給主 agent。不會污染你的主對話。

---

## 5. 陷阱清單

### 5.1 「我覺得應該沒問題」症候群

Claude 寫完一段 code，有時不跑就說「應該可以」。**不要信**。每次讓他
實測、印結果、跟你貼。本專案我們驗了無數次：

- read_amira roundtrip（writer 改完）
- labels 整數對應 voxel 數量（compose 後）
- 背景噪聲分布 mean/std（noise 加完）
- 10 runs 彼此 Jaccard（批次結束）

### 5.2 「看起來加速了」但其實沒

見 3.4。全函式端到端時間是唯一可靠指標。

### 5.3 auto mode 跑偏要及時踩煞車

Auto mode（自主執行）方便，但也意味著你要更積極檢查。本專案有一次
Claude 開始自動把 labels 從 uint16 改成 uint8（為了避開 Avizo 對話框），
我說「不，我還是用拉範圍的方法就好」，他立刻 revert。**看到 Claude 在
做你沒要求的事時要立刻停**。

### 5.4 避免跨專案 import

Claude 第一次寫 `from kaleido.amira_io import ...` 想直接用隔壁專案
的 reader。結果是隱形 dependency，git 上看不見。糾正後改成把檔案**複製
進本專案**。新手專案常有這問題，守住「git 看得到的才是 dependency」。

### 5.5 第一印象不要當事實

見 3.2。「我看了第一個 sample 是 X、所以全部都是 X」、「這條命令
returned 0 所以成功」、「tests 沒跑出錯所以過了」—— 都是待驗證的猜測，
不是結論。

---

## 6. Cheat Sheet

```
# 開新專案
claude          # 進互動模式
/init           # Claude 產 CLAUDE.md

# 對話技巧
「先說明你的計畫，別急著做」     ← 強制 plan mode
「拆 3 個 commit」               ← 拆 commit
「加 sanity check」              ← 驗證
「跑一次實測給我看」             ← 別空口說

# 專案結構
.
├── CLAUDE.md              # 專案規則
├── tasks/
│   ├── todo.md            # 當前計畫
│   └── lessons.md         # 累積教訓
└── <你的 code>

# 全域設定
~/.claude/CLAUDE.md        # 全域規則
~/.claude/projects/<proj>/memory/*.md   # 長期記憶

# 長任務
「在背景跑，跑完通知我」         ← run_in_background
「設 Monitor 每個 run 完成通知」 ← streaming events
```

---

## 附錄：推薦閱讀

- [Anthropic 官方 Claude Code 文件](https://docs.claude.com/claude-code)
- 本專案 `tasks/lessons.md` — 所有實際踩坑紀錄
- 本專案 `.claude/projects/.../memory/*.md` — 累積 domain memory

---

**最後一句**：Claude Code 最大的價值不是「寫得快」，是**把你從低階雜事
中解放出來、專注決策**。你負責判斷什麼是對的方向，他負責把方向變成
code。這份分工抓到了，你就真的能跑。
