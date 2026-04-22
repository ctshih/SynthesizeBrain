## Lessons

### 2026-04-21 DATA_FORMAT
**Mistake**: Assumed all warp `.am` files share one lattice size (386×345×182) after reading a single sample.
**Correction**: User pointed out the Kaleido composite brain is 958×601×328, which forced me to verify — different neurons have different lattice sizes, and the shared standard brain space is reconstructed by placing each cropped lattice at its own per-file origin.
**Rule**: Never generalize from one sample in a multi-file dataset. Verify by probing several files (different drivers, different filename ids), especially for header fields like lattice / bbox.

### 2026-04-22 SKILL_USAGE
**Mistake**: 被要求寫「Claude Code 初學者 tutorial」時，我自己手動寫，沒有先檢查有無專屬 skill。
**Correction**: 使用者說我應該用 `/claude-code-research-tutorial` 這個 skill。該 skill 專門產這類文件。
**Rule**: 接到「寫 tutorial / 教學 / 案例研究」類型的任務時，**先掃描 available-skills 清單**，有對應觸發詞的 skill 一律先 Skill() 觸發 — 不要自己從零寫。這條守則也適用於其他專業任務類別（.pptx、.docx、.xlsx 等都有 skill）。

### 2026-04-21 PROFILING
**Mistake**: Added parallel prefetch to the greedy-selection loop, saw greedy fill go 171s → 47s, assumed the optimization was done. Actually total select time stayed at ~220s because a SECOND serial loop — the repair refill — was re-reading the same 9000 candidates.
**Correction**: Self-corrected after bisecting with timing instrumentation. The refill re-reads matched the 170s missing time exactly.
**Rule**: When optimizing, add wall-clock timing around every non-trivial block in the function and verify the measured components sum to the total. If they don't, there's hidden work. Don't trust a speedup claim without whole-function accounting.
