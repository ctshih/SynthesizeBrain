// Build SynthesizeBrain.pptx — technical overview deck for lab meeting.
// Palette: Ocean Gradient + Coral accent (brain imaging vibe).
// Font mix: Microsoft JhengHei (Chinese + English) for broad compatibility.

const pptxgen = require("pptxgenjs");
const path = require("path");

const PROJECT = "C:/Users/USER/Work/SynthesizeBrain";
const MIP_HERO = path.join(PROJECT, "output/output_N500_K150_R156_s91136998/mip.png");
const MIP_N50 = path.join(PROJECT, "output/output_N50_K50_s1682055724/mip.png");
const OUTPUT_PATH = path.join(PROJECT, "docs/SynthesizeBrain.pptx");

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "SynthesizeBrain";
pres.title = "SynthesizeBrain: Paired Training Volumes for Single-Neuron Auto-Segmentation";

// ---- palette ----
const BG_DARK = "0A2D3D";     // deep teal-navy (title / section)
const BG_LIGHT = "F7F5F0";    // cream (content)
const PRIMARY = "0E7C7B";     // teal (accent bars, headings)
const ACCENT = "E76F51";      // coral (numbers, key stats)
const TEXT_DARK = "1D2D3D";   // near-black
const TEXT_LIGHT = "F7F5F0";  // cream
const MUTED = "6B8A95";       // desaturated teal (captions)
const GRID = "D5D0C0";        // warm grey (borders, dividers)

const HEAD_FONT = "Microsoft JhengHei";
const BODY_FONT = "Microsoft JhengHei";

// fresh shadow object — pptxgenjs mutates these, so we spawn per call
const shadow = () => ({ type: "outer", color: "000000", blur: 8, offset: 2, angle: 90, opacity: 0.12 });

// -------- helpers --------
function addSlideHeader(slide, label, pageNum, total) {
    // tiny section label top-right
    slide.addText(label.toUpperCase(), {
        x: 8.0, y: 0.25, w: 1.8, h: 0.3,
        fontSize: 9, fontFace: HEAD_FONT, color: MUTED,
        align: "right", charSpacing: 3, margin: 0,
    });
    // page number bottom-right
    slide.addText(`${pageNum} / ${total}`, {
        x: 9.0, y: 5.25, w: 0.9, h: 0.25,
        fontSize: 8, fontFace: BODY_FONT, color: MUTED, align: "right", margin: 0,
    });
    // left accent bar (full height, thin)
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0, y: 0, w: 0.1, h: 5.625,
        fill: { color: PRIMARY }, line: { color: PRIMARY, width: 0 },
    });
}

function addSlideTitle(slide, title) {
    slide.addText(title, {
        x: 0.5, y: 0.55, w: 8.5, h: 0.7,
        fontSize: 28, fontFace: HEAD_FONT, bold: true, color: TEXT_DARK, margin: 0,
    });
}

const TOTAL_SLIDES = 11;

// =========================================================================
// SLIDE 1 — Title
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_DARK };

    // Hero MIP moved to top-right corner, smaller — leaves the whole left half
    // for text plus an L-shaped text column down the side.
    slide.addImage({
        path: MIP_HERO,
        x: 5.7, y: 0.5, w: 4.0, h: 2.4,
        transparency: 30,
    });

    // title block, now wide enough for "SynthesizeBrain" at 44pt
    slide.addText("SynthesizeBrain", {
        x: 0.6, y: 1.5, w: 5.2, h: 0.9,
        fontSize: 44, fontFace: HEAD_FONT, bold: true, color: TEXT_LIGHT, margin: 0,
    });
    slide.addText("以 FlyCircuit warp volumes\n合成單神經元自動分割的訓練資料", {
        x: 0.6, y: 2.5, w: 5.2, h: 1.3,
        fontSize: 18, fontFace: BODY_FONT, color: ACCENT, margin: 0,
        paraSpaceAfter: 4,
    });
    // thin accent line
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.6, y: 3.95, w: 0.8, h: 0.05,
        fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
    });
    slide.addText("Paired intensity + instance-label volumes\nfor dense-neuron auto-segmentation training", {
        x: 0.6, y: 4.1, w: 7, h: 0.7,
        fontSize: 12, fontFace: BODY_FONT, color: TEXT_LIGHT, italic: true, margin: 0,
    });

    slide.addText("Lab Meeting · 2026", {
        x: 0.6, y: 4.8, w: 4, h: 0.3,
        fontSize: 10, fontFace: BODY_FONT, color: MUTED, margin: 0, charSpacing: 2,
    });
}

// =========================================================================
// SLIDE 2 — Motivation
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Motivation", 2, TOTAL_SLIDES);
    addSlideTitle(slide, "為什麼需要這個專案");

    // Problem statement
    slide.addText("密集染色腦影像的 single-neuron auto-segmentation 需要大量「每顆神經元都有 ground-truth 遮罩」的訓練資料 — 而真實標註代價太高。", {
        x: 0.5, y: 1.35, w: 9, h: 0.8,
        fontSize: 15, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0,
    });

    // Three cards
    const cardY = 2.5;
    const cardH = 2.3;
    const cards = [
        {
            title: "問題",
            body: "手動標註每顆神經元的 voxel mask 極度耗時；公開資料稀少且規模小。",
            color: ACCENT,
        },
        {
            title: "觀察",
            body: "FlyCircuit 有 ~10000 顆已 warp 到共同腦空間的單神經元影像，每顆都是乾淨的 ground truth。",
            color: PRIMARY,
        },
        {
            title: "解法",
            body: "從中挑 N 顆疊成密集配對資料：intensity 影像 + voxel-level label 圖，拿來訓練 auto-seg model。",
            color: "2A9D8F",
        },
    ];
    cards.forEach((c, i) => {
        const x = 0.5 + i * 3.1;
        // card
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y: cardY, w: 2.9, h: cardH,
            fill: { color: "FFFFFF" }, line: { color: GRID, width: 0.5 },
            shadow: shadow(),
        });
        // left accent bar on card
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y: cardY, w: 0.08, h: cardH,
            fill: { color: c.color }, line: { color: c.color, width: 0 },
        });
        // title
        slide.addText(c.title, {
            x: x + 0.25, y: cardY + 0.2, w: 2.6, h: 0.4,
            fontSize: 14, fontFace: HEAD_FONT, bold: true, color: c.color, charSpacing: 2, margin: 0,
        });
        // body
        slide.addText(c.body, {
            x: x + 0.25, y: cardY + 0.7, w: 2.5, h: cardH - 0.8,
            fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0,
        });
    });
}

// =========================================================================
// SLIDE 3 — Dataset
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Dataset", 3, TOTAL_SLIDES);
    addSlideTitle(slide, "Source dataset: FlyCircuit warp volumes");

    // big stat
    slide.addText("9,987", {
        x: 0.5, y: 1.6, w: 4, h: 1.5,
        fontSize: 88, fontFace: HEAD_FONT, bold: true, color: PRIMARY, margin: 0,
    });
    slide.addText("warped single-neuron\nvolumes (AmiraMesh)", {
        x: 0.5, y: 3.15, w: 4, h: 0.8,
        fontSize: 14, fontFace: BODY_FONT, color: TEXT_DARK, italic: true, margin: 0,
    });
    slide.addText("共同標準腦座標系 989 × 646 × 337 voxels",{
        x: 0.5, y: 3.9, w: 4, h: 0.4,
        fontSize: 11, fontFace: BODY_FONT, color: MUTED, margin: 0,
    });

    // Driver breakdown as right-side bar chart
    const drivers = [
        { name: "VGlut", count: 6001, color: "0E7C7B" },
        { name: "fru",   count: 2728, color: "E76F51" },
        { name: "Trh",   count: 994,  color: "2A9D8F" },
        { name: "Tdc2",  count: 264,  color: "F4A261" },
    ];
    const barX = 5.2, barY = 1.9, barW = 4.3;
    slide.addText("GAL4 driver 組成", {
        x: barX, y: 1.45, w: barW, h: 0.35,
        fontSize: 13, fontFace: HEAD_FONT, bold: true, color: TEXT_DARK, charSpacing: 2, margin: 0,
    });
    const maxCount = 6001;
    const rowH = 0.55;
    drivers.forEach((d, i) => {
        const y = barY + i * (rowH + 0.2);
        const w = barW * (d.count / maxCount);
        // label
        slide.addText(d.name, {
            x: barX, y, w: 0.8, h: rowH,
            fontSize: 13, fontFace: BODY_FONT, bold: true, color: TEXT_DARK, valign: "middle", margin: 0,
        });
        // bar
        slide.addShape(pres.shapes.RECTANGLE, {
            x: barX + 0.85, y: y + 0.1, w: Math.max(w - 0.85, 0.2), h: rowH - 0.2,
            fill: { color: d.color }, line: { color: d.color, width: 0 },
        });
        // count label
        slide.addText(d.count.toLocaleString(), {
            x: barX + 0.85 + Math.max(w - 0.85, 0.2) + 0.1, y: y + 0.05, w: 1.0, h: rowH - 0.1,
            fontSize: 12, fontFace: BODY_FONT, color: TEXT_DARK, valign: "middle", margin: 0,
        });
    });
}

// =========================================================================
// SLIDE 4 — Constraints C1 + C2
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Constraints", 4, TOTAL_SLIDES);
    addSlideTitle(slide, "兩條幾何限制決定可否共存");

    // Two panels C1 + C2
    const panel = (x, tag, name, formula, explain, color) => {
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y: 1.35, w: 4.3, h: 3.8,
            fill: { color: "FFFFFF" }, line: { color: GRID, width: 0.5 },
            shadow: shadow(),
        });
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y: 1.35, w: 0.08, h: 3.8,
            fill: { color }, line: { color, width: 0 },
        });
        // tag
        slide.addText(tag, {
            x: x + 0.3, y: 1.55, w: 0.8, h: 0.5,
            fontSize: 24, fontFace: HEAD_FONT, bold: true, color, margin: 0,
        });
        // name
        slide.addText(name, {
            x: x + 1.1, y: 1.6, w: 3, h: 0.4,
            fontSize: 14, fontFace: HEAD_FONT, bold: true, color: TEXT_DARK, margin: 0,
        });
        // formula card
        slide.addShape(pres.shapes.RECTANGLE, {
            x: x + 0.3, y: 2.3, w: 3.8, h: 1.0,
            fill: { color: BG_LIGHT }, line: { color: GRID, width: 0.3 },
        });
        slide.addText(formula, {
            x: x + 0.3, y: 2.3, w: 3.8, h: 1.0,
            fontSize: 13, fontFace: "Consolas", color: TEXT_DARK, align: "center", valign: "middle", margin: 0,
        });
        // explain
        slide.addText(explain, {
            x: x + 0.3, y: 3.5, w: 3.8, h: 1.5,
            fontSize: 11, fontFace: BODY_FONT, color: MUTED, margin: 0,
        });
    };
    panel(0.5, "C1", "bbox 覆蓋率 ≥ 50 %",
        "|bbox(n) ∩ ⋃ bbox(m)| / |bbox(n)| ≥ 0.5",
        "每顆 neuron 的 tight bbox 至少有一半體積被其他已選 neuron 的 bbox 聯集覆蓋 — 防止空間分離太散、失去 segmentation 難度。", PRIMARY);
    panel(5.2, "C2", "Voxel 不重疊",
        "∀ m ≠ n:  voxels(n) ∩ voxels(m) = ∅",
        "任兩顆 neuron 的非零 voxel 必須互斥 — 真實腦中不可能重疊，但 warp 自不同個體容易有像素撞在一起。", ACCENT);
}

// =========================================================================
// SLIDE 5 — Pipeline
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Pipeline", 5, TOTAL_SLIDES);
    addSlideTitle(slide, "Processing pipeline");

    const stages = [
        { tag: "1", name: "INDEX",    desc: "scan 9987 .am → bbox + nnz cache", detail: "單次執行 18 秒" },
        { tag: "2", name: "SELECT",   desc: "挑 N 顆滿足 C1+C2",                 detail: "~60 秒 / N=500" },
        { tag: "3", name: "COMPOSE",  desc: "貼到 989×646×337 canvas",           detail: "intensity + labels" },
        { tag: "4", name: "OUTPUTS",  desc: "am / nii.gz / MIP / video / csv",  detail: "~8 秒 I/O" },
    ];
    const boxW = 2.0, boxH = 2.0, gap = 0.35;
    const totalW = stages.length * boxW + (stages.length - 1) * gap;
    const startX = (10 - totalW) / 2;
    const y = 1.8;

    stages.forEach((s, i) => {
        const x = startX + i * (boxW + gap);
        // box
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y, w: boxW, h: boxH,
            fill: { color: "FFFFFF" }, line: { color: PRIMARY, width: 1.5 },
            shadow: shadow(),
        });
        // number circle
        slide.addShape(pres.shapes.OVAL, {
            x: x + boxW/2 - 0.3, y: y - 0.3, w: 0.6, h: 0.6,
            fill: { color: PRIMARY }, line: { color: PRIMARY, width: 0 },
        });
        slide.addText(s.tag, {
            x: x + boxW/2 - 0.3, y: y - 0.3, w: 0.6, h: 0.6,
            fontSize: 20, fontFace: HEAD_FONT, bold: true, color: TEXT_LIGHT, align: "center", valign: "middle", margin: 0,
        });
        // name
        slide.addText(s.name, {
            x, y: y + 0.5, w: boxW, h: 0.4,
            fontSize: 14, fontFace: HEAD_FONT, bold: true, color: PRIMARY, align: "center", charSpacing: 3, margin: 0,
        });
        // desc
        slide.addText(s.desc, {
            x: x + 0.15, y: y + 1.0, w: boxW - 0.3, h: 0.5,
            fontSize: 10, fontFace: BODY_FONT, color: TEXT_DARK, align: "center", margin: 0,
        });
        // detail caption
        slide.addText(s.detail, {
            x: x + 0.15, y: y + 1.5, w: boxW - 0.3, h: 0.4,
            fontSize: 9, fontFace: BODY_FONT, color: MUTED, italic: true, align: "center", margin: 0,
        });
        // arrow to next
        if (i < stages.length - 1) {
            const ax = x + boxW + 0.05;
            slide.addShape(pres.shapes.RIGHT_TRIANGLE, {
                x: ax, y: y + boxH/2 - 0.1, w: 0.22, h: 0.2,
                fill: { color: PRIMARY }, line: { color: PRIMARY, width: 0 }, rotate: 90,
            });
        }
    });

    // bottom caption
    slide.addText("所有輸出都在 output/output_N{req}_K{c1}_R{total}_s{seed}/ 同一個資料夾，方便成對使用。", {
        x: 0.5, y: 4.3, w: 9, h: 0.4,
        fontSize: 11, fontFace: BODY_FONT, color: MUTED, italic: true, align: "center", margin: 0,
    });
}

// =========================================================================
// SLIDE 6 — Selection algorithm (3-phase)
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Algorithm", 6, TOTAL_SLIDES);
    addSlideTitle(slide, "三階段選樣演算法");

    const phases = [
        {
            tag: "GREEDY",
            color: PRIMARY,
            sub: "依「bbox 密度分數」排序，依序試裝",
            bullets: [
                "建立每顆 neuron 的 tight-bbox 密度分數",
                "高分者優先嘗試加入",
                "只檢查 C2（voxel 不重疊）— C1 後續再管",
                "加入 seed-driven noise 讓每次結果不同",
            ],
        },
        {
            tag: "REPAIR",
            color: ACCENT,
            sub: "找 C1 違規者、丟掉並用其他 candidate 補回",
            bullets: [
                "計算每顆的 bbox 覆蓋率",
                "違規者（<50%）被 drop",
                "從剩餘 pool 補進不違規的",
                "最多迭代 5 輪或收斂為止",
            ],
        },
        {
            tag: "EXPAND",
            color: "2A9D8F",
            sub: "塞剩下任何能容納的 neuron（放寬 C1）",
            bullets: [
                "只看 C2 是否通過",
                "允許 C1 違規 — 這些額外顆數算 bonus",
                "將 K (under-C1) 升級到 R (rendered total)",
                "R − K = 多得的訓練密度",
            ],
        },
    ];

    const pw = 3.0, ph = 3.6, py = 1.4, pgap = 0.3;
    const totalW = phases.length * pw + (phases.length - 1) * pgap;
    const startX = (10 - totalW) / 2;

    phases.forEach((p, i) => {
        const x = startX + i * (pw + pgap);
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y: py, w: pw, h: ph,
            fill: { color: "FFFFFF" }, line: { color: GRID, width: 0.5 },
            shadow: shadow(),
        });
        // top banner
        slide.addShape(pres.shapes.RECTANGLE, {
            x, y: py, w: pw, h: 0.7,
            fill: { color: p.color }, line: { color: p.color, width: 0 },
        });
        slide.addText(`${i + 1}. ${p.tag}`, {
            x, y: py, w: pw, h: 0.7,
            fontSize: 18, fontFace: HEAD_FONT, bold: true, color: TEXT_LIGHT,
            align: "center", valign: "middle", charSpacing: 4, margin: 0,
        });
        // subtitle
        slide.addText(p.sub, {
            x: x + 0.2, y: py + 0.8, w: pw - 0.4, h: 0.55,
            fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, italic: true, valign: "top", margin: 0,
        });
        // bullets — valign top pulls them up against the subtitle
        slide.addText(
            p.bullets.map((b, j) => ({ text: b, options: { bullet: { code: "25AA" }, breakLine: j < p.bullets.length - 1 } })),
            { x: x + 0.2, y: py + 1.35, w: pw - 0.4, h: ph - 1.45, fontSize: 10.5, fontFace: BODY_FONT, color: TEXT_DARK, valign: "top", paraSpaceAfter: 3, margin: 0 }
        );
    });
}

// =========================================================================
// SLIDE 7 — Empirical findings
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Findings", 7, TOTAL_SLIDES);
    addSlideTitle(slide, "經驗性發現");

    // big stats on top
    const stats = [
        { big: "~150", small: "packing ceiling K", sub: "voxel 互斥下的物理上限" },
        { big: "0.059", small: "mean pairwise Jaccard", sub: "11 runs 之間的重疊度" },
        { big: "70 %", small: "neurons used only once", sub: "出現在唯一 run" },
    ];
    stats.forEach((s, i) => {
        const x = 0.5 + i * 3.15;
        slide.addText(s.big, {
            x, y: 1.3, w: 3, h: 1.0,
            fontSize: 56, fontFace: HEAD_FONT, bold: true, color: ACCENT, align: "center", margin: 0,
        });
        slide.addText(s.small, {
            x, y: 2.35, w: 3, h: 0.3,
            fontSize: 12, fontFace: HEAD_FONT, bold: true, color: TEXT_DARK, align: "center", charSpacing: 2, margin: 0,
        });
        slide.addText(s.sub, {
            x, y: 2.65, w: 3, h: 0.35,
            fontSize: 10, fontFace: BODY_FONT, color: MUTED, italic: true, align: "center", margin: 0,
        });
    });

    // bottom: interpretation
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y: 3.4, w: 9, h: 1.6,
        fill: { color: "FFFFFF" }, line: { color: GRID, width: 0.5 },
        shadow: shadow(),
    });
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y: 3.4, w: 0.08, h: 1.6,
        fill: { color: PRIMARY }, line: { color: PRIMARY, width: 0 },
    });
    slide.addText("Interpretation", {
        x: 0.75, y: 3.5, w: 3, h: 0.35,
        fontSize: 12, fontFace: HEAD_FONT, bold: true, color: PRIMARY, charSpacing: 3, margin: 0,
    });
    slide.addText([
        { text: "飽和原因：", options: { bold: true } },
        { text: "9987 顆總 nnz voxels ≈ 250 M，canvas 只有 215 M voxels，密集區平均每 voxel 要被 1.16+ 顆神經元占用。voxel 互斥下只能撐 ~150 顆。", options: { breakLine: true } },
        { text: "多樣性夠：", options: { bold: true } },
        { text: "任兩組訓練樣本平均僅共享 ~6% neurons，沒有 neuron 跨 7 runs 以上出現 → 無「過擬合熱點」。", options: {} },
    ], {
        x: 0.75, y: 3.9, w: 8.7, h: 1.05,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, margin: 0, paraSpaceAfter: 3,
    });
}

// =========================================================================
// SLIDE 8 — MIP hero
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Result", 8, TOTAL_SLIDES);
    addSlideTitle(slide, "合成結果：N = 500、R = 156 顆");

    // Big MIP image — constrain height so nothing runs off the bottom.
    // Slide is 5.625" tall; title occupies y ∈ [0.55, 1.25]; page num at y=5.25.
    // Safe image zone: y 1.35 .. 4.95 = 3.6" tall. Image aspect 1650/990 = 1.667.
    const imgH = 3.55;
    const imgW = imgH * (1650/990);   // ≈ 5.92
    const imgX = (10 - imgW) / 2;
    slide.addImage({
        path: MIP_HERO,
        x: imgX, y: 1.35, w: imgW, h: imgH,
    });

    // caption
    slide.addText("MIP 三軸視圖：上列 intensity、下列 instance labels（亂色顯示）。Canvas 989 × 646 × 337、~2 M non-zero voxels。",{
        x: 0.5, y: 1.35 + imgH + 0.1, w: 9, h: 0.4,
        fontSize: 10, fontFace: BODY_FONT, color: MUTED, italic: true, align: "center", margin: 0,
    });
}

// =========================================================================
// SLIDE 9 — Outputs
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Outputs", 9, TOTAL_SLIDES);
    addSlideTitle(slide, "每次合成產生的檔案");

    const items = [
        { file: "intensity.am / .nii.gz",   desc: "uint16 強度影像，canvas 989×646×337 + 加上 N(100, 50) Gaussian 雜訊", size: "~400 MB" },
        { file: "labels.am / .nii.gz",      desc: "uint16 label 欄位（Avizo 原生 label field，有 Materials 區段與 per-label 顏色）", size: "~400 MB" },
        { file: "neuron_list.tsv",          desc: "每顆 label 對應到哪顆原始 warp 檔 + driver + phase (greedy/repair/expand) + bbox coverage", size: "~20 KB" },
        { file: "contacts.csv",             desc: "兩兩神經元 voxel 接觸統計：面 (F)、邊 (E)、頂點 (V) 各自的 voxel pair 數量", size: "~5 KB" },
        { file: "mip.png",                  desc: "三軸 MIP 預覽圖（intensity + labels 六宮格）用於快速肉眼驗證", size: "~300 KB" },
        { file: "scan_video.mp4",           desc: "逐 label 掃描影片 — 新 neuron 亮白、舊的暗灰、左上角字卡 N = i", size: "~1 MB" },
    ];

    const rowY = 1.4, rowH = 0.55;
    items.forEach((it, i) => {
        const y = rowY + i * rowH;
        // alternating row background for readability
        if (i % 2 === 0) {
            slide.addShape(pres.shapes.RECTANGLE, {
                x: 0.5, y, w: 9, h: rowH,
                fill: { color: "FFFFFF" }, line: { color: "FFFFFF", width: 0 },
            });
        }
        // file name
        slide.addText(it.file, {
            x: 0.65, y: y + 0.05, w: 2.6, h: rowH - 0.1,
            fontSize: 11, fontFace: "Consolas", bold: true, color: PRIMARY, valign: "middle", margin: 0,
        });
        // description
        slide.addText(it.desc, {
            x: 3.35, y: y + 0.05, w: 5.1, h: rowH - 0.1,
            fontSize: 10, fontFace: BODY_FONT, color: TEXT_DARK, valign: "middle", margin: 0,
        });
        // size
        slide.addText(it.size, {
            x: 8.45, y: y + 0.05, w: 1.0, h: rowH - 0.1,
            fontSize: 9, fontFace: BODY_FONT, color: MUTED, italic: true, valign: "middle", align: "right", margin: 0,
        });
    });
}

// =========================================================================
// SLIDE 10 — Noise model
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_LIGHT };
    addSlideHeader(slide, "Noise model", 10, TOTAL_SLIDES);
    addSlideTitle(slide, "雜訊模型：模擬 sCMOS / confocal 相機");

    // Left: formula
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y: 1.4, w: 4.3, h: 2.4,
        fill: { color: "FFFFFF" }, line: { color: GRID, width: 0.5 },
        shadow: shadow(),
    });
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.5, y: 1.4, w: 0.08, h: 2.4,
        fill: { color: PRIMARY }, line: { color: PRIMARY, width: 0 },
    });
    slide.addText("Model", {
        x: 0.75, y: 1.5, w: 3, h: 0.35,
        fontSize: 12, fontFace: HEAD_FONT, bold: true, color: PRIMARY, charSpacing: 3, margin: 0,
    });
    slide.addText("intensity' = intensity + baseline + 𝒩(0, σ)", {
        x: 0.75, y: 1.95, w: 4.0, h: 0.6,
        fontSize: 14, fontFace: "Consolas", color: TEXT_DARK, margin: 0,
    });
    slide.addText("clipped to [0, 65535]", {
        x: 0.75, y: 2.55, w: 4.0, h: 0.35,
        fontSize: 11, fontFace: "Consolas", color: MUTED, italic: true, margin: 0,
    });
    slide.addText([
        { text: "baseline = 100", options: { bold: true, breakLine: true } },
        { text: "σ = 50", options: { bold: true, breakLine: true } },
        { text: "~ sCMOS dark-current level", options: { italic: true } },
    ], {
        x: 0.75, y: 3.05, w: 4.0, h: 0.7,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 3, margin: 0,
    });

    // Right: rationale
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 5.2, y: 1.4, w: 4.3, h: 3.5,
        fill: { color: "FFFFFF" }, line: { color: GRID, width: 0.5 },
        shadow: shadow(),
    });
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 5.2, y: 1.4, w: 0.08, h: 3.5,
        fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
    });
    slide.addText("為什麼需要 baseline offset", {
        x: 5.45, y: 1.5, w: 3.9, h: 0.35,
        fontSize: 12, fontFace: HEAD_FONT, bold: true, color: ACCENT, charSpacing: 2, margin: 0,
    });
    slide.addText([
        { text: "無 baseline：", options: { bold: true } },
        { text: "clip at 0 會讓背景變成 folded Gaussian、mean 被推高到 ≈ 0.4σ、分布偏斜。", options: { breakLine: true } },
        { text: "有 baseline = 100：", options: { bold: true } },
        { text: "背景落在乾淨的 N(100, 50)，只有 ~2.3% 尾巴會 clip；符合真實相機 dark-current offset 物理。", options: { breakLine: true } },
        { text: "實測驗證：", options: { bold: true } },
        { text: "背景 10×10×10 corner mean = 97.84, std = 50.17 — 對應理論 100/50。", options: {} },
    ], {
        x: 5.45, y: 1.95, w: 4.0, h: 3.0,
        fontSize: 11, fontFace: BODY_FONT, color: TEXT_DARK, paraSpaceAfter: 4, margin: 0,
    });

    slide.addText("labels.am 完全不加雜訊 — ground truth 保持乾淨、供 loss function 使用。", {
        x: 0.5, y: 5.0, w: 9, h: 0.35,
        fontSize: 11, fontFace: BODY_FONT, color: MUTED, italic: true, align: "center", margin: 0,
    });
}

// =========================================================================
// SLIDE 11 — Summary
// =========================================================================
{
    const slide = pres.addSlide();
    slide.background = { color: BG_DARK };
    // no left bar on summary slide for visual distinction

    slide.addText("Summary", {
        x: 0.6, y: 0.7, w: 8, h: 0.7,
        fontSize: 32, fontFace: HEAD_FONT, bold: true, color: TEXT_LIGHT, charSpacing: 2, margin: 0,
    });
    slide.addShape(pres.shapes.RECTANGLE, {
        x: 0.6, y: 1.4, w: 0.8, h: 0.05,
        fill: { color: ACCENT }, line: { color: ACCENT, width: 0 },
    });

    const points = [
        { k: "輸入",   v: "9987 顆 FlyCircuit warp volumes（4 個 GAL4 driver 混合）" },
        { k: "輸出",   v: "每次一組 paired (intensity, labels) 3D volume + 分析檔 + 視訊" },
        { k: "演算法", v: "3-phase selection（greedy → repair → expand），支援隨機 seed" },
        { k: "規模",   v: "N = 500 request、K ≈ 140 / R ≈ 150 achieved per run" },
        { k: "多樣性", v: "11 runs 產出 1079 unique 神經元，pairwise Jaccard 0.06" },
        { k: "用途",   v: "single-neuron auto-segmentation 演算法訓練資料" },
    ];

    const lineY = 1.8, lineH = 0.5;
    points.forEach((p, i) => {
        const y = lineY + i * lineH;
        slide.addText(p.k, {
            x: 0.6, y, w: 1.6, h: lineH - 0.05,
            fontSize: 12, fontFace: HEAD_FONT, bold: true, color: ACCENT, valign: "middle", charSpacing: 2, margin: 0,
        });
        slide.addText(p.v, {
            x: 2.3, y, w: 7.2, h: lineH - 0.05,
            fontSize: 12, fontFace: BODY_FONT, color: TEXT_LIGHT, valign: "middle", margin: 0,
        });
    });

    // page number
    slide.addText(`${TOTAL_SLIDES} / ${TOTAL_SLIDES}`, {
        x: 9.0, y: 5.25, w: 0.9, h: 0.25,
        fontSize: 8, fontFace: BODY_FONT, color: MUTED, align: "right", margin: 0,
    });
}

// Write out
pres.writeFile({ fileName: OUTPUT_PATH }).then(() => {
    console.log("wrote", OUTPUT_PATH);
});
