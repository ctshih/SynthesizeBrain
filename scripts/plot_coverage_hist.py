"""Plot C1 bbox_coverage histograms for the same-seed score-mode comparison.

Reads `neuron_list.tsv` from each of the 4 selected output dirs, plots them
overlaid on one figure plus a 2x2 grid of per-config histograms.
"""
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_ROOT = Path("output")

# (label, dir name) — same seed 683619746
CONFIGS = [
    ("density / T=0.5",     "output_N500_K147_R148_s683619746"),
    ("hybrid / T=0.5",      "output_N500_K169_R173_s683619746"),
    ("small-first / T=0.5", "output_N500_K191_R195_s683619746"),
    ("small-first / T=0.0", "output_N500_K193_R193_s683619746"),
]


def load_coverages(d: Path) -> tuple[np.ndarray, list[str]]:
    covs, phases = [], []
    with open(d / "neuron_list.tsv") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            covs.append(float(row["bbox_coverage"]))
            phases.append(row["phase"])
    return np.array(covs), phases


datasets = [(label, *load_coverages(OUTPUT_ROOT / d)) for label, d in CONFIGS]

# Figure 1: 2x2 grid with phase coloring
fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), sharex=True, sharey=True)
bins = np.linspace(0, 1, 21)  # 0.05 wide bins
phase_colors = {"greedy": "#0E7C7B", "repair": "#E76F51", "expand": "#FFC857"}

for ax, (label, covs, phases) in zip(axes.flat, datasets):
    phases = np.array(phases)
    bottoms = np.zeros(len(bins) - 1)
    for ph in ["greedy", "repair", "expand"]:
        mask = phases == ph
        if mask.sum() == 0:
            continue
        counts, _ = np.histogram(covs[mask], bins=bins)
        ax.bar(bins[:-1], counts, width=np.diff(bins), align="edge",
               bottom=bottoms, color=phase_colors[ph], edgecolor="white",
               linewidth=0.4, label=f"{ph} (n={mask.sum()})")
        bottoms += counts
    ax.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_title(f"{label}   N={len(covs)}   mean={covs.mean():.3f}   min={covs.min():.3f}",
                 fontsize=10)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.25)

for ax in axes[-1]:
    ax.set_xlabel("bbox_coverage")
for ax in axes[:, 0]:
    ax.set_ylabel("count")

fig.suptitle("C1 bbox_coverage distribution by score_mode / threshold "
             "(seed=683619746, N=500)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out_path = Path("output") / "coverage_hist_grid.png"
fig.savefig(out_path, dpi=140, bbox_inches="tight")
print(f"saved {out_path}")

# Figure 2: overlay
fig2, ax2 = plt.subplots(figsize=(10, 5))
overlay_colors = ["#0E7C7B", "#9B5DE5", "#E76F51", "#FFC857"]
for (label, covs, _), c in zip(datasets, overlay_colors):
    ax2.hist(covs, bins=bins, histtype="step", linewidth=2.2,
             label=f"{label}  (N={len(covs)}, mean={covs.mean():.3f})",
             color=c)
ax2.axvline(0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6,
            label="default C1 threshold = 0.5")
ax2.set_xlabel("bbox_coverage")
ax2.set_ylabel("count")
ax2.set_title("C1 bbox_coverage overlay (seed=683619746, N=500)")
ax2.legend(fontsize=9, loc="upper left", frameon=False)
ax2.grid(alpha=0.25)
out_path2 = Path("output") / "coverage_hist_overlay.png"
fig2.savefig(out_path2, dpi=140, bbox_inches="tight")
print(f"saved {out_path2}")

# Text summary
print("\n=== text summary ===")
for label, covs, phases in datasets:
    phases = np.array(phases)
    print(f"\n{label}:  N={len(covs)}  mean={covs.mean():.3f}  "
          f"median={np.median(covs):.3f}  min={covs.min():.3f}  "
          f"max={covs.max():.3f}  std={covs.std():.3f}")
    print(f"  below 0.5: {(covs < 0.5).sum():3d}  "
          f"below 0.3: {(covs < 0.3).sum():3d}  "
          f"below 0.1: {(covs < 0.1).sum():3d}")
    for ph in ["greedy", "repair", "expand"]:
        mask = phases == ph
        if mask.sum():
            print(f"  {ph:7s}: n={mask.sum():3d}  mean={covs[mask].mean():.3f}  "
                  f"min={covs[mask].min():.3f}")
