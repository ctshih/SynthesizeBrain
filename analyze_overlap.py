"""Pairwise neuron-overlap report across multiple synthesize runs.

For a training-data diversity check: if the same neuron shows up in many
different runs, those runs are correlated and add less information to the
training set.

Scans every `output/output_N*_K*_R*_s*/neuron_list.tsv`, reads each run's
set of neuron filenames, and prints:
  * per-run size (R)
  * pairwise intersection size matrix
  * pairwise Jaccard (intersection / union)
  * per-neuron "run count" distribution: how many runs each unique neuron
    appeared in — this is the headline diversity metric
  * total unique neurons across all runs vs sum of R_i (raw redundancy)

Usage:
    python analyze_overlap.py
    python analyze_overlap.py --pattern "output_N500_*"
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np


def _read_neurons(tsv_path: Path) -> set[str]:
    names: set[str] = set()
    with open(tsv_path) as fh:
        r = csv.reader(fh, delimiter="\t")
        header = next(r)
        fn_idx = header.index("filename")
        for row in r:
            names.add(row[fn_idx])
    return names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-root", type=Path, default=Path("output"))
    ap.add_argument("--pattern", type=str, default="output_N500_*",
                    help="Glob (default: output_N500_*). Pass output_N*_* to "
                         "include every run regardless of N.")
    args = ap.parse_args()

    dirs = sorted(d for d in args.output_root.glob(args.pattern)
                  if d.is_dir() and (d / "neuron_list.tsv").exists())
    if not dirs:
        raise SystemExit(f"no matching run dirs under {args.output_root}/{args.pattern}")

    runs: list[tuple[str, set[str]]] = []
    for d in dirs:
        runs.append((d.name, _read_neurons(d / "neuron_list.tsv")))

    nR = len(runs)
    sizes = [len(n) for _, n in runs]

    print(f"# Found {nR} runs under {args.output_root}/{args.pattern}")
    print()
    print("## Per-run size (R)")
    for name, s in zip([n for n, _ in runs], sizes):
        print(f"  {name:<48}  R = {s}")
    print()

    # Pairwise intersection matrix (upper triangle).
    print("## Pairwise intersection |A ∩ B|")
    print("     " + " ".join(f"r{j+1:>2}" for j in range(nR)))
    for i in range(nR):
        row = [f"r{i+1:>2}:"]
        for j in range(nR):
            if j <= i:
                row.append("   .")
            else:
                inter = len(runs[i][1] & runs[j][1])
                row.append(f"{inter:>4}")
        print(" ".join(row))
    print()

    # Jaccard similarity.
    print("## Pairwise Jaccard  |A ∩ B| / |A ∪ B|")
    print("     " + " ".join(f"  r{j+1:>2}" for j in range(nR)))
    for i in range(nR):
        row = [f"r{i+1:>2}:"]
        for j in range(nR):
            if j <= i:
                row.append("     .")
            else:
                A, B = runs[i][1], runs[j][1]
                j_val = len(A & B) / len(A | B) if (A | B) else 0.0
                row.append(f"{j_val:5.3f}")
        print(" ".join(row))
    print()

    # Headline diversity: per-neuron run count distribution.
    run_count = Counter()
    for _, neurons in runs:
        for n in neurons:
            run_count[n] += 1
    count_hist = Counter(run_count.values())

    all_unique = len(run_count)
    total_placements = sum(sizes)
    print("## Diversity (headline metrics)")
    print(f"  unique neurons across all {nR} runs: {all_unique}")
    print(f"  total placements (sum of R): {total_placements}")
    print(f"  redundancy factor: {total_placements/all_unique:.2f}  "
          f"(1 = no sharing, {nR} = every neuron in every run)")
    print()
    print("  distribution: how many runs each unique neuron appeared in")
    print("    runs appeared  |  neurons  |  % of unique")
    print("    ---------------+-----------+-------------")
    for k in range(1, nR + 1):
        n = count_hist.get(k, 0)
        pct = 100 * n / all_unique if all_unique else 0
        bar = "#" * int(round(40 * n / max(count_hist.values())))
        print(f"    {k:>13}  |  {n:>7}  |  {pct:5.1f}%  {bar}")
    print()

    # Average pairwise Jaccard as a single-number "correlation" summary.
    jaccards = []
    for i in range(nR):
        for j in range(i + 1, nR):
            A, B = runs[i][1], runs[j][1]
            if A | B:
                jaccards.append(len(A & B) / len(A | B))
    if jaccards:
        print(f"## Pairwise Jaccard summary")
        print(f"  mean:   {np.mean(jaccards):.3f}")
        print(f"  median: {np.median(jaccards):.3f}")
        print(f"  min:    {np.min(jaccards):.3f}")
        print(f"  max:    {np.max(jaccards):.3f}")
        print()
        print("  Interpretation: 0 = no shared neurons, 1 = identical runs.")
        print("  For training-data diversity, lower is better.")


if __name__ == "__main__":
    main()
