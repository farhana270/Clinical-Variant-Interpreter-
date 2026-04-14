"""
cnv_analysis.py
===============
Module 2: Copy Number Variant (CNV) Detection and Analysis
===========================================================
Detects deletions and duplications from read depth coverage data
using log2 ratio analysis and circular binary segmentation.

What is a CNV?
    A Copy Number Variant is a region of the genome where the number
    of copies of a DNA segment differs from the normal 2 copies
    (one from each parent). Deletions have fewer copies (<2),
    duplications have more copies (>2).

    CNVs are clinically significant because:
    - Large deletions in BRCA1/BRCA2 cause hereditary breast cancer
    - CNVs in cancer genes drive tumour progression
    - They account for ~15% of disease-causing variants missed by SNV pipelines

Pipeline overview:
    Step 1 → Simulate realistic chr20 coverage data (tumour + normal)
    Step 2 → Normalise coverage and calculate log2 ratio per window
    Step 3 → Segment the genome using circular binary segmentation (CBS)
    Step 4 → Call CNVs from segments using log2 ratio thresholds
    Step 5 → Annotate CNVs with overlapping cancer genes
    Step 6 → Generate plots and HTML report

Standard reference:
    Miller DT et al. (2010) Consensus statement: chromosomal microarray
    is a first-tier clinical diagnostic test. Am J Hum Genet 86:749-764

Author  : Farhana Sayed
Project : Clinical Variant Interpreter — Module 2 of 3
Input   : Simulated chr20 coverage data (based on GIAB HG002 profiles)
Output  : cnv_calls.csv | cnv_plots.png | cnv_report.html

Run:
    pip3 install pandas matplotlib numpy scipy --break-system-packages
    python3 scripts/cnv_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats
from datetime import datetime

# ── Output paths ──────────────────────────────────────────
BASE      = os.path.expanduser("~/variant-annotation-pipeline")
OUT_COV   = os.path.join(BASE, "results", "chr20_coverage.bed")
OUT_CSV   = os.path.join(BASE, "results", "cnv_calls.csv")
OUT_SEG   = os.path.join(BASE, "results", "cnv_segments.csv")
OUT_PNG   = os.path.join(BASE, "results", "cnv_plots.png")
OUT_HTML  = os.path.join(BASE, "results", "cnv_report.html")

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

# ── Chromosome 20 parameters ──────────────────────────────
CHROM         = "chr20"
CHROM_LENGTH  = 64_444_167   # GRCh38 chr20 length in base pairs
WINDOW_SIZE   = 50_000        # 50kb windows — standard for CNV analysis
MIN_WINDOWS   = 3             # minimum windows to call a CNV segment

# ── Log2 ratio thresholds ─────────────────────────────────
# These are industry-standard thresholds used in clinical CNV calling
# log2(0/2) = -inf (homozygous deletion)
# log2(1/2) = -1.0 (heterozygous deletion — one copy lost)
# log2(2/2) =  0.0 (normal diploid — two copies)
# log2(3/2) = +0.585 (one copy gain — duplication)
# log2(4/2) = +1.0 (two copy gain — amplification)
DEL_THRESHOLD  = -0.4   # below this = deletion
DUP_THRESHOLD  =  0.3   # above this = duplication
AMP_THRESHOLD  =  1.0   # above this = high-level amplification

# ── Cancer genes on chromosome 20 ─────────────────────────
# Each entry: gene_name: (start, end, significance)
CANCER_GENES_CHR20 = {
    "ASXL1":  (31_020_000, 31_070_000,
                "Myeloid malignancies — loss causes clonal haematopoiesis"),
    "DNMT3B": (32_750_000, 32_830_000,
                "ICF syndrome — DNA methylation gene"),
    "PTPRT":  (40_700_000, 41_100_000,
                "Colorectal cancer — tumour suppressor"),
    "PTPN1":  (48_680_000, 48_760_000,
                "Leukaemia — protein tyrosine phosphatase"),
    "BCAS1":  (50_600_000, 50_680_000,
                "Breast cancer amplification target — 20q13"),
    "AURKA":  (54_940_000, 55_000_000,
                "Aurora kinase A — amplified in breast/colorectal cancer"),
    "MYBL2":  (42_280_000, 42_360_000,
                "Cell cycle — amplified in many tumour types"),
    "NCOA3":  (46_140_000, 46_270_000,
                "Steroid receptor co-activator — breast cancer amplification"),
    "ZNF217": (52_180_000, 52_230_000,
                "20q13 amplicon — breast/ovarian/colorectal cancer"),
    "GNAS":   (57_400_000, 57_500_000,
                "Endocrine tumours — activating mutations"),
}

# ── CNV type colours ──────────────────────────────────────
CNV_COLORS = {
    "deletion":      "#c0392b",
    "duplication":   "#2980b9",
    "amplification": "#8e44ad",
    "normal":        "#95a5a6",
}


# ══════════════════════════════════════════════════════════
# STEP 1 — SIMULATE REALISTIC COVERAGE DATA
# ══════════════════════════════════════════════════════════
def simulate_coverage():
    """
    Generates realistic tumour and normal coverage profiles for chr20.

    WHY SIMULATE?
    Real coverage data requires BAM files (>10GB) from whole-genome
    sequencing. For a portfolio project, simulating realistic coverage
    profiles demonstrates the same analytical skills while avoiding
    massive downloads. The simulation uses parameters derived from
    published GIAB (Genome in a Bottle) HG002 sample profiles.

    WHAT IS COVERAGE?
    When a genome is sequenced, each base pair is read multiple times.
    Coverage (read depth) is how many times a specific position was
    sequenced. Standard whole-genome sequencing targets 30x coverage
    (each base read ~30 times on average).

    HOW CNVs SHOW UP IN COVERAGE:
    - Normal region: ~30 reads per position (diploid, 2 copies)
    - Deletion: ~15 reads (only 1 copy to sequence)
    - Duplication: ~45 reads (3 copies produce more fragments)
    - Amplification: >60 reads (many copies)
    """
    print("\n[1/6] Simulating chr20 coverage data...")

    np.random.seed(42)   # reproducibility — same random numbers every run

    # Create 50kb windows across chr20
    windows = list(range(0, CHROM_LENGTH, WINDOW_SIZE))
    n_windows = len(windows)

    # ── Normal sample baseline coverage ───────────────────
    # Real WGS data has:
    # - Mean ~30x coverage
    # - GC content bias (high-GC regions have lower coverage)
    # - Random Poisson noise (sequencing is a random sampling process)
    normal_base = 30.0
    gc_bias = np.sin(np.linspace(0, 4 * np.pi, n_windows)) * 3
    # Poisson noise — the natural statistical variation in sequencing depth
    normal_coverage = np.random.poisson(
        lam=np.maximum(normal_base + gc_bias, 5),
        size=n_windows
    ).astype(float)

    # ── Tumour sample coverage ─────────────────────────────
    # Start with the same baseline as normal
    tumour_coverage = normal_coverage.copy()

    # Introduce realistic CNV events:
    # Tumour genomes accumulate chromosomal copy number changes
    # as cancer cells divide and acquire genomic instability

    # Event 1: ASXL1 region deletion (chr20p loss — common in AML)
    # Window indices for 31.0–31.5 Mb
    del1_start = int(31_000_000 / WINDOW_SIZE)
    del1_end   = int(31_500_000 / WINDOW_SIZE)
    # Halve coverage in deletion region (loss of one allele)
    tumour_coverage[del1_start:del1_end] *= 0.5
    tumour_coverage[del1_start:del1_end] += np.random.normal(0, 1, del1_end - del1_start)

    # Event 2: 20q13 amplification — AURKA/ZNF217 amplicon
    # This is one of the most common amplifications in breast cancer
    amp_start = int(54_000_000 / WINDOW_SIZE)
    amp_end   = int(56_000_000 / WINDOW_SIZE)
    tumour_coverage[amp_start:amp_end] *= 3.5
    tumour_coverage[amp_start:amp_end] += np.random.normal(0, 3, amp_end - amp_start)

    # Event 3: NCOA3 duplication — breast cancer co-amplification target
    dup_start = int(46_000_000 / WINDOW_SIZE)
    dup_end   = int(46_500_000 / WINDOW_SIZE)
    tumour_coverage[dup_start:dup_end] *= 1.6
    tumour_coverage[dup_start:dup_end] += np.random.normal(0, 2, dup_end - dup_start)

    # Event 4: PTPRT deletion — colorectal tumour suppressor loss
    del2_start = int(40_700_000 / WINDOW_SIZE)
    del2_end   = int(41_200_000 / WINDOW_SIZE)
    tumour_coverage[del2_start:del2_end] *= 0.45
    tumour_coverage[del2_start:del2_end] += np.random.normal(0, 1, del2_end - del2_start)

    # Event 5: BCAS1/ZNF217 duplication — 20q gain
    dup2_start = int(50_000_000 / WINDOW_SIZE)
    dup2_end   = int(53_000_000 / WINDOW_SIZE)
    tumour_coverage[dup2_start:dup2_end] *= 1.8
    tumour_coverage[dup2_start:dup2_end] += np.random.normal(0, 2, dup2_end - dup2_start)

    # Ensure no negative coverage (impossible in real data)
    tumour_coverage = np.maximum(tumour_coverage, 1.0)
    normal_coverage = np.maximum(normal_coverage, 1.0)

    # Build DataFrame in BED format (standard genomics interval format)
    df = pd.DataFrame({
        "chrom":           CHROM,
        "start":           windows,
        "end":             [min(w + WINDOW_SIZE, CHROM_LENGTH) for w in windows],
        "normal_coverage": np.round(normal_coverage, 2),
        "tumour_coverage": np.round(tumour_coverage, 2),
    })

    # Save as BED file (tab-separated genomic intervals)
    df.to_csv(OUT_COV, sep="\t", index=False)

    print(f"  OK: {len(df)} windows of {WINDOW_SIZE//1000}kb across chr20")
    print(f"  OK: Normal mean coverage: {normal_coverage.mean():.1f}x")
    print(f"  OK: Tumour mean coverage: {tumour_coverage.mean():.1f}x")
    print(f"  OK: Saved: {OUT_COV}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 2 — NORMALISE AND CALCULATE LOG2 RATIO
# ══════════════════════════════════════════════════════════
def calculate_log2_ratio(df):
    """
    Calculates the log2 ratio of tumour vs normal coverage per window.

    WHY LOG2 RATIO?
    Raw coverage values fluctuate due to GC bias, mappability differences,
    and sequencing noise. By dividing tumour by normal and taking log2,
    we:
    1. Cancel out systematic biases that affect both samples equally
    2. Centre the scale at 0 (normal diploid state)
    3. Make deletions and duplications symmetrically visible
    4. Convert multiplicative changes to additive ones (easier to segment)

    NORMALISATION STEPS:
    1. Median normalise each sample (correct for different sequencing depths)
    2. Divide tumour by normal (remove shared biases)
    3. Take log2 (centre at 0, linearise the scale)

    WHAT THE VALUES MEAN:
        log2(1/2) = -1.00  → homozygous deletion (0 copies)
        log2(1/2) = -1.00  → heterozygous deletion (1 copy)
        log2(2/2) =  0.00  → normal diploid (2 copies)
        log2(3/2) = +0.58  → single copy gain (3 copies)
        log2(4/2) = +1.00  → two copy gain (4 copies)
        log2(8/2) = +2.00  → high amplification (8 copies)
    """
    print("\n[2/6] Calculating log2 ratios...")

    # Step 2a: Median normalisation
    # Dividing by the median corrects for different total sequencing depths
    # between the tumour and normal samples
    normal_median = df["normal_coverage"].median()
    tumour_median = df["tumour_coverage"].median()

    df["normal_norm"] = df["normal_coverage"] / normal_median
    df["tumour_norm"] = df["tumour_coverage"] / tumour_median

    # Step 2b: Calculate ratio
    # Adding a small epsilon (1e-6) prevents log2(0) = -infinity
    ratio = df["tumour_norm"] / (df["normal_norm"] + 1e-6)

    # Step 2c: Log2 transform
    df["log2_ratio"] = np.log2(ratio + 1e-6)

    # Step 2d: Smooth with rolling median to reduce noise
    # A 5-window rolling median removes single-window spikes
    # while preserving real CNV boundaries
    df["log2_ratio_smooth"] = (
        df["log2_ratio"]
        .rolling(window=5, center=True, min_periods=1)
        .median()
    )

    # Window midpoint for plotting
    df["midpoint"] = (df["start"] + df["end"]) / 2

    print(f"  OK: Log2 ratio range: {df['log2_ratio'].min():.2f} "
          f"to {df['log2_ratio'].max():.2f}")
    print(f"  OK: Median log2 ratio: {df['log2_ratio'].median():.3f} "
          f"(expected ~0.0 for diploid regions)")
    return df


# ══════════════════════════════════════════════════════════
# STEP 3 — SEGMENTATION (CIRCULAR BINARY SEGMENTATION)
# ══════════════════════════════════════════════════════════
def segment_genome(df):
    """
    Segments the genome into regions of uniform copy number.

    WHY SEGMENTATION?
    Individual window log2 ratios are noisy. Segmentation groups
    consecutive windows with similar log2 ratios into segments —
    like finding the boundaries between normal and altered regions.

    CIRCULAR BINARY SEGMENTATION (CBS):
    CBS is the gold standard algorithm for CNV segmentation
    (Olshen et al. 2004, Biostatistics 5:557-572). It recursively
    finds the best split point in the data until no significant
    change points remain. We implement a simplified version using
    sliding window t-tests to detect breakpoints.

    SIMPLIFIED CBS LOGIC:
    1. Start with the whole chromosome as one segment
    2. Find the position where a t-test shows the greatest
       mean difference between left and right halves
    3. If the difference is statistically significant, split there
    4. Recursively apply to each sub-segment
    5. Stop when no significant breakpoints remain
    """
    print("\n[3/6] Segmenting genome using circular binary segmentation...")

    log2 = df["log2_ratio_smooth"].values
    positions = df["midpoint"].values
    segments = []

    def find_breakpoints(start_idx, end_idx, depth=0):
        """Recursively find significant breakpoints in a region."""
        if end_idx - start_idx < MIN_WINDOWS * 2:
            return

        region = log2[start_idx:end_idx]
        best_t    = 0
        best_split = -1

        # Scan every possible split point
        for i in range(MIN_WINDOWS, len(region) - MIN_WINDOWS):
            left  = region[:i]
            right = region[i:]

            # Two-sample t-test: are the means significantly different?
            if len(left) >= 3 and len(right) >= 3:
                try:
                    t_stat, p_val = stats.ttest_ind(left, right)
                    if abs(t_stat) > abs(best_t) and p_val < 0.01:
                        best_t     = t_stat
                        best_split = start_idx + i
                except Exception:
                    continue

        if best_split > 0 and abs(best_t) > 2.0:
            # Significant breakpoint found — recurse on both halves
            find_breakpoints(start_idx, best_split, depth + 1)
            find_breakpoints(best_split, end_idx, depth + 1)
        else:
            # No significant breakpoint — this is a segment
            seg_log2 = log2[start_idx:end_idx]
            segments.append({
                "chrom":      CHROM,
                "start":      int(positions[start_idx]),
                "end":        int(positions[end_idx - 1]),
                "n_windows":  end_idx - start_idx,
                "mean_log2":  float(np.mean(seg_log2)),
                "std_log2":   float(np.std(seg_log2)),
                "median_log2": float(np.median(seg_log2)),
            })

    find_breakpoints(0, len(log2))

    seg_df = pd.DataFrame(segments)

    # Sort by start position
    if len(seg_df) > 0:
        seg_df = seg_df.sort_values("start").reset_index(drop=True)

    print(f"  OK: Found {len(seg_df)} genomic segments")
    if len(seg_df) > 0:
        print(f"  OK: Segment log2 range: "
              f"{seg_df['mean_log2'].min():.2f} to "
              f"{seg_df['mean_log2'].max():.2f}")
    return seg_df


# ══════════════════════════════════════════════════════════
# STEP 4 — CALL CNVs FROM SEGMENTS
# ══════════════════════════════════════════════════════════
def call_cnvs(seg_df):
    """
    Applies log2 ratio thresholds to classify each segment as
    deletion, duplication, amplification, or normal.

    CNV CALLING THRESHOLDS (clinical standard):
    ┌─────────────────────────────┬──────────────┬──────────────────────┐
    │ Copy number state            │ Log2 ratio   │ Classification       │
    ├─────────────────────────────┼──────────────┼──────────────────────┤
    │ Homozygous deletion (0 copy) │ < -1.0       │ Deletion (severe)    │
    │ Heterozygous deletion        │ -1.0 to -0.4 │ Deletion             │
    │ Normal diploid (2 copies)    │ -0.4 to +0.3 │ Normal               │
    │ Single copy gain             │ +0.3 to +1.0 │ Duplication          │
    │ High-level amplification     │ > +1.0       │ Amplification        │
    └─────────────────────────────┴──────────────┴──────────────────────┘

    MINIMUM SIZE FILTER:
    Segments smaller than 3 windows (150kb at 50kb resolution) are
    filtered out because they are likely to be noise rather than
    real CNVs. Clinical CNV calling typically requires events >100kb.
    """
    print("\n[4/6] Calling CNVs from segments...")

    if len(seg_df) == 0:
        print("  WARNING: No segments found — check coverage data")
        return pd.DataFrame()

    cnv_calls = []

    for _, seg in seg_df.iterrows():
        log2 = seg["mean_log2"]
        size = seg["end"] - seg["start"]

        # Classify by log2 ratio threshold
        if log2 < DEL_THRESHOLD:
            cnv_type = "deletion"
        elif log2 > AMP_THRESHOLD:
            cnv_type = "amplification"
        elif log2 > DUP_THRESHOLD:
            cnv_type = "duplication"
        else:
            cnv_type = "normal"

        # Estimate copy number from log2 ratio
        # Formula: CN = 2 × 2^(log2_ratio) for diploid tumour purity
        estimated_cn = round(2 * (2 ** log2), 1)
        estimated_cn = max(0.0, estimated_cn)

        cnv_calls.append({
            "chrom":        CHROM,
            "start":        int(seg["start"]),
            "end":          int(seg["end"]),
            "size_kb":      round(size / 1000, 1),
            "n_windows":    int(seg["n_windows"]),
            "mean_log2":    round(log2, 3),
            "std_log2":     round(seg["std_log2"], 3),
            "cnv_type":     cnv_type,
            "estimated_cn": estimated_cn,
            "genes":        "",       # filled in step 5
            "clinical_sig": "",       # filled in step 5
        })

    cnv_df = pd.DataFrame(cnv_calls)

    # Summary
    for cnv_type in ["deletion", "duplication", "amplification", "normal"]:
        n = len(cnv_df[cnv_df["cnv_type"] == cnv_type])
        if n:
            print(f"  {cnv_type:<15}: {n} segment(s)")

    return cnv_df


# ══════════════════════════════════════════════════════════
# STEP 5 — ANNOTATE CNVs WITH CANCER GENES
# ══════════════════════════════════════════════════════════
def annotate_cnvs(cnv_df):
    """
    Overlaps CNV segments with known cancer gene coordinates.

    WHY GENE ANNOTATION?
    A CNV's clinical significance depends entirely on which genes
    it disrupts. A 500kb deletion in a gene desert has no clinical
    impact. The same 500kb deletion overlapping BRCA1 is
    potentially pathogenic.

    OVERLAP LOGIC:
    Two genomic intervals overlap if:
        CNV_start < gene_end AND CNV_end > gene_start
    This is the standard interval overlap test used in tools like
    bedtools, which is the industry standard for genomic annotation.

    CLINICAL SIGNIFICANCE RULES:
    - Deletion of tumour suppressor gene → likely pathogenic
    - Amplification of oncogene → likely pathogenic
    - Deletion of oncogene → neutral/benign (loses growth advantage)
    - Amplification of tumour suppressor → neutral/benign
    """
    print("\n[5/6] Annotating CNVs with cancer gene overlaps...")

    for idx, cnv in cnv_df.iterrows():
        overlapping_genes = []
        clinical_notes = []

        for gene_name, (gene_start, gene_end, gene_sig) in \
                CANCER_GENES_CHR20.items():

            # Standard interval overlap test
            if cnv["start"] < gene_end and cnv["end"] > gene_start:
                overlapping_genes.append(gene_name)

                # Determine clinical significance of CNV + gene combination
                cnv_type = cnv["cnv_type"]

                # Known tumour suppressor genes (loss = pathogenic)
                tumour_suppressors = {"ASXL1", "PTPRT", "PTPN1"}
                # Known oncogenes (gain = pathogenic)
                oncogenes = {"BCAS1", "AURKA", "MYBL2", "NCOA3", "ZNF217"}

                if cnv_type == "deletion" and gene_name in tumour_suppressors:
                    clinical_notes.append(
                        f"{gene_name}: deletion of tumour suppressor — "
                        f"likely pathogenic"
                    )
                elif cnv_type in ("duplication", "amplification") and \
                        gene_name in oncogenes:
                    clinical_notes.append(
                        f"{gene_name}: {cnv_type} of oncogene — "
                        f"likely pathogenic"
                    )
                elif cnv_type == "deletion" and gene_name in oncogenes:
                    clinical_notes.append(
                        f"{gene_name}: deletion of oncogene — "
                        f"uncertain significance"
                    )
                else:
                    clinical_notes.append(
                        f"{gene_name}: {cnv_type} — uncertain significance"
                    )

        cnv_df.at[idx, "genes"] = ", ".join(overlapping_genes) \
            if overlapping_genes else "intergenic"
        cnv_df.at[idx, "clinical_sig"] = " | ".join(clinical_notes) \
            if clinical_notes else "no known cancer gene overlap"

    # Count events with gene overlap
    gene_overlap = cnv_df[
        (cnv_df["cnv_type"] != "normal") &
        (cnv_df["genes"] != "intergenic")
    ]
    print(f"  OK: {len(gene_overlap)} CNV(s) overlap cancer genes")
    print(f"  OK: Total non-normal segments: "
          f"{len(cnv_df[cnv_df['cnv_type'] != 'normal'])}")

    # Save CNV calls
    cnv_df.to_csv(OUT_CSV, index=False)
    print(f"  OK: Saved: {OUT_CSV}")
    return cnv_df


# ══════════════════════════════════════════════════════════
# STEP 6 — GENERATE PLOTS
# ══════════════════════════════════════════════════════════
def plot(cov_df, cnv_df):
    """
    Generates a 4-panel CNV summary figure.

    Panel 1: Genome-wide log2 ratio plot (the main CNV view)
    Panel 2: CNV type distribution bar chart
    Panel 3: CNV size distribution histogram
    Panel 4: Cancer gene overlap summary
    """
    print("\n[6a/6] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"Copy Number Variant Analysis — {CHROM} (GRCh38)\n"
        "Data: Simulated tumour coverage (GIAB HG002 parameters) | "
        "Author: Farhana Sayed",
        fontsize=11, fontweight="bold", y=1.01
    )

    # ── Panel 1: Genome-wide log2 ratio ───────────────────
    ax = axes[0, 0]

    # Plot individual window log2 ratios as grey dots
    ax.scatter(
        cov_df["midpoint"] / 1e6,
        cov_df["log2_ratio"],
        s=2, alpha=0.3, color="#95a5a6", label="Window log2 ratio"
    )

    # Plot smoothed log2 ratio as a line
    ax.plot(
        cov_df["midpoint"] / 1e6,
        cov_df["log2_ratio_smooth"],
        color="#2c3e50", linewidth=0.8, alpha=0.7, label="Smoothed"
    )

    # Draw reference lines
    ax.axhline(y=0,               color="black",   linewidth=1.0,
               linestyle="-",  alpha=0.5, label="Normal diploid")
    ax.axhline(y=DEL_THRESHOLD,   color="#c0392b", linewidth=1.0,
               linestyle="--", alpha=0.7, label=f"Del threshold ({DEL_THRESHOLD})")
    ax.axhline(y=DUP_THRESHOLD,   color="#2980b9", linewidth=1.0,
               linestyle="--", alpha=0.7, label=f"Dup threshold ({DUP_THRESHOLD})")
    ax.axhline(y=AMP_THRESHOLD,   color="#8e44ad", linewidth=1.0,
               linestyle=":",  alpha=0.7, label=f"Amp threshold ({AMP_THRESHOLD})")

    # Shade CNV regions
    for _, cnv in cnv_df.iterrows():
        if cnv["cnv_type"] != "normal":
            color = CNV_COLORS.get(cnv["cnv_type"], "#888")
            ax.axvspan(
                cnv["start"] / 1e6,
                cnv["end"] / 1e6,
                alpha=0.15, color=color
            )

    # Label cancer genes on the plot
    for gene, (gstart, gend, _) in CANCER_GENES_CHR20.items():
        gmid = (gstart + gend) / 2 / 1e6
        ax.text(gmid, 2.3, gene, ha="center", va="bottom",
                fontsize=7, rotation=45, color="#2c3e50", alpha=0.8)

    ax.set_xlabel("Chromosome 20 position (Mb)", fontsize=10)
    ax.set_ylabel("Log2 ratio (tumour/normal)", fontsize=10)
    ax.set_title("Genome-wide copy number profile", fontweight="bold")
    ax.set_ylim(-2.5, 2.8)
    ax.set_xlim(0, CHROM_LENGTH / 1e6)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 2: CNV type distribution ────────────────────
    ax = axes[0, 1]
    cnv_counts = cnv_df["cnv_type"].value_counts()
    cnv_order  = ["deletion", "duplication", "amplification", "normal"]
    cnv_order  = [t for t in cnv_order if t in cnv_counts.index]
    counts     = [cnv_counts.get(t, 0) for t in cnv_order]
    colors     = [CNV_COLORS.get(t, "#888") for t in cnv_order]

    bars = ax.bar(cnv_order, counts, color=colors,
                  edgecolor="white", width=0.6)
    ax.set_title("CNV type distribution", fontweight="bold")
    ax.set_ylabel("Number of segments")
    ax.tick_params(axis="x", rotation=15)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val), ha="center", fontsize=10, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 3: CNV size distribution ────────────────────
    ax = axes[1, 0]
    non_normal = cnv_df[cnv_df["cnv_type"] != "normal"]
    if len(non_normal) > 0:
        colors_hist = [CNV_COLORS.get(t, "#888")
                       for t in non_normal["cnv_type"]]
        bars = ax.barh(
            range(len(non_normal)),
            non_normal["size_kb"],
            color=colors_hist, edgecolor="white", height=0.7
        )
        labels = [
            f"{row['genes']} ({row['cnv_type']})"
            for _, row in non_normal.iterrows()
        ]
        ax.set_yticks(range(len(non_normal)))
        ax.set_yticklabels(labels, fontsize=9)
        ax.set_xlabel("Size (kb)", fontsize=10)
        ax.set_title("CNV size and location", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Add legend
        patches = [
            mpatches.Patch(color=CNV_COLORS["deletion"],      label="Deletion"),
            mpatches.Patch(color=CNV_COLORS["duplication"],   label="Duplication"),
            mpatches.Patch(color=CNV_COLORS["amplification"], label="Amplification"),
        ]
        ax.legend(handles=patches, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No CNVs detected",
                ha="center", va="center", fontsize=12)

    # ── Panel 4: Cancer gene impact ───────────────────────
    ax = axes[1, 1]
    gene_events = []
    for _, cnv in non_normal.iterrows():
        if cnv["genes"] != "intergenic":
            for gene in cnv["genes"].split(", "):
                gene_events.append({
                    "gene":     gene,
                    "cnv_type": cnv["cnv_type"],
                    "log2":     cnv["mean_log2"],
                    "size_kb":  cnv["size_kb"],
                })

    if gene_events:
        ge_df = pd.DataFrame(gene_events)
        gene_colors = [CNV_COLORS.get(t, "#888") for t in ge_df["cnv_type"]]
        bars = ax.barh(
            ge_df["gene"],
            ge_df["log2"],
            color=gene_colors, edgecolor="white", height=0.6
        )
        ax.axvline(x=0, color="black", linewidth=1, alpha=0.5)
        ax.axvline(x=DEL_THRESHOLD, color="#c0392b",
                   linewidth=1, linestyle="--", alpha=0.5)
        ax.axvline(x=DUP_THRESHOLD, color="#2980b9",
                   linewidth=1, linestyle="--", alpha=0.5)
        ax.set_xlabel("Mean log2 ratio", fontsize=10)
        ax.set_title("Cancer gene CNV log2 ratios", fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    else:
        ax.text(0.5, 0.5, "No cancer gene overlaps",
                ha="center", va="center", fontsize=12)
        ax.set_title("Cancer gene CNV log2 ratios", fontweight="bold")

    plt.tight_layout(pad=2.5)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK: Saved {OUT_PNG}")


# ══════════════════════════════════════════════════════════
# STEP 7 — HTML REPORT
# ══════════════════════════════════════════════════════════
def html_report(cnv_df):
    """Generates the portfolio HTML report."""
    print("\n[6b/6] Generating HTML report...")

    non_normal = cnv_df[cnv_df["cnv_type"] != "normal"]
    n_del  = len(cnv_df[cnv_df["cnv_type"] == "deletion"])
    n_dup  = len(cnv_df[cnv_df["cnv_type"] == "duplication"])
    n_amp  = len(cnv_df[cnv_df["cnv_type"] == "amplification"])
    n_gene = len(cnv_df[
        (cnv_df["cnv_type"] != "normal") &
        (cnv_df["genes"] != "intergenic")
    ])

    def cnv_badge(cnv_type):
        c = CNV_COLORS.get(cnv_type, "#888")
        return (f'<span style="background:{c};color:white;padding:2px 9px;'
                f'border-radius:10px;font-size:11px;font-weight:bold">'
                f'{cnv_type.upper()}</span>')

    rows_html = ""
    for _, row in non_normal.iterrows():
        log2 = row["mean_log2"]
        lc   = "#c0392b" if log2 < 0 else "#2980b9" \
               if log2 < AMP_THRESHOLD else "#8e44ad"
        rows_html += f"""<tr>
          <td><code>{row['chrom']}:{row['start']:,}–{row['end']:,}</code></td>
          <td>{row['size_kb']} kb</td>
          <td>{row['n_windows']}</td>
          <td style="font-weight:bold;color:{lc}">{row['mean_log2']}</td>
          <td>{row['estimated_cn']}</td>
          <td>{cnv_badge(row['cnv_type'])}</td>
          <td><strong>{row['genes']}</strong></td>
          <td style="font-size:11px">{row['clinical_sig']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>CNV Analysis Report — Farhana Sayed</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f4f6fb;color:#1a1a2e}}
.hdr{{background:linear-gradient(135deg,#1a3a5c,#0d2137);
      color:white;padding:36px 48px}}
.hdr h1{{font-size:21px;margin-bottom:6px}}
.hdr p{{font-size:13px;opacity:.78}}
.body{{padding:28px 48px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);
        gap:12px;margin-bottom:24px}}
.card{{background:white;border-radius:10px;padding:16px;
       text-align:center;box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.card .num{{font-size:28px;font-weight:700}}
.card .lbl{{font-size:10px;color:#888;text-transform:uppercase;
            letter-spacing:.4px;margin-top:4px}}
h2{{font-size:15px;color:#1a3a5c;border-left:4px solid #e74c3c;
    padding-left:12px;margin:22px 0 12px}}
img{{width:100%;border-radius:10px;
     box-shadow:0 2px 10px rgba(0,0,0,.08);margin-bottom:22px}}
table{{width:100%;border-collapse:collapse;font-size:12px;
       background:white;border-radius:8px;overflow:hidden;
       box-shadow:0 2px 8px rgba(0,0,0,.06)}}
th{{background:#1a3a5c;color:white;padding:9px 10px;
    text-align:left;font-size:11px}}
td{{padding:8px 10px;border-bottom:1px solid #eee;vertical-align:middle}}
tr:nth-child(even){{background:#f8f9fd}}
tr:hover{{background:#eef2ff}}
code{{font-size:11px;background:#eef;padding:1px 5px;border-radius:3px}}
.ref{{background:white;border-radius:8px;padding:18px;
      border-left:4px solid #1a3a5c;margin-top:16px;
      font-size:12.5px;line-height:1.8;color:#333}}
.footer{{background:#1a3a5c;color:white;text-align:center;
         padding:16px;font-size:12px;margin-top:22px}}
</style></head><body>
<div class="hdr">
  <h1>Copy Number Variant Analysis Report — chr20</h1>
  <p>Method: Log2 ratio analysis + circular binary segmentation
     &nbsp;|&nbsp; Assembly: GRCh38
     &nbsp;|&nbsp; Window size: {WINDOW_SIZE//1000}kb
     &nbsp;|&nbsp; Generated: {datetime.now().strftime("%d %b %Y %H:%M")}</p>
  <p style="margin-top:5px">Author: Farhana Sayed &nbsp;|&nbsp;
     B.Tech Bioinformatics &amp; Data Science, D.Y. Patil
     &nbsp;|&nbsp; Module 2 of 3 — Clinical Variant Interpreter</p>
</div>
<div class="body">
  <div class="cards">
    <div class="card" style="border-top:4px solid #c0392b">
      <div class="num" style="color:#c0392b">{n_del}</div>
      <div class="lbl">Deletions</div>
    </div>
    <div class="card" style="border-top:4px solid #2980b9">
      <div class="num" style="color:#2980b9">{n_dup}</div>
      <div class="lbl">Duplications</div>
    </div>
    <div class="card" style="border-top:4px solid #8e44ad">
      <div class="num" style="color:#8e44ad">{n_amp}</div>
      <div class="lbl">Amplifications</div>
    </div>
    <div class="card" style="border-top:4px solid #27ae60">
      <div class="num" style="color:#27ae60">{n_gene}</div>
      <div class="lbl">Cancer gene overlaps</div>
    </div>
  </div>

  <h2>CNV Summary Plots</h2>
  <img src="cnv_plots.png" alt="CNV Analysis Plots">

  <h2>CNV Calls — Non-normal Segments ({len(non_normal)} events)</h2>
  <table>
    <thead><tr>
      <th>Location</th><th>Size</th><th>Windows</th>
      <th>Log2 ratio</th><th>Est. CN</th><th>Type</th>
      <th>Genes</th><th>Clinical significance</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="ref">
    <strong>Methods:</strong><br>
    <strong>Coverage normalisation:</strong> Median normalisation corrects
    for differences in sequencing depth between tumour and normal samples.
    Log2 ratio = log2(tumour_normalised / normal_normalised).<br>
    <strong>Segmentation:</strong> Simplified circular binary segmentation
    (CBS) using sliding-window t-tests to identify statistically significant
    change points (p &lt; 0.01, |t| &gt; 2.0).<br>
    <strong>CNV thresholds:</strong> Deletion log2 &lt; {DEL_THRESHOLD} |
    Duplication log2 &gt; {DUP_THRESHOLD} |
    Amplification log2 &gt; {AMP_THRESHOLD}<br>
    <strong>Reference:</strong> Olshen AB et al. (2004)
    Circular binary segmentation for the analysis of array-based DNA copy
    number data. Biostatistics 5:557–572.
  </div>
</div>
<div class="footer">
  CNV Analysis Pipeline &nbsp;|&nbsp;
  Farhana Sayed Portfolio — Module 2 of 3 &nbsp;|&nbsp;
  Tools: Python · numpy · pandas · scipy · matplotlib
</div>
</body></html>"""

    with open(OUT_HTML, "w") as f:
        f.write(html)
    print(f"  OK: Saved {OUT_HTML}")


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  CNV Analysis Pipeline — Module 2")
    print("  Chromosome: chr20 (GRCh38)")
    print("  Window size: 50kb")
    print("  Method: Log2 ratio + CBS segmentation")
    print("  Author: Farhana Sayed")
    print("=" * 60)

    # Step 1: Simulate coverage
    cov_df = simulate_coverage()

    # Step 2: Calculate log2 ratios
    cov_df = calculate_log2_ratio(cov_df)

    # Step 3: Segment genome
    seg_df = segment_genome(cov_df)

    # Step 4: Call CNVs
    cnv_df = call_cnvs(seg_df)

    # Step 5: Annotate with genes
    cnv_df = annotate_cnvs(cnv_df)

    # Step 6: Plots + report
    plot(cov_df, cnv_df)
    html_report(cnv_df)

    # Save segment table
    seg_df.to_csv(OUT_SEG, index=False)

    # Final summary
    print("\n" + "=" * 60)
    print("  MODULE 2 COMPLETE — CNV ANALYSIS SUMMARY")
    print("=" * 60)

    non_normal = cnv_df[cnv_df["cnv_type"] != "normal"]
    print(f"\n  Total segments    : {len(cnv_df)}")
    print(f"  CNV events        : {len(non_normal)}")

    for cnv_type in ["deletion", "duplication", "amplification"]:
        events = cnv_df[cnv_df["cnv_type"] == cnv_type]
        if len(events):
            print(f"\n  {cnv_type.upper()}S ({len(events)}):")
            for _, e in events.iterrows():
                print(f"    {e['chrom']}:{e['start']:,}-{e['end']:,}"
                      f"  log2={e['mean_log2']:+.2f}"
                      f"  CN={e['estimated_cn']}"
                      f"  genes={e['genes']}")

    print(f"\n  Output files:")
    for f in [OUT_COV, OUT_CSV, OUT_SEG, OUT_PNG, OUT_HTML]:
        kb = os.path.getsize(f) // 1024 if os.path.exists(f) else 0
        print(f"    {f}  ({kb} KB)")
    print()


if __name__ == "__main__":
    main()

