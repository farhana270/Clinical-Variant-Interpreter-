"""
mito_analysis.py
================
Module 3: Mitochondrial Genome Variant Analysis
================================================
Annotates variants in the human mitochondrial genome (chrM, 16,569 bp)
with disease associations, OXPHOS complex classification, heteroplasmy
levels, and haplogroup assignments.

WHY MITOCHONDRIAL ANALYSIS IS CLINICALLY IMPORTANT:
    The mitochondrial genome is a 16,569 bp circular DNA molecule
    present in hundreds to thousands of copies per cell. Unlike the
    nuclear genome, it is:
    - Maternally inherited (no recombination — ideal for ancestry tracing)
    - Present in many copies (heteroplasmy — mixture of WT and mutant)
    - Encodes 13 proteins essential for cellular energy production (OXPHOS)
    - Mutated in many diseases: MELAS, LHON, Leigh syndrome, MERRF

    HETEROPLASMY is the unique feature of mitochondrial genetics:
    A cell can carry a MIX of normal and mutant mtDNA. The ratio
    determines disease severity. When mutant load exceeds a threshold
    (typically 60–90%), the cell can no longer produce enough energy
    and disease manifests. This is unlike nuclear variants where
    you either have or don't have the variant.

OXPHOS COMPLEXES (what they do):
    Complex I  (NADH dehydrogenase)    — MT-ND1 to MT-ND6, MT-ND4L
    Complex III (Cytochrome bc1)       — MT-CYB
    Complex IV  (Cytochrome c oxidase) — MT-CO1, MT-CO2, MT-CO3
    Complex V  (ATP synthase)          — MT-ATP6, MT-ATP8
    + 22 tRNA genes and 2 rRNA genes

Pipeline:
    Step 1 → Create realistic chrM variant dataset (MitoMap-based)
    Step 2 → Map variants to mitochondrial gene regions
    Step 3 → Look up disease associations from MitoMap database
    Step 4 → Classify by OXPHOS complex affected
    Step 5 → Assign haplogroup based on variant signature
    Step 6 → Generate plots and HTML report

Reference:
    Gorman GS et al. (2016) Mitochondrial diseases. Nature Reviews
    Disease Primers 2:16080

Author  : Farhana Sayed
Project : Clinical Variant Interpreter — Module 3 of 3
Output  : mito_variants.csv | mito_plots.png | mito_report.html

Run:
    pip3 install pandas matplotlib numpy --break-system-packages
    python3 scripts/mito_analysis.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime

# ── Output paths ──────────────────────────────────────────
BASE      = os.path.expanduser("~/variant-annotation-pipeline")
OUT_CSV   = os.path.join(BASE, "results", "mito_variants.csv")
OUT_PNG   = os.path.join(BASE, "results", "mito_plots.png")
OUT_HTML  = os.path.join(BASE, "results", "mito_report.html")

os.makedirs(os.path.join(BASE, "results"), exist_ok=True)

# ── Mitochondrial genome length ───────────────────────────
MITO_LENGTH = 16_569   # base pairs in human chrM (GRCh38)

# ── Mitochondrial gene map ─────────────────────────────────
# Format: gene_name: (start, end, complex, function)
# Positions are based on NC_012920.1 (revised Cambridge Reference Sequence)
MITO_GENES = {
    "MT-RNR1": (648,   1601,  "rRNA",       "12S ribosomal RNA — translation"),
    "MT-RNR2": (1671,  3229,  "rRNA",       "16S ribosomal RNA — translation"),
    "MT-ND1":  (3307,  4262,  "Complex I",  "NADH dehydrogenase subunit 1"),
    "MT-ND2":  (4470,  5511,  "Complex I",  "NADH dehydrogenase subunit 2"),
    "MT-CO1":  (5904,  7445,  "Complex IV", "Cytochrome c oxidase subunit 1"),
    "MT-CO2":  (7586,  8269,  "Complex IV", "Cytochrome c oxidase subunit 2"),
    "MT-ATP8": (8366,  8572,  "Complex V",  "ATP synthase subunit 8"),
    "MT-ATP6": (8527,  9207,  "Complex V",  "ATP synthase subunit 6"),
    "MT-CO3":  (9207,  9990,  "Complex IV", "Cytochrome c oxidase subunit 3"),
    "MT-ND3":  (10059, 10404, "Complex I",  "NADH dehydrogenase subunit 3"),
    "MT-ND4L": (10470, 10766, "Complex I",  "NADH dehydrogenase subunit 4L"),
    "MT-ND4":  (10760, 12137, "Complex I",  "NADH dehydrogenase subunit 4"),
    "MT-ND5":  (12337, 14148, "Complex I",  "NADH dehydrogenase subunit 5"),
    "MT-ND6":  (14149, 14673, "Complex I",  "NADH dehydrogenase subunit 6"),
    "MT-CYB":  (14747, 15887, "Complex III","Cytochrome b — electron transfer"),
    "MT-DLOOP":(15888, 16569, "D-loop",     "Displacement loop — replication origin"),
    "MT-DLOOP2":(1,    576,   "D-loop",     "D-loop region — hypervariable"),
    "MT-TL1":  (3230,  3304,  "tRNA",       "tRNA-Leu(UUR) — MELAS hotspot"),
    "MT-TK":   (8295,  8364,  "tRNA",       "tRNA-Lys — MERRF hotspot"),
    "MT-TH":   (12138, 12206, "tRNA",       "tRNA-His"),
    "MT-TP":   (15956, 16023, "tRNA",       "tRNA-Pro"),
}

# ── MitoMap disease database ──────────────────────────────
# Real curated variants from MitoMap (mitomap.org)
# Format: position: (ref, alt, gene, disease, inheritance,
#                    heteroplasmy_threshold, status, severity)
MITOMAP_VARIANTS = {
    3243:  ("A", "G", "MT-TL1",  "MELAS syndrome",
            "Maternal", 60,  "Confirmed",      "Severe"),
    8344:  ("A", "G", "MT-TK",   "MERRF syndrome",
            "Maternal", 85,  "Confirmed",      "Severe"),
    11778: ("G", "A", "MT-ND4",  "LHON — Leber hereditary optic neuropathy",
            "Maternal", 100, "Confirmed",      "Moderate"),
    3460:  ("G", "A", "MT-ND1",  "LHON primary mutation",
            "Maternal", 100, "Confirmed",      "Moderate"),
    14484: ("T", "C", "MT-ND6",  "LHON — lowest penetrance primary mutation",
            "Maternal", 100, "Confirmed",      "Mild"),
    8993:  ("T", "G", "MT-ATP6", "NARP / Leigh syndrome",
            "Maternal", 90,  "Confirmed",      "Severe"),
    8993:  ("T", "C", "MT-ATP6", "NARP — milder phenotype",
            "Maternal", 95,  "Confirmed",      "Moderate"),
    9176:  ("T", "C", "MT-ATP6", "Leigh syndrome",
            "Maternal", 95,  "Confirmed",      "Severe"),
    7445:  ("A", "G", "MT-CO1",  "Sensorineural hearing loss",
            "Maternal", 0,   "Confirmed",      "Mild"),
    1555:  ("A", "G", "MT-RNR1", "Aminoglycoside-induced deafness",
            "Maternal", 0,   "Confirmed",      "Mild"),
    3256:  ("C", "T", "MT-TL1",  "MELAS — secondary mutation",
            "Maternal", 70,  "Probable",       "Moderate"),
    3271:  ("T", "C", "MT-TL1",  "MELAS — secondary mutation",
            "Maternal", 75,  "Probable",       "Moderate"),
    4216:  ("T", "C", "MT-ND1",  "LHON secondary — with 11778",
            "Maternal", 100, "Probable",       "Mild"),
    13513: ("G", "A", "MT-ND5",  "MELAS / Leigh overlap",
            "Maternal", 80,  "Confirmed",      "Severe"),
    15257: ("G", "A", "MT-CYB",  "Exercise intolerance / myopathy",
            "Maternal", 90,  "Probable",       "Mild"),
    15885: ("T", "C", "MT-CYB",  "Cytochrome b deficiency",
            "Maternal", 85,  "Probable",       "Moderate"),
    10197: ("G", "A", "MT-ND3",  "MELAS / Leigh — rare mutation",
            "Maternal", 70,  "Possible",       "Moderate"),
    12315: ("G", "A", "MT-TH",   "Cardiomyopathy / hearing loss",
            "Maternal", 80,  "Probable",       "Mild"),
}

# ── Haplogroup signatures ──────────────────────────────────
# Major haplogroups defined by key diagnostic variants
# Haplogroups trace maternal ancestry and some have disease associations
HAPLOGROUP_SIGNATURES = {
    "H":   {73: ("A","G"), 11719: ("G","A")},
    "L0":  {73: ("A","G"), 263: ("A","G"), 2758: ("A","G")},
    "L1":  {3594: ("C","T"), 4104: ("A","G"), 7256: ("C","T")},
    "L2":  {73: ("A","G"), 150: ("C","T"), 152: ("T","C")},
    "L3":  {769: ("G","A"), 1018: ("G","A"), 16311: ("T","C")},
    "M":   {489: ("T","C"), 10400: ("C","T"), 14783: ("T","C")},
    "N":   {8701: ("A","G"), 9540: ("T","C"), 10873: ("T","C")},
    "R":   {73: ("A","G"), 11719: ("G","A"), 12705: ("C","T")},
    "U":   {73: ("A","G"), 11467: ("A","G"), 12308: ("A","G")},
    "K":   {73: ("A","G"), 9055: ("G","A"), 16224: ("T","C")},
    "J":   {73: ("A","G"), 295: ("G","A"), 462: ("T","C")},
    "T":   {73: ("A","G"), 4216: ("T","C"), 15607: ("A","G")},
    "X":   {73: ("A","G"), 1719: ("G","A"), 6221: ("T","C")},
}

# ── Severity colour map ───────────────────────────────────
SEVERITY_COLORS = {
    "Severe":   "#c0392b",
    "Moderate": "#e67e22",
    "Mild":     "#f1c40f",
    "Benign":   "#27ae60",
    "Unknown":  "#95a5a6",
}

STATUS_COLORS = {
    "Confirmed": "#2980b9",
    "Probable":  "#8e44ad",
    "Possible":  "#95a5a6",
}


# ══════════════════════════════════════════════════════════
# STEP 1 — CREATE CHRM VARIANT DATASET
# ══════════════════════════════════════════════════════════
def create_mito_variants():
    """
    Creates a realistic chrM variant dataset simulating a patient
    sample with a mixture of pathogenic, benign, and haplogroup
    variants — all based on real MitoMap records.

    WHY SIMULATE RATHER THAN DOWNLOAD?
    The 1000 Genomes chrM VCF (~2MB) contains real population variants
    but most are haplogroup-defining benign SNPs. For a clinical
    portfolio demo, using a curated set of known pathogenic variants
    from MitoMap is more educationally valuable and demonstrates
    understanding of the clinical database.

    WHAT IS A REALISTIC PATIENT CHRM PROFILE?
    A real mitochondrial sequencing report contains:
    - Haplogroup-defining variants (benign, in most people)
    - Private variants unique to the patient
    - Potentially pathogenic variants at known disease positions
    - Heteroplasmic variants (present in only a fraction of mtDNA copies)
    """
    print("\n[1/5] Creating mitochondrial variant dataset...")

    # Simulate 25 variants representing a realistic patient chrM profile
    variants = [
        # ── PATHOGENIC DISEASE VARIANTS ────────────────────────────────
        # MELAS — m.3243A>G in MT-TL1 tRNA
        # Most common pathogenic mtDNA mutation (~80% of MELAS cases)
        # Present at 72% heteroplasmy — above 60% threshold for MELAS
        {
            "position":        3243,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    0.72,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # MERRF — m.8344A>G in MT-TK tRNA
        # Causes myoclonic epilepsy with ragged-red fibres
        # At 91% — well above 85% threshold
        {
            "position":        8344,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    0.91,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # LHON primary — m.11778G>A in MT-ND4
        # Most common LHON mutation — causes optic neuropathy
        # Homoplasmic (100%) — characteristic of LHON
        {
            "position":        11778,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    1.00,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # NARP / Leigh syndrome — m.8993T>G in MT-ATP6
        # At 94% heteroplasmy — above 90% threshold for NARP
        {
            "position":        8993,
            "ref":             "T",
            "alt":             "G",
            "heteroplasmy":    0.94,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # Leigh syndrome — m.9176T>C in MT-ATP6
        {
            "position":        9176,
            "ref":             "T",
            "alt":             "C",
            "heteroplasmy":    0.88,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # LHON secondary — m.3460G>A in MT-ND1
        {
            "position":        3460,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    1.00,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # MELAS/Leigh overlap — m.13513G>A in MT-ND5
        {
            "position":        13513,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    0.78,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # Deafness — m.1555A>G in MT-RNR1
        # Homoplasmic — aminoglycoside-induced deafness
        {
            "position":        1555,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "pathogenic",
            "source":          "MitoMap confirmed",
        },
        # ── PROBABLE PATHOGENIC VARIANTS ───────────────────────────────
        # MELAS secondary mutation
        {
            "position":        3256,
            "ref":             "C",
            "alt":             "T",
            "heteroplasmy":    0.68,
            "variant_type":    "probable_pathogenic",
            "source":          "MitoMap probable",
        },
        # CYB deficiency — exercise intolerance
        {
            "position":        15257,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    0.88,
            "variant_type":    "probable_pathogenic",
            "source":          "MitoMap probable",
        },
        # Cardiomyopathy
        {
            "position":        12315,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    0.82,
            "variant_type":    "probable_pathogenic",
            "source":          "MitoMap probable",
        },
        # LHON secondary
        {
            "position":        4216,
            "ref":             "T",
            "alt":             "C",
            "heteroplasmy":    1.00,
            "variant_type":    "probable_pathogenic",
            "source":          "MitoMap probable",
        },
        # ── VUS (UNCERTAIN SIGNIFICANCE) ───────────────────────────────
        {
            "position":        10197,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    0.62,
            "variant_type":    "VUS",
            "source":          "MitoMap possible",
        },
        {
            "position":        15885,
            "ref":             "T",
            "alt":             "C",
            "heteroplasmy":    0.71,
            "variant_type":    "VUS",
            "source":          "MitoMap possible",
        },
        # ── HAPLOGROUP-DEFINING BENIGN VARIANTS ────────────────────────
        # These are normal population variants that define
        # maternal ancestry lineage — NOT disease-causing
        {
            "position":        73,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        263,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        750,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        1438,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        4769,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        7028,
            "ref":             "C",
            "alt":             "T",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        8860,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        {
            "position":        15326,
            "ref":             "A",
            "alt":             "G",
            "heteroplasmy":    1.00,
            "variant_type":    "haplogroup",
            "source":          "Haplogroup H defining",
        },
        # ── PRIVATE VARIANTS (novel, not in MitoMap) ───────────────────
        {
            "position":        6062,
            "ref":             "G",
            "alt":             "A",
            "heteroplasmy":    0.34,
            "variant_type":    "private",
            "source":          "Novel variant",
        },
        {
            "position":        14832,
            "ref":             "C",
            "alt":             "T",
            "heteroplasmy":    0.18,
            "variant_type":    "private",
            "source":          "Novel variant — low heteroplasmy",
        },
        {
            "position":        16519,
            "ref":             "T",
            "alt":             "C",
            "heteroplasmy":    1.00,
            "variant_type":    "benign",
            "source":          "Common benign polymorphism",
        },
    ]

    df = pd.DataFrame(variants)
    df = df.sort_values("position").reset_index(drop=True)

    print(f"  OK: {len(df)} chrM variants created")
    for vtype in ["pathogenic","probable_pathogenic","VUS",
                  "haplogroup","private","benign"]:
        n = len(df[df["variant_type"] == vtype])
        if n:
            print(f"    {vtype:<22} : {n}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 2 — MAP VARIANTS TO MITOCHONDRIAL GENES
# ══════════════════════════════════════════════════════════
def map_to_genes(df):
    """
    Maps each variant position to the mitochondrial gene it falls in.

    The mitochondrial genome is densely packed — nearly every base pair
    belongs to a gene, unlike the nuclear genome where most sequence
    is intergenic. Genes are also overlapping in some regions
    (MT-ATP8 and MT-ATP6 overlap by 46bp).

    We check every gene's start/end coordinates and assign the gene
    with the most specific match (smallest gene wins for overlaps).
    """
    print("\n[2/5] Mapping variants to mitochondrial genes...")

    genes_col    = []
    complex_col  = []
    function_col = []

    for _, row in df.iterrows():
        pos = row["position"]
        matched_gene    = "MT-DLOOP"
        matched_complex = "D-loop"
        matched_func    = "Non-coding control region"
        matched_size    = MITO_LENGTH  # start with largest possible

        for gene, (start, end, oxphos, func) in MITO_GENES.items():
            if start <= pos <= end:
                gene_size = end - start
                # Prefer the smallest (most specific) overlapping gene
                if gene_size < matched_size:
                    matched_gene    = gene
                    matched_complex = oxphos
                    matched_func    = func
                    matched_size    = gene_size

        genes_col.append(matched_gene)
        complex_col.append(matched_complex)
        function_col.append(matched_func)

    df["gene"]     = genes_col
    df["complex"]  = complex_col
    df["function"] = function_col

    print(f"  OK: Variants mapped to {df['gene'].nunique()} unique genes")
    print(f"  OK: Complexes affected: "
          f"{', '.join(df['complex'].unique())}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 3 — MITOMAP DISEASE LOOKUP
# ══════════════════════════════════════════════════════════
def lookup_mitomap(df):
    """
    Looks up each variant in the MitoMap disease database.

    MitoMap (mitomap.org) is the primary curated database for
    human mitochondrial DNA variants. It contains:
    - Confirmed pathogenic variants with published clinical evidence
    - Probable pathogenic variants with limited evidence
    - Possible pathogenic variants with preliminary evidence
    - Population polymorphisms (benign)

    For each variant we extract:
    - Disease association (e.g., MELAS, LHON, Leigh syndrome)
    - Disease status (Confirmed / Probable / Possible)
    - Severity classification
    - Heteroplasmy threshold required for disease manifestation
    """
    print("\n[3/5] Looking up MitoMap disease associations...")

    disease_col   = []
    status_col    = []
    severity_col  = []
    threshold_col = []
    mitomap_col   = []

    for _, row in df.iterrows():
        pos = row["position"]
        ref = row["ref"]
        alt = row["alt"]

        if pos in MITOMAP_VARIANTS:
            m_ref, m_alt, m_gene, disease, inherit, threshold, status, sev \
                = MITOMAP_VARIANTS[pos]
            if m_ref == ref and m_alt == alt:
                disease_col.append(disease)
                status_col.append(status)
                severity_col.append(sev)
                threshold_col.append(threshold)
                mitomap_col.append("Yes")
            else:
                # Position matches but allele is different
                disease_col.append("Different allele at known disease position")
                status_col.append("Review")
                severity_col.append("Unknown")
                threshold_col.append(None)
                mitomap_col.append("Partial")
        else:
            disease_col.append("Not in MitoMap — novel variant")
            status_col.append("Novel")
            severity_col.append("Unknown")
            threshold_col.append(None)
            mitomap_col.append("No")

    df["disease"]              = disease_col
    df["mitomap_status"]       = status_col
    df["severity"]             = severity_col
    df["heteroplasmy_threshold"] = threshold_col
    df["in_mitomap"]           = mitomap_col

    # Flag variants above their disease threshold
    df["above_threshold"] = df.apply(
        lambda r: (
            r["in_mitomap"] == "Yes" and
            r["heteroplasmy_threshold"] is not None and
            r["heteroplasmy"] * 100 >= r["heteroplasmy_threshold"]
        ),
        axis=1
    )

    confirmed_disease = df[
        (df["mitomap_status"] == "Confirmed") & df["above_threshold"]
    ]
    print(f"  OK: {len(confirmed_disease)} variants above disease threshold")
    print(f"  OK: MitoMap confirmed pathogenic: "
          f"{len(df[df['mitomap_status']=='Confirmed'])}")
    return df


# ══════════════════════════════════════════════════════════
# STEP 4 — HAPLOGROUP ASSIGNMENT
# ══════════════════════════════════════════════════════════
def assign_haplogroup(df):
    """
    Assigns a mitochondrial haplogroup based on variant signatures.

    WHAT IS A HAPLOGROUP?
    Because the mitochondrial genome is inherited without recombination,
    all humans share a maternal lineage traceable back to a single
    ancestral sequence. Over thousands of years, specific mutations
    accumulated in different populations, creating distinct lineages
    called haplogroups.

    Major haplogroups by ancestry:
    - L haplogroups: African origin (oldest — all humans trace back here)
    - M, N: Out-of-Africa migration
    - H, U, K, J, T, X: European ancestry
    - M, C, D, G, Z: Asian ancestry
    - A, B, C, D: Native American ancestry

    CLINICAL RELEVANCE:
    - Haplogroup J is associated with LHON — increases penetrance
    - Haplogroup H is associated with better response to some treatments
    - African haplogroups (L0, L1) have different disease penetrance
    - Essential for interpreting pathogenic variants in ancestry context

    ALGORITHM:
    Counts how many signature variants of each haplogroup are present
    in the patient's chrM. The haplogroup with the most matching
    signatures is assigned.
    """
    print("\n[4/5] Assigning mitochondrial haplogroup...")

    # Build a set of (position, ref, alt) from patient variants
    patient_variants = set(
        zip(df["position"], df["ref"], df["alt"])
    )

    # Score each haplogroup by number of matching signature variants
    scores = {}
    for hg, signatures in HAPLOGROUP_SIGNATURES.items():
        score = 0
        for pos, (ref, alt) in signatures.items():
            if (pos, ref, alt) in patient_variants:
                score += 1
        scores[hg] = score

    # Best matching haplogroup
    best_hg    = max(scores, key=scores.get)
    best_score = scores[best_hg]
    total_sigs = len(HAPLOGROUP_SIGNATURES[best_hg])

    print(f"  OK: Assigned haplogroup: {best_hg} "
          f"({best_score}/{total_sigs} signatures matched)")

    # Haplogroup clinical notes
    hg_notes = {
        "H":  "Most common European haplogroup (~40–50% Europeans). "
              "Associated with better prognosis in some mito diseases.",
        "J":  "European haplogroup associated with increased LHON penetrance.",
        "U":  "Ancient European haplogroup — increased risk in some populations.",
        "K":  "Descended from haplogroup U — European ancestry.",
        "L0": "Oldest human haplogroup — sub-Saharan African origin.",
        "L3": "African haplogroup — ancestor of all non-African lineages.",
        "M":  "Major Out-of-Africa haplogroup — South/East Asian ancestry.",
    }
    note = hg_notes.get(best_hg, "Population-specific haplogroup.")

    print(f"  OK: {note}")
    return best_hg, note, scores


# ══════════════════════════════════════════════════════════
# STEP 5 — GENERATE PLOTS
# ══════════════════════════════════════════════════════════
def plot(df, haplogroup, hg_note):
    """
    Generates a 4-panel mitochondrial variant summary figure.

    Panel 1: Circular chrM map with variant positions and severity
    Panel 2: Heteroplasmy levels for pathogenic variants
    Panel 3: OXPHOS complex distribution
    Panel 4: Disease association summary
    """
    print("\n[5a/5] Generating plots...")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Mitochondrial Genome Variant Analysis — chrM (16,569 bp)\n"
        f"Haplogroup: {haplogroup}  |  GRCh38  |  "
        f"Author: Farhana Sayed",
        fontsize=11, fontweight="bold", y=1.01
    )

    # ── Panel 1: chrM linear map with variant positions ───
    ax = axes[0, 0]

    # Draw chromosome backbone
    ax.barh(0, MITO_LENGTH, left=0, height=0.3,
            color="#ecf0f1", edgecolor="#bdc3c7", linewidth=1)

    # Draw gene blocks
    gene_y_offset = 0
    gene_colors = {
        "Complex I":   "#3498db",
        "Complex III": "#e74c3c",
        "Complex IV":  "#2ecc71",
        "Complex V":   "#f39c12",
        "rRNA":        "#9b59b6",
        "tRNA":        "#1abc9c",
        "D-loop":      "#95a5a6",
    }
    for gene, (start, end, oxphos, _) in MITO_GENES.items():
        if gene.startswith("MT-DLOOP2"):
            continue
        color = gene_colors.get(oxphos, "#888")
        ax.barh(gene_y_offset, end - start, left=start,
                height=0.25, color=color, alpha=0.6, edgecolor="none")

    # Plot variants as vertical lines coloured by severity
    sev_colors = {
        "pathogenic":          "#c0392b",
        "probable_pathogenic": "#e67e22",
        "VUS":                 "#f1c40f",
        "haplogroup":          "#27ae60",
        "private":             "#95a5a6",
        "benign":              "#2980b9",
    }
    for _, row in df.iterrows():
        color = sev_colors.get(row["variant_type"], "#888")
        ax.vlines(row["position"], 0.15, 0.55,
                  color=color, linewidth=2, alpha=0.85)

    # Legend for gene colours
    gene_patches = [
        mpatches.Patch(color=c, label=k, alpha=0.7)
        for k, c in gene_colors.items()
        if k != "D-loop"
    ]
    ax.legend(handles=gene_patches, fontsize=7,
              loc="upper right", ncol=2, title="OXPHOS complex")

    ax.set_xlim(0, MITO_LENGTH)
    ax.set_ylim(-0.2, 0.8)
    ax.set_xlabel("chrM position (bp)", fontsize=10)
    ax.set_title(f"chrM variant map — haplogroup {haplogroup}",
                 fontweight="bold")
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    # ── Panel 2: Heteroplasmy levels ──────────────────────
    ax = axes[0, 1]
    path_vars = df[df["variant_type"].isin(
        ["pathogenic", "probable_pathogenic", "VUS"]
    )].copy().sort_values("heteroplasmy", ascending=True)

    if len(path_vars) > 0:
        labels = [f"m.{int(r['position'])}{r['ref']}>{r['alt']}\n"
                  f"({r['gene']})"
                  for _, r in path_vars.iterrows()]
        het_vals  = path_vars["heteroplasmy"].values * 100
        bar_colors = [sev_colors.get(vt, "#888")
                      for vt in path_vars["variant_type"]]

        bars = ax.barh(range(len(path_vars)), het_vals,
                       color=bar_colors, edgecolor="white", height=0.7)
        ax.set_yticks(range(len(path_vars)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Heteroplasmy level (%)", fontsize=10)
        ax.set_title("Heteroplasmy levels — disease variants",
                     fontweight="bold")
        ax.set_xlim(0, 110)
        ax.axvline(x=60, color="#c0392b", linewidth=1,
                   linestyle="--", alpha=0.5, label="~MELAS threshold (60%)")
        ax.axvline(x=90, color="#e67e22", linewidth=1,
                   linestyle="--", alpha=0.5, label="~NARP threshold (90%)")
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        for bar, val in zip(bars, het_vals):
            ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%", va="center", fontsize=8)

    # ── Panel 3: OXPHOS complex breakdown ─────────────────
    ax = axes[1, 0]
    complex_counts = df["complex"].value_counts()
    complex_order = ["Complex I", "Complex III", "Complex IV",
                     "Complex V", "tRNA", "rRNA", "D-loop"]
    complex_order = [c for c in complex_order if c in complex_counts.index]
    counts = [complex_counts.get(c, 0) for c in complex_order]
    colors = [gene_colors.get(c, "#888") for c in complex_order]

    bars = ax.bar(range(len(complex_order)), counts,
                  color=colors, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(complex_order)))
    ax.set_xticklabels(complex_order, rotation=20, ha="right", fontsize=9)
    ax.set_title("Variants by OXPHOS complex", fontweight="bold")
    ax.set_ylabel("Number of variants")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val), ha="center", fontsize=9, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # ── Panel 4: Disease summary ───────────────────────────
    ax = axes[1, 1]
    confirmed = df[df["mitomap_status"] == "Confirmed"]
    diseases = confirmed["disease"].value_counts().head(8)

    if len(diseases) > 0:
        dis_colors = [
            SEVERITY_COLORS.get(
                confirmed[confirmed["disease"] == d]["severity"].values[0],
                "#888"
            )
            for d in diseases.index
        ]
        ax.barh(diseases.index[::-1], diseases.values[::-1],
                color=dis_colors[::-1], edgecolor="white", height=0.6)
        ax.set_title("MitoMap confirmed disease associations",
                     fontweight="bold")
        ax.set_xlabel("Number of variants")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        patches = [
            mpatches.Patch(color=SEVERITY_COLORS["Severe"],   label="Severe"),
            mpatches.Patch(color=SEVERITY_COLORS["Moderate"], label="Moderate"),
            mpatches.Patch(color=SEVERITY_COLORS["Mild"],     label="Mild"),
        ]
        ax.legend(handles=patches, fontsize=8)
    else:
        ax.text(0.5, 0.5, "No confirmed disease associations",
                ha="center", va="center")
        ax.set_title("MitoMap confirmed disease associations",
                     fontweight="bold")

    plt.tight_layout(pad=2.5)
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  OK: Saved {OUT_PNG}")


# ══════════════════════════════════════════════════════════
# STEP 6 — HTML REPORT
# ══════════════════════════════════════════════════════════
def html_report(df, haplogroup, hg_note):
    """Generates the portfolio HTML report for Module 3."""
    print("\n[5b/5] Generating HTML report...")

    n_path    = len(df[df["variant_type"] == "pathogenic"])
    n_prob    = len(df[df["variant_type"] == "probable_pathogenic"])
    n_vus     = len(df[df["variant_type"] == "VUS"])
    n_above   = len(df[df["above_threshold"] == True])

    def sev_badge(status):
        c = STATUS_COLORS.get(status, "#95a5a6")
        return (f'<span style="background:{c};color:white;padding:2px 8px;'
                f'border-radius:10px;font-size:11px;font-weight:bold">'
                f'{status}</span>')

    def vtype_badge(vtype):
        colors = {
            "pathogenic":          "#c0392b",
            "probable_pathogenic": "#e67e22",
            "VUS":                 "#f1c40f",
            "haplogroup":          "#27ae60",
            "private":             "#7f8c8d",
            "benign":              "#2980b9",
        }
        c = colors.get(vtype, "#888")
        return (f'<span style="background:{c};color:white;padding:2px 8px;'
                f'border-radius:10px;font-size:11px;font-weight:bold">'
                f'{vtype.replace("_"," ").upper()}</span>')

    rows_html = ""
    for _, row in df.iterrows():
        het_pct = round(row["heteroplasmy"] * 100, 1)
        thresh  = row.get("heteroplasmy_threshold", None)
        thresh_str = f"{thresh}%" if thresh else "N/A"
        above_str  = (
            '<span style="color:#c0392b;font-weight:bold">YES ⚠</span>'
            if row["above_threshold"]
            else '<span style="color:#27ae60">No</span>'
        )
        rows_html += f"""<tr>
          <td><code>m.{int(row['position'])}{row['ref']}>{row['alt']}</code></td>
          <td><strong>{row['gene']}</strong></td>
          <td>{row['complex']}</td>
          <td>{het_pct}%</td>
          <td>{thresh_str}</td>
          <td>{above_str}</td>
          <td>{vtype_badge(row['variant_type'])}</td>
          <td>{sev_badge(row['mitomap_status'])}</td>
          <td style="font-size:11px">{row['disease']}</td>
          <td style="font-size:10px">{row['severity']}</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8">
<title>Mitochondrial Variant Report — Farhana Sayed</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:'Segoe UI',sans-serif;background:#f4f6fb;color:#1a1a2e}}
.hdr{{background:linear-gradient(135deg,#1b4332,#081c15);
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
.hg-box{{background:white;border-radius:10px;padding:20px;
          border-left:5px solid #1b4332;margin-bottom:20px;
          box-shadow:0 2px 8px rgba(0,0,0,.06)}}
.hg-box h3{{font-size:15px;color:#1b4332;margin-bottom:6px}}
.hg-box p{{font-size:13px;color:#555;line-height:1.6}}
h2{{font-size:15px;color:#1b4332;border-left:4px solid #52b788;
    padding-left:12px;margin:22px 0 12px}}
img{{width:100%;border-radius:10px;
     box-shadow:0 2px 10px rgba(0,0,0,.08);margin-bottom:22px}}
table{{width:100%;border-collapse:collapse;font-size:11.5px;
       background:white;border-radius:8px;overflow:hidden;
       box-shadow:0 2px 8px rgba(0,0,0,.06)}}
th{{background:#1b4332;color:white;padding:9px 8px;
    text-align:left;font-size:11px}}
td{{padding:7px 8px;border-bottom:1px solid #eee;vertical-align:middle}}
tr:nth-child(even){{background:#f8f9fd}}
tr:hover{{background:#eef2ff}}
code{{font-size:11px;background:#eef;padding:1px 5px;border-radius:3px}}
.ref{{background:white;border-radius:8px;padding:18px;
      border-left:4px solid #1b4332;margin-top:16px;
      font-size:12.5px;line-height:1.8;color:#333}}
.footer{{background:#1b4332;color:white;text-align:center;
         padding:16px;font-size:12px;margin-top:22px}}
</style></head><body>
<div class="hdr">
  <h1>Mitochondrial Genome Variant Analysis Report</h1>
  <p>Genome: chrM (16,569 bp, NC_012920.1) &nbsp;|&nbsp;
     Database: MitoMap &nbsp;|&nbsp;
     Assembly: GRCh38 &nbsp;|&nbsp;
     Generated: {datetime.now().strftime("%d %b %Y %H:%M")}</p>
  <p style="margin-top:5px">Author: Farhana Sayed &nbsp;|&nbsp;
     B.Tech Bioinformatics &amp; Data Science, D.Y. Patil
     &nbsp;|&nbsp; Module 3 of 3 — Clinical Variant Interpreter</p>
</div>
<div class="body">
  <div class="cards">
    <div class="card" style="border-top:4px solid #c0392b">
      <div class="num" style="color:#c0392b">{n_path}</div>
      <div class="lbl">Confirmed pathogenic</div>
    </div>
    <div class="card" style="border-top:4px solid #e67e22">
      <div class="num" style="color:#e67e22">{n_prob}</div>
      <div class="lbl">Probable pathogenic</div>
    </div>
    <div class="card" style="border-top:4px solid #f1c40f">
      <div class="num" style="color:#c8a000">{n_vus}</div>
      <div class="lbl">VUS</div>
    </div>
    <div class="card" style="border-top:4px solid #c0392b">
      <div class="num" style="color:#c0392b">{n_above}</div>
      <div class="lbl">Above disease threshold</div>
    </div>
  </div>

  <div class="hg-box">
    <h3>Haplogroup Assignment: {haplogroup}</h3>
    <p>{hg_note}</p>
  </div>

  <h2>Variant Summary Plots</h2>
  <img src="mito_plots.png" alt="Mitochondrial Variant Plots">

  <h2>All Mitochondrial Variants ({len(df)} total)</h2>
  <table>
    <thead><tr>
      <th>Variant (HGVS)</th><th>Gene</th><th>OXPHOS Complex</th>
      <th>Heteroplasmy</th><th>Disease threshold</th>
      <th>Above threshold</th><th>Type</th>
      <th>MitoMap status</th><th>Disease association</th><th>Severity</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
  </table>

  <div class="ref">
    <strong>Key findings:</strong><br>
    m.3243A&gt;G (MT-TL1) at 72% heteroplasmy — above MELAS threshold (60%).
    Most common pathogenic mtDNA variant worldwide.<br>
    m.8344A&gt;G (MT-TK) at 91% — above MERRF threshold (85%). Causes
    myoclonic epilepsy with ragged-red fibres.<br>
    m.11778G&gt;A (MT-ND4) at 100% — homoplasmic LHON primary mutation.
    Causes bilateral visual loss typically in young adults.<br>
    m.8993T&gt;G (MT-ATP6) at 94% — above NARP threshold (90%).
    Neurogenic weakness with ataxia and retinitis pigmentosa.<br><br>
    <strong>Database:</strong> MitoMap (mitomap.org) — Comprehensive
    Human Mitochondrial Genome Database.<br>
    <strong>Reference:</strong> Gorman GS et al. (2016) Mitochondrial
    diseases. Nature Reviews Disease Primers 2:16080.
  </div>
</div>
<div class="footer">
  Mitochondrial Variant Pipeline &nbsp;|&nbsp;
  Farhana Sayed Portfolio — Module 3 of 3 &nbsp;|&nbsp;
  Tools: Python · pandas · matplotlib · MitoMap database
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
    print("  Mitochondrial Variant Analysis — Module 3")
    print("  Genome: chrM (16,569 bp) | GRCh38")
    print("  Database: MitoMap curated variants")
    print("  Author: Farhana Sayed")
    print("=" * 60)

    # Step 1: Create variant dataset
    df = create_mito_variants()

    # Step 2: Map to genes
    df = map_to_genes(df)

    # Step 3: MitoMap lookup
    df = lookup_mitomap(df)

    # Step 4: Haplogroup assignment
    haplogroup, hg_note, scores = assign_haplogroup(df)

    # Step 5: Plots + report
    plot(df, haplogroup, hg_note)
    html_report(df, haplogroup, hg_note)

    # Save CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"  OK: Saved {OUT_CSV}")

    # Final summary
    print("\n" + "=" * 60)
    print("  MODULE 3 COMPLETE — MITOCHONDRIAL ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"\n  Total variants      : {len(df)}")
    print(f"  Haplogroup          : {haplogroup}")

    for vtype in ["pathogenic","probable_pathogenic","VUS",
                  "haplogroup","private","benign"]:
        n = len(df[df["variant_type"] == vtype])
        if n:
            print(f"  {vtype:<24} : {n}")

    above = df[df["above_threshold"] == True]
    if len(above) > 0:
        print(f"\n  VARIANTS ABOVE DISEASE THRESHOLD ({len(above)}):")
        for _, r in above.iterrows():
            print(f"    m.{int(r['position'])}{r['ref']}>{r['alt']}"
                  f"  {r['gene']}"
                  f"  het={r['heteroplasmy']*100:.0f}%"
                  f"  threshold={r['heteroplasmy_threshold']}%"
                  f"  disease={r['disease'][:40]}")

    print(f"\n  Output files:")
    for f in [OUT_CSV, OUT_PNG, OUT_HTML]:
        kb = os.path.getsize(f) // 1024 if os.path.exists(f) else 0
        print(f"    {f}  ({kb} KB)")
    print()


if __name__ == "__main__":
    main()
