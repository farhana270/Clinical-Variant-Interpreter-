# Clinical Variant Interpreter

> A three-module bioinformatics pipeline covering the full spectrum of
> genomic variant analysis used in clinical genetics — SNV classification,
> copy number detection, and mitochondrial genome annotation.

Built to demonstrate core skills for **Genome Variant Analyst** roles:
ACMG/AMP classification · CNV detection · Mitochondrial variant analysis ·
Python pipelines · Clinical database integration · Reproducible reporting

---

## The Three Modules at a Glance

| Module | Focus | Variants | Genes / Regions | Key output |
|---|---|---|---|---|
| 1 — ACMG SNV | Clinical SNV classification | 20 variants | BRCA1/2, TP53, MLH1, MSH2, PALB2, CHEK2 | 5-tier ACMG classification |
| 2 — CNV Analysis | Copy number detection from coverage | 7 CNV events | ASXL1, PTPRT, AURKA, BCAS1, ZNF217, NCOA3 | Genome-wide CNV profile |
| 3 — Mitochondrial | chrM variant annotation | 25 variants | 13 OXPHOS genes + tRNA + rRNA | Disease + heteroplasmy report |

---

## Project Structure

```
clinical-variant-interpreter/
│
├── scripts/
│   ├── create_and_classify.py    ← Module 1: ACMG/AMP SNV classification
│   ├── cnv_analysis.py           ← Module 2: CNV detection from coverage
│   └── mito_analysis.py          ← Module 3: Mitochondrial variant analysis
│
├── results/
│   ├── acmg_report.html          ← Module 1 report
│   ├── acmg_plots.png
│   ├── acmg_classified_variants.csv
│   ├── cnv_report.html           ← Module 2 report
│   ├── cnv_plots.png
│   ├── cnv_calls.csv
│   ├── mito_report.html          ← Module 3 report
│   ├── mito_plots.png
│   └── mito_variants.csv
│
└── README.md
```

---

## How to Run

```bash
# Install all dependencies
pip3 install pandas matplotlib numpy scipy --break-system-packages

# Clone repository
git clone https://github.com/farhanasayed/clinical-variant-interpreter
cd clinical-variant-interpreter

# Run Module 1 — ACMG SNV classification
python3 scripts/create_and_classify.py

# Run Module 2 — CNV detection
python3 scripts/cnv_analysis.py

# Run Module 3 — Mitochondrial analysis
python3 scripts/mito_analysis.py
```

Open any `.html` file in `results/` in your browser for the full interactive report.

---

## Module 1 — ACMG/AMP Variant Classification

### What it does

Classifies 20 clinically significant variants across 7 hereditary cancer
predisposition genes using the ACMG/AMP 2015 evidence framework.

### Why it matters clinically

Every variant identified in a patient's sequencing report must be
assigned one of five clinical significance tiers before it can be
reported to a clinician. This process — variant interpretation — is the
core responsibility of a Genome Variant Analyst. A Pathogenic or Likely
Pathogenic call triggers genetic counselling and clinical management
decisions. A Benign call provides reassurance. A VUS requires ongoing
monitoring as evidence accumulates.

### Results

| Tier | Count | Example variant |
|---|---|---|
| Pathogenic | 9 | BRCA1 p.Gln1756ProfsTer74 — stop_gained, PVS1+PS1+PM2, score=14 |
| Likely Pathogenic | 5 | BRCA2 p.Ala1708Glu — missense, PS1+PM2+PP2+PP3, score=8 |
| VUS | 1 | BRCA1 p.Ser1613Cys — conflicting in silico, score=4 |
| Likely Benign | 2 | BRCA2 p.Asn289Tyr — tolerated, BS1+BS3+BP4, score=−8 |
| Benign | 3 | TP53 p.Pro72Arg — gnomAD AF=0.38, BA1+BS3+BP4, score=−21 |

### ACMG criteria implemented

| Code | Strength | Rule |
|---|---|---|
| PVS1 | Very Strong (8 pts) | Null variant in loss-of-function gene |
| PS1 | Strong (4 pts) | ClinVar confirmed pathogenic |
| PM1 | Moderate (2 pts) | Missense in mutational hotspot |
| PM2 | Moderate (2 pts) | Absent/ultra-rare in gnomAD (AF < 0.01%) |
| PP2 | Supporting (1 pt) | Missense in low-benign-variation gene |
| PP3 | Supporting (1 pt) | SIFT deleterious + PolyPhen damaging |
| BA1 | Stand-alone (16 pts) | gnomAD AF > 5% — Benign standalone |
| BS3 | Strong (4 pts) | ClinVar Benign |
| BP4 | Supporting (1 pt) | SIFT tolerated + PolyPhen benign |
| BP7 | Supporting (1 pt) | Synonymous variant |

### How the scoring works

```
Score = sum(pathogenic evidence weights) − sum(benign evidence weights)

Pathogenic        : score ≥ 10
Likely Pathogenic : score ≥ 6
VUS               : score 1–5
Likely Benign     : score ≤ −2
Benign            : score ≤ −6  OR  BA1 alone
```

Reference: Richards et al. (2015) Genetics in Medicine 17:405–424
Scoring: Tavtigian et al. (2020) Human Mutation 41:1734–1737

---

## Module 2 — Copy Number Variant Detection

### What it does

Detects deletions, duplications, and amplifications from read depth
coverage data using log2 ratio analysis and circular binary segmentation.

### Why it matters clinically

SNV pipelines only detect single nucleotide changes and small indels.
They completely miss Copy Number Variants — regions where entire segments
of DNA are deleted or duplicated. CNVs account for approximately 15% of
disease-causing variants that are missed by SNV-only analysis. Large
BRCA1 exon deletions, for example, are CNVs that would be reported as
normal by a standard variant calling pipeline.

### Results

| Type | Count | Location | Genes | Log2 | CN | Clinical note |
|---|---|---|---|---|---|---|
| Deletion | 1 | chr20:31.0–31.5Mb | ASXL1 | −1.07 | 1.0 | Tumour suppressor loss — leukaemia |
| Deletion | 1 | chr20:40.7–41.2Mb | PTPRT | −1.16 | 0.9 | Tumour suppressor loss — colorectal |
| Duplication | 1 | chr20:46.0–46.5Mb | NCOA3 | +0.60 | 3.0 | Oncogene gain — breast cancer |
| Duplication | 1 | chr20:50.0–53.0Mb | BCAS1, ZNF217 | +0.80 | 3.5 | 20q gain — breast/ovarian |
| Amplification | 3 | chr20:54.0–56.0Mb | AURKA | +1.76 | 6.8 | 20q13 amplicon — high-level |

### Pipeline steps explained

```
Step 1 — Coverage simulation
    Generates 50kb window read depth for tumour and normal samples
    using Poisson statistics (the correct model for sequencing depth).
    Five real cancer CNV events embedded based on published profiles.

Step 2 — Log2 ratio calculation
    Normalises both samples by their median coverage to remove
    sequencing depth bias. Calculates log2(tumour/normal) per window.
    Normal diploid = 0.0, deletion ≈ −1.0, duplication ≈ +0.58.

Step 3 — Circular Binary Segmentation
    Groups consecutive windows of similar log2 ratio into segments.
    Uses sliding-window t-tests to find statistically significant
    change points (p < 0.01, |t| > 2.0).
    Reference: Olshen et al. (2004) Biostatistics 5:557–572

Step 4 — CNV calling
    Applies clinical thresholds to classify each segment.
    Deletion: log2 < −0.4 | Duplication: log2 > +0.3 | Amp: log2 > +1.0

Step 5 — Gene annotation
    Overlaps CNV coordinates with cancer gene positions using the
    standard interval test: CNV_start < gene_end AND CNV_end > gene_start
```

---

## Module 3 — Mitochondrial Genome Variant Analysis

### What it does

Annotates 25 variants in the human mitochondrial genome (chrM, 16,569 bp)
with disease associations from MitoMap, OXPHOS complex classification,
heteroplasmy level analysis, and haplogroup assignment.

### Why it matters clinically

Mitochondrial variants are fundamentally different from nuclear variants:

**Heteroplasmy** — unlike nuclear DNA where you have exactly 2 copies,
each cell contains hundreds to thousands of mtDNA molecules. A cell
can carry a mixture of normal and mutant mtDNA. The percentage of mutant
mtDNA — called heteroplasmy — determines disease severity. The m.3243A>G
MELAS mutation causes disease only when heteroplasmy exceeds ~60%.
Below this threshold, sufficient normal mtDNA compensates and the cell
functions normally.

**OXPHOS diseases** — the 13 protein-coding genes in mtDNA all encode
subunits of the oxidative phosphorylation (OXPHOS) system — the cellular
machinery that produces ATP (energy). Mutations cause energy failure in
high-demand tissues: brain, muscle, heart — the organs affected in
MELAS, LHON, Leigh syndrome, and MERRF.

**Maternal inheritance** — mtDNA passes exclusively from mother to child
with no recombination. Haplogroup analysis traces maternal ancestry and
is clinically relevant because some haplogroups modify disease penetrance
(haplogroup J increases LHON penetrance).

### Results

| Finding | Detail |
|---|---|
| Haplogroup | H (most common European haplogroup) |
| Confirmed pathogenic | 8 variants in MitoMap |
| Above disease threshold | 6 variants exceeding heteroplasmy threshold |
| Diseases detected | MELAS, MERRF, LHON, NARP, Leigh syndrome, deafness |
| OXPHOS complexes affected | Complex I, Complex IV, Complex V, tRNA |

### Key variant findings

```
m.3243A>G  MT-TL1   heteroplasmy=72%  threshold=60%  → MELAS CONFIRMED
m.8344A>G  MT-TK    heteroplasmy=91%  threshold=85%  → MERRF CONFIRMED
m.11778G>A MT-ND4   heteroplasmy=100% threshold=100% → LHON CONFIRMED
m.8993T>G  MT-ATP6  heteroplasmy=94%  threshold=90%  → NARP CONFIRMED
m.9176T>C  MT-ATP6  heteroplasmy=88%  threshold=95%  → Below threshold
m.1555A>G  MT-RNR1  heteroplasmy=100% threshold=0%   → Deafness risk
```

### OXPHOS complexes

```
Complex I  (MT-ND1, ND2, ND3, ND4, ND4L, ND5, ND6) — NADH dehydrogenase
Complex III (MT-CYB)                                — Cytochrome bc1
Complex IV  (MT-CO1, CO2, CO3)                     — Cytochrome c oxidase
Complex V   (MT-ATP6, ATP8)                        — ATP synthase
+ 22 tRNA genes + 2 rRNA genes
```

---

## Tools and Technologies

| Tool | Purpose |
|---|---|
| Python 3.8+ | Pipeline orchestration |
| pandas | Data manipulation and DataFrame operations |
| numpy | Numerical computing, log2 transforms, Poisson simulation |
| scipy | Statistical tests — t-tests for CBS segmentation |
| matplotlib | Publication-style 4-panel summary figures |
| MitoMap | Mitochondrial disease database (mitomap.org) |
| ClinVar | SNV clinical significance (Module 1) |
| gnomAD v4 | Population allele frequencies (Module 1) |
| Ensembl VEP v115 | Variant annotation format |

---

## References

1. Richards S, et al. (2015) Standards and guidelines for the interpretation of sequence variants. *Genetics in Medicine* 17:405–424
2. Tavtigian SV, et al. (2020) Fitting a naturally scaled point system to the ACMG/AMP variant classification guidelines. *Human Mutation* 41:1734–1737
3. Olshen AB, et al. (2004) Circular binary segmentation for the analysis of array-based DNA copy number data. *Biostatistics* 5:557–572
4. Gorman GS, et al. (2016) Mitochondrial diseases. *Nature Reviews Disease Primers* 2:16080
5. McLaren W, et al. (2016) The Ensembl Variant Effect Predictor. *Genome Biology* 17:122
6. Karczewski KJ, et al. (2020) The mutational constraint spectrum from variation in 141,456 humans. *Nature* 581:434–443
7. Miller DT, et al. (2010) Consensus statement: chromosomal microarray is a first-tier clinical diagnostic test. *Am J Hum Genet* 86:749–764

---

## Author

**Farhana Sayed**
B.Tech Bioinformatics and Data Science — D.Y. Patil School of Biotechnology and Bioinformatics, Navi Mumbai
Email: farhanasayed27@gmail.com

*This project is a 3-module Clinical Variant Interpreter portfolio targeting Genome Variant Analyst roles.*
