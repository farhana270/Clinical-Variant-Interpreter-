# Clinical Variant Interpreter — ACMG/AMP Classification Pipeline

> Automated classification of genomic variants using the ACMG/AMP 2015 guidelines,  
> covering 20 clinically significant variants across 7 hereditary cancer genes.

---

## What This Project Does

When a clinical genetics lab receives a patient's sequencing result, every variant must be assigned one of five clinical significance categories defined by the American College of Medical Genetics (ACMG). This process — called variant interpretation — is done manually by specialist scientists and typically takes hours per variant.

This pipeline automates that process computationally.

It takes a set of genomic variants annotated with population frequency, molecular consequence, and in silico predictions, then applies the full ACMG/AMP 2015 evidence framework to produce a 5-tier clinical classification for each variant — complete with the specific evidence codes that drove the decision.

---

## Report Output

| Tier | Count | Meaning |
|---|---|---|
| Pathogenic | 9 | Causes disease — clinically actionable |
| Likely Pathogenic | 5 | Probably causes disease — reported to clinicians |
| VUS | 1 | Uncertain significance — needs more evidence |
| Likely Benign | 2 | Probably benign — not clinically actioned |
| Benign | 3 | Confirmed benign — common population variant |

**Genes covered:** BRCA1, BRCA2, TP53, MLH1, MSH2, PALB2, CHEK2

**Standard:** Richards et al. (2015) Genetics in Medicine 17:405–424

**Assembly:** GRCh38

---

## Project Structure

```
clinical-variant-interpreter/
│
├── scripts/
│   └── create_and_classify.py    ← complete pipeline (single file)
│
├── results/
│   ├── clinvar_panel_annotated.txt  ← VEP-format input data created by script
│   ├── acmg_classified_variants.csv ← classified variant table
│   ├── acmg_plots.png               ← 3-panel summary figure
│   └── acmg_report.html             ← full interactive HTML report
│
└── README.md
```

---

## How to Run

```bash
# Install dependencies
pip3 install pandas matplotlib --break-system-packages

# Clone and run
git clone https://github.com/farhanasayed/clinical-variant-interpreter
cd clinical-variant-interpreter
python3 scripts/create_and_classify.py
```

Expected output:
```
[1/5] Creating clinically annotated variant dataset...
  OK: 20 variants created across 7 genes

[2/5] Applying ACMG/AMP 2015 classification rules...
  Pathogenic        : 9  █████████
  Likely Pathogenic : 5  █████
  VUS               : 1  █
  Likely Benign     : 2  ██
  Benign            : 3  ███

[3/5] Saving CSV...
[4/5] Generating plots...
[5/5] Generating HTML report...

PIPELINE COMPLETE
```

---

## Code Architecture — Block by Block

### Block 1 — Imports and Global Constants (lines 1–60)

```python
import os, sys, pandas as pd
import matplotlib
matplotlib.use("Agg")          # headless rendering — no display needed
import matplotlib.pyplot as plt
from datetime import datetime
from collections import Counter
```

**Why Agg backend?** On a Linux server or Ubuntu without a graphical desktop, matplotlib cannot open a window. The `Agg` backend renders plots directly to PNG files without needing any display. Forgetting this line causes the script to crash on headless systems.

```python
LOF_GENES = {"BRCA1","BRCA2","TP53","MLH1","MSH2","PALB2","CHEK2",...}
```

**Why a Python set?** Set membership lookup (`gene in LOF_GENES`) is O(1) — constant time regardless of how many genes are in the set. A list would be O(n) — slower as the list grows. This matters when processing thousands of variants in a real pipeline.

```python
LOF_CONSEQUENCES = ["stop_gained","frameshift_variant","splice_donor_variant",...]
```

**What is a LoF consequence?** Loss-of-Function variants destroy the gene's protein product — a premature stop codon (`stop_gained`) truncates the protein, a frameshift shifts the reading frame causing a nonsense sequence downstream, and splice site variants cause exons to be skipped or retained incorrectly. The ACMG PVS1 criterion (the strongest single piece of pathogenic evidence) only fires when BOTH conditions are true: the gene is known to cause disease via haploinsufficiency AND the variant creates one of these LoF consequences.

---

### Block 2 — create_vep_data() (lines 65–617)

This function creates a realistic clinical variant dataset based on published ClinVar and LOVD records. It deliberately covers all five ACMG tiers and includes representative examples of every major variant type.

```python
cols = [
    "Uploaded_variation","Location","Allele","Consequence",
    "IMPACT","SYMBOL","Gene","Feature_type","Feature",
    "BIOTYPE","HGVSc","HGVSp","Existing_variation",
    "REF_ALLELE","SIFT","PolyPhen","AF","gnomAD_AF",
    "CLIN_SIG","CANONICAL","MANE_SELECT",
    "Amino_acids","Protein_position",
]
```

**Why these exact columns?** These are the standard output columns from Ensembl VEP (Variant Effect Predictor) — the gold-standard annotation tool used by TCGA, Genomics England, and clinical labs globally. By matching this format exactly, the pipeline can process real VEP output files from any project without modification.

**Key fields explained:**

- `HGVSp` — Human Genome Variation Society protein notation. `p.Arg273Cys` means arginine at position 273 changed to cysteine. This is the internationally standardised way to describe protein-level changes in clinical reports.
- `gnomAD_AF` — allele frequency in the Genome Aggregation Database (125,748 exomes + 15,708 genomes from healthy individuals). A frequency of 0.000004 means 4 people per million carry this variant — ultra-rare.
- `MANE_SELECT` — Matched Annotation from NCBI and EMBL-EBI. The single clinically approved transcript per gene agreed by both databases. Used in all clinical reporting since 2022.
- `CLIN_SIG` — ClinVar clinical significance. Expert-curated classifications submitted by clinical labs worldwide.
- `CANONICAL` — whether this is the canonical (standard reference) transcript for the gene.

```python
# Example: BRCA1 stop_gained — causes premature protein termination
{
    "Uploaded_variation": "rs80357906",
    "Consequence":        "stop_gained",         # null variant
    "SYMBOL":             "BRCA1",               # LoF gene
    "HGVSp":              "p.Gln1756ProfsTer74", # stops 74 aa early
    "gnomAD_AF":          "0.000004",            # ultra-rare
    "CLIN_SIG":           "pathogenic",          # ClinVar confirmed
}
```

This single variant triggers PVS1 (null in LoF gene) + PS1 (ClinVar pathogenic) + PM2 (ultra-rare) = 14 points → Pathogenic.

```python
# Saves in VEP tab-delimited format with ## header lines
with open(OUT_TXT, "w") as f:
    f.write("## VEP annotation — Clinical Panel\n")
    f.write("#" + "\t".join(cols) + "\n")  # column header with single #
    for _, row in df.iterrows():
        f.write("\t".join(str(row[c]) for c in cols) + "\n")
```

**Why this format?** The double-hash (`##`) lines are VEP metadata comments. The single-hash (`#`) line is the column header. Everything after is tab-separated data. This is exactly the format Ensembl VEP produces — any downstream tool that reads VEP output will read this file correctly.

---

### Block 3 — apply_acmg() — The Classification Engine (lines 623–778)

This is the scientific core of the pipeline. It loops over every variant and applies ACMG criteria one by one, building an evidence dictionary.

```python
for _, row in df.iterrows():
    evidence = {}   # starts fresh for every variant
```

**Why a fresh dictionary per variant?** Evidence from one variant must never contaminate another. Resetting to `{}` ensures complete independence between classifications.

```python
gene        = str(row.get("SYMBOL", "")).upper()
consequence = str(row.get("Consequence", "")).lower()
af_gn       = pd.to_numeric(row.get("gnomAD_AF", ""), errors="coerce")
```

**Why .upper() and .lower()?** Gene names are case-sensitive in Python set lookup. "brca1" would fail to match "BRCA1" in LOF_GENES. Normalising to uppercase prevents silent misclassification. Consequence terms from VEP are always lowercase, so lowercasing enables reliable substring matching.

**Why pd.to_numeric with errors="coerce"?** If gnomAD_AF is an empty string, direct float conversion would crash. `errors="coerce"` converts invalid values to NaN instead of raising an exception. We then use `pd.notna()` to safely handle missing AF values.

```python
# Parsing SIFT: "deleterious(0.001)" → label + score
import re
sift_match  = re.search(r"^([a-z_]+)", sift_raw)
sift_pred   = sift_match.group(1) if sift_match else ""
sift_score_m = re.search(r"\(([0-9.]+)\)", sift_raw)
sift_score  = float(sift_score_m.group(1)) if sift_score_m else None
```

**Regex explained:**
- `r"^([a-z_]+)"` — from the start of string, capture one or more lowercase letters or underscores. Extracts `deleterious` from `deleterious(0.001)`.
- `r"\(([0-9.]+)\)"` — find parentheses containing digits and decimal points. Extracts `0.001`.

VEP always formats SIFT and PolyPhen as `prediction(score)` — regex is the correct and robust way to parse this.

**The 12 ACMG criteria — what each one checks:**

```python
# PVS1 — Pathogenic Very Strong 1
# Null variant (stop/frameshift/splice) in a gene where
# loss-of-function IS the established disease mechanism
if gene in LOF_GENES and any(c in consequence for c in LOF_CONSEQUENCES):
    evidence["PVS1"] = ("Very Strong", "Pathogenic", "...")
```

PVS1 is the single strongest piece of evidence. It only applies when BOTH conditions hold — the gene causes disease through haploinsufficiency (BRCA1, TP53, MLH1 etc.) AND the variant destroys the gene product. Score: 8 points.

```python
# PS1 — Pathogenic Strong 1
# Same amino acid change at same position as a known pathogenic variant
if "pathogenic" in clinvar and "benign" not in clinvar:
    evidence["PS1"] = ("Strong", "Pathogenic", "...")
```

PS1 uses ClinVar expert curation. If the same variant has been classified Pathogenic by clinical labs worldwide and submitted to ClinVar, that is Strong evidence. Score: 4 points. The `and "benign" not in clinvar` guard handles conflicting classifications like `pathogenic/likely_pathogenic` (pass) vs `pathogenic/benign` (fail).

```python
# PM1 — Pathogenic Moderate 1
# Missense in a mutational hotspot or critical functional domain
if "missense" in consequence and gene in {"TP53","BRCA1","BRCA2",...}:
    evidence["PM1"] = ("Moderate", "Pathogenic", "...")
```

TP53's DNA-binding domain is one of the most studied mutational hotspots in cancer. A missense variant here is moderate pathogenic evidence even without ClinVar data. Score: 2 points.

```python
# PM2 — Pathogenic Moderate 2
# Absent from or extremely rare in population databases
if pd.notna(af) and af < 0.0001:
    evidence["PM2"] = ("Moderate", "Pathogenic", "...")
```

A disease-causing variant cannot be common in healthy populations. gnomAD AF < 0.0001 means fewer than 1 in 10,000 people carry this variant — consistent with a rare disease-causing allele. Score: 2 points.

```python
# PP2 — Pathogenic Supporting 2
# Missense variant in a gene with a low rate of benign missense variation
low_benign = {"BRCA1","BRCA2","TP53","PTEN","MLH1","MSH2","PALB2"}
if "missense" in consequence and gene in low_benign:
    evidence["PP2"] = ("Supporting", "Pathogenic", "...")
```

BRCA1 and BRCA2 are constrained genes — evolution has selected against changes in their protein sequence because they are essential. A missense variant in such a gene is more likely to be damaging than one in a gene that tolerates many amino acid changes. Score: 1 point.

```python
# PP3 — Pathogenic Supporting 3
# Multiple concordant computational tools predict damaging
if sift_pred == "deleterious" and "damaging" in pp_pred:
    evidence["PP3"] = ("Supporting", "Pathogenic", "...")
```

SIFT and PolyPhen are independent algorithms — SIFT uses evolutionary conservation, PolyPhen uses structural modelling. When both independently predict deleterious, it is Supporting Pathogenic evidence. Score: 1 point.

```python
# BA1 — Benign Stand-alone 1
# Allele frequency > 5% in gnomAD — benign by itself
if pd.notna(af) and af > 0.05:
    evidence["BA1"] = ("Stand-alone", "Benign", "...")
    evidence = {k: v for k, v in evidence.items() if v[1] == "Benign"}
```

BA1 is unique — it classifies a variant as Benign by itself, overriding all pathogenic evidence. TP53 Pro72Arg (rs1042522) has AF = 0.38 — 38% of the population carries it. It cannot be a rare disease-causing variant. The dictionary comprehension removes all previously collected pathogenic evidence, keeping only benign criteria. Score: 16 points benign.

```python
# BS3 — Benign Strong 3
# Well-established functional studies show benign effect
if "benign" in clinvar and "pathogenic" not in clinvar:
    evidence["BS3"] = ("Strong", "Benign", "...")
```

ClinVar Benign classifications represent consensus of clinical lab submissions over many years. Score: 4 points benign.

```python
# BP4 — Benign Pathogenic 4
# Computational tools all predict benign/tolerated
if "tolerated" in sift_pred and "benign" in pp_pred:
    evidence["BP4"] = ("Supporting", "Benign", "...")
```

Mirror of PP3 — when SIFT and PolyPhen both predict benign, it is Supporting Benign evidence. Score: 1 point benign.

```python
# BP7 — Benign Supporting 7
# Synonymous variant with no predicted splice effect
if "synonymous" in consequence:
    evidence["BP7"] = ("Supporting", "Benign", "...")
```

Synonymous variants change the DNA but not the amino acid (`Leu151=` notation means leucine stays leucine). Without a splice effect, these are generally benign. Score: 1 point benign.

---

### Block 4 — score_and_classify() (lines 793–824)

```python
WEIGHTS = {
    "Very Strong": 8,   # PVS1
    "Strong":      4,   # PS1-4, BS1-4
    "Moderate":    2,   # PM1-6
    "Supporting":  1,   # PP1-5, BP1-7
    "Stand-alone": 16,  # BA1
}
```

**Why these specific numbers?** The 8/4/2/1 weighting system is a mathematical formalisation from Tavtigian et al. (2020) that maps to the original ACMG classification rules. A Pathogenic call requires two Strong + one Moderate (4+4+2=10) or one Very Strong + two Moderate (8+2+2=12), matching the original ACMG combination rules.

```python
net = path_score - benign_score

if benign_score >= 16:         tier = "Benign"           # BA1 alone
elif path_score >= 10 and benign_score == 0:
                               tier = "Pathogenic"
elif path_score >= 6:          tier = "Likely Pathogenic"
elif benign_score >= 6:        tier = "Likely Benign"
else:                          tier = "VUS"
```

**Classification thresholds explained with real examples from the report:**

| Variant | Criteria | Score | Tier |
|---|---|---|---|
| BRCA1 stop_gained | PVS1(8) + PS1(4) + PM2(2) | 14 | Pathogenic |
| TP53 Arg273Cys | PS1(4) + PM1(2) + PM2(2) + PP2(1) + PP3(1) | 10 | Pathogenic |
| BRCA1 Ala1708Glu | PS1(4) + PM2(2) + PP2(1) + PP3(1) | 8 | Likely Pathogenic |
| BRCA1 Ser1613Cys | PM2(2) + PP2(1) + PP3(1) | 4 | VUS |
| BRCA2 Asn289Tyr | BS1(4) + BS3(4) + BP4(1) | -9 | Likely Benign |
| TP53 Pro72Arg | BA1(16) + BS3(4) + BP4(1) | -21 | Benign |

---

### Block 5 — save_csv() (lines 830–839)

```python
keep = [c for c in [
    "Uploaded_variation","SYMBOL","Consequence","HGVSp",
    "SIFT","PolyPhen","gnomAD_AF","CLIN_SIG",
    "acmg_tier","acmg_score","acmg_codes",
] if c in df.columns]
df[keep].to_csv(OUT_CSV, index=False)
```

**Why the list comprehension guard?** `[c for c in [...] if c in df.columns]` only saves columns that actually exist in the DataFrame. If a column name changes or is missing, the script doesn't crash — it simply skips that column. This is defensive programming that makes the pipeline robust to different input formats.

---

### Block 6 — plot() (lines 845–924)

Three panels telling a complete story:

```python
# Panel 1: ACMG tier distribution
counts = df["acmg_tier"].value_counts().reindex(tier_order, fill_value=0)
```

`.reindex(tier_order, fill_value=0)` ensures all 5 tiers appear in the correct clinical order (Pathogenic → Benign), even if some have zero variants. Without this, tiers with no variants would be absent from the chart, making it look incomplete.

```python
# Panel 2: evidence codes frequency
code_counts = Counter(all_codes).most_common(10)
bar_col = ["#e74c3c" if c.startswith(("PVS","PS","PM","PP")) else "#27ae60"
           for c in codes]
```

Red bars = pathogenic evidence codes, green bars = benign. The colour coding makes it immediately clear which evidence type dominates.

```python
# Panel 3: gene × tier stacked bar
gt = pd.crosstab(df["SYMBOL"], df["acmg_tier"])
bottom = pd.Series([0]*len(gt), index=gt.index, dtype=float)
for tier in present:
    ax.barh(gt.index, gt[tier], left=bottom, color=TIER_COLORS[tier])
    bottom = bottom + gt[tier]  # each tier stacks on top of previous
```

`pd.crosstab` creates a gene × tier count matrix. The stacked bar loop adds each tier's portion starting where the previous one ended, using `left=bottom` as the offset. This gives an instant visual comparison of how each gene's variants distribute across tiers.

---

### Block 7 — html_report() and main() (lines 930–1085)

```python
def badge(tier):
    c = TIER_COLORS.get(tier, "#888")
    return f'<span style="background:{c};color:white;...">{tier}</span>'
```

Each ACMG tier gets a colour-coded badge in the HTML table — red for Pathogenic, orange for Likely Pathogenic, yellow for VUS, green for Benign. This makes the report scannable at a glance, exactly like a real clinical variant report system.

```python
if __name__ == "__main__":
    main()
```

This guard means `main()` only runs when the script is executed directly (`python3 script.py`). If another script imports this file to reuse its functions, `main()` doesn't run automatically. This is standard Python practice for reusable modules.

---

## ACMG Evidence Criteria Reference

| Code | Strength | Direction | Rule |
|---|---|---|---|
| PVS1 | Very Strong | Pathogenic | Null variant in LoF gene |
| PS1 | Strong | Pathogenic | ClinVar pathogenic / same AA change |
| PM1 | Moderate | Pathogenic | Missense in mutational hotspot |
| PM2 | Moderate | Pathogenic | Absent/ultra-rare in gnomAD (AF < 0.01%) |
| PM4 | Moderate | Pathogenic | Protein length change via indel |
| PP2 | Supporting | Pathogenic | Missense in low-benign-variation gene |
| PP3 | Supporting | Pathogenic | SIFT deleterious + PolyPhen damaging |
| BA1 | Stand-alone | Benign | gnomAD AF > 5% |
| BS1 | Strong | Benign | AF higher than expected for disorder |
| BS3 | Strong | Benign | ClinVar benign / functional evidence |
| BP4 | Supporting | Benign | SIFT tolerated + PolyPhen benign |
| BP7 | Supporting | Benign | Synonymous variant |

---

## Biological Context — Why These Genes

All 7 genes in this panel are hereditary cancer predisposition genes:

- **BRCA1 / BRCA2** — Breast and ovarian cancer. Homologous recombination DNA repair. BRCA1/2 mutation carriers have 50–80% lifetime breast cancer risk.
- **TP53** — Li-Fraumeni syndrome. The most frequently mutated gene in all human cancers. Guardian of the genome.
- **MLH1 / MSH2** — Lynch syndrome (hereditary colorectal + endometrial cancer). Mismatch repair deficiency causes microsatellite instability.
- **PALB2** — Partner and localiser of BRCA2. Moderate-to-high breast cancer risk gene.
- **CHEK2** — Cell cycle checkpoint kinase. Moderate breast and colorectal cancer risk.

---

## Connection to Dissertation Research

During my dissertation at **Tata Memorial Centre – ACTREC**, I investigated differentially expressed genes in Asian breast cancer patients using RNA-Seq from TCGA-BRCA. That work focused on the transcriptomic layer — how much each gene is expressed.

This project operates at the **DNA variant layer** — characterising whether specific mutations in these same cancer genes are clinically significant. Together they span the full genomic analysis workflow:

```
DNA variants → Gene expression → Pathway analysis
   (this project)    (TCGA RNA-Seq)    (GSEA / GO terms)
```

---

## Tools and Technologies

| Tool | Version | Purpose |
|---|---|---|
| Python | 3.8+ | Pipeline orchestration |
| pandas | 2.0+ | Data manipulation and DataFrame operations |
| matplotlib | 3.7+ | 3-panel summary plot generation |
| Ensembl VEP | v115 | Variant annotation format (data source) |
| ClinVar | 2026 | Clinical significance evidence (PS1, BS3) |
| gnomAD | v4 | Population allele frequencies (PM2, BA1, BS1) |

---

## References

1. Richards S, et al. (2015) Standards and guidelines for the interpretation of sequence variants. *Genetics in Medicine* 17:405–424. **[The primary ACMG/AMP framework]**
2. Tavtigian SV, et al. (2020) Fitting a naturally scaled point system to the ACMG/AMP variant classification guidelines. *Human Mutation* 41:1734–1737. **[Scoring weights used in this pipeline]**
3. McLaren W, et al. (2016) The Ensembl Variant Effect Predictor. *Genome Biology* 17:122. **[VEP output format]**
4. Karczewski KJ, et al. (2020) The mutational constraint spectrum from variation in 141,456 humans. *Nature* 581:434–443. **[gnomAD database]**
5. Landrum MJ, et al. (2018) ClinVar: improving access to variant interpretations and supporting evidence. *Nucleic Acids Research* 46:D1062–D1067. **[ClinVar database]**

---

## Author

**Farhana Sayed**
B.Tech Bioinformatics and Data Science — D.Y. Patil School of Biotechnology and Bioinformatics, Navi Mumbai
Email: farhanasayed27@gmail.com

*This project is Module 1 of a 3-module Clinical Variant Interpreter portfolio project targeting Genome Variant Analyst roles.*
