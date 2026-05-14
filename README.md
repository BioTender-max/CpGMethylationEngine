# CpGMethylationEngine

**Whole-Genome Bisulfite Sequencing (WGBS) Analysis Pipeline**

A pure-Python computational engine for DNA methylation analysis from bisulfite sequencing data.

## Features
- CpG island detection (obs/exp CpG ratio, GC content, length filters)
- Differentially methylated region (DMR) calling (t-test + BH FDR, |Δβ| > 0.2)
- Epigenetic clock (Horvath-style: weighted CpG methylation → age prediction)
- Methylation entropy (epiallele heterogeneity per CpG)
- Tissue-specific methylation signatures

## Results
- 500 samples × 5000 CpG sites (normal vs cancer)
- CpG islands detected: 1983 (39.7%)
- Total DMRs: 4052 (hyper: 4041, hypo: 11)
- Epigenetic clock MAE: 27.05 years, r=0.276
- Mean methylation entropy: 1.755 bits

## Usage
```bash
pip install numpy scipy matplotlib
python cpg_methylation_engine.py
```

## Tags
`dna-methylation` `bisulfite-seq` `cpg-island` `dmr` `epigenetic-clock` `wgbs`
