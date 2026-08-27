# Reproduction report

Running the pipeline documented in [`README.md`](README.md) on the Yakubovsky
2026 live-donor liver atlas reproduces every reported number and figure of the
manuscript. The tables below give the value produced by each analysis step; each
matches the corresponding value in the manuscript. The final section covers the
second-tissue validation, which runs on a separate dataset (Moor *et al.* 2018).

## Pipeline and axis validation

| Quantity | Reproduced value | Manuscript |
|---|---|---|
| Visium cohort | 16 samples, 47,828 spots (8 live-donor = 18,934; 8 patient = 28,894) | Fig 1 |
| Canonical zonation markers | 9/9, correct in all 8 live donors | Fig 1D |
| Leave-family-out, pooled | 4/4 CYP, 2/2 urea-cycle, 1/1 GLUL | Fig 1E |
| Leave-family-out, per donor | 7/8 CYP (M8), 6/8 urea-cycle (M5, M7 miss CPS1), 8/8 GLUL | Fig S2 |
| Independent-axis check | 8/9 markers recover; CPS1 the traceable miss | Fig 1E |
| Cross-donor gradient correlation | mean off-diagonal r = 0.873; minimum 0.770 (M3-M7) | Fig 1F |
| Cross-donor, relaxed panel | 15,192 genes; mean r = 0.670; minimum r = 0.561 | Results |
| Cross-donor, relaxed top effect-size quartile | 3,798 genes; mean r = 0.722; minimum r = 0.635 | Results |

## Cross-modality reproducibility

| Quantity | Reproduced value | Manuscript |
|---|---|---|
| Cross-platform, donor M2 | 14/14 high-confidence sign agreement; rank ρ = 0.83 | Fig 2A |
| Cross-platform, donor M6 | 13/14; ρ = 0.90 | Fig 2A |
| Cross-platform, pooled | 96.9% sign agreement; ρ = 0.86 | Results |
| Cross-pipeline (slope vs layer difference) | 76.3% sign agreement, n = 3,207 | Fig 2B |
| Cross-pipeline (8-quantile aggregation) | 75.6%, n = 3,220 | Methods |
| Cross-pipeline, top magnitude quartile | 78.8% | Fig S3 |
| Analyst-degrees-of-freedom sweep | 0 sign changes across normalisation, mt-fraction and bin count | Fig S8 |
| Sign-of-correlation baseline | 63/66 agreement; signed-magnitude ρ = 0.96 | Fig 2C |

## Phenomenological coefficient matrix

| Quantity | Reproduced value | Manuscript |
|---|---|---|
| Effective rank | 46/66 at N = 50 bins; 66/66 at N = 100 | Fig 3A |
| Descriptive bootstrap stable-entry fraction | 7.69% at N = 50; range 5.46%-10.35% across the Visium bin sweep | Fig 3B |
| Donor-split symmetric Frobenius distance | mean 1.4108 at N = 50 (IQR 1.3999-1.4223), over all 35 unique 4-vs-4 partitions | Fig 3C |
| Platform-split entry correlation (M2) | Spearman ρ = -0.006 with the specified convex-hull C5 mask | Fig 3D |
| Platform-split entry correlation (M6) | Spearman ρ = 0.015 | Fig 3D |
| Nullspace dimensionality | 20 at N = 50; 0 at N >= 100 | Fig 3E |
| Trajectory PCA at N = 50 | PC1 = 94.10%; PC1-2 = 98.41%; 2 dimensions at the 95% threshold | Results |

The earlier response-gene-label "permutation null" is not reproduced because it
is algebraically invalid: permuting response columns only permutes rows of
`W`, leaving any matrix-wide stable-entry fraction unchanged. A unit test now
enforces this invariance. The empirical bootstrap fraction is retained only as
a descriptive resampling diagnostic.

## Steatosis

| Quantity | Reproduced value | Manuscript |
|---|---|---|
| Steatosis cohort | 10,399 spots (M1 3,588; P6 3,134; M3 2,448; M2 1,229) | Fig 4 |
| Robust genes (all six criteria) | GLUL, LYZ, ORM1 | Fig 4A |
| Lipid tertile cuts | 0.13% and 1.98% | Fig 4B |
| Supplementary Table 11 calibration, 3 genes | 3/3 sign agreement; signed-magnitude ρ = 1.00 | Fig 4C |
| Pathway enrichment | Reactome "Drug ADME" NES = -1.80, FDR q = 0.050 | Fig 4D |
| Exhaustive Webb wild bootstrap | GLUL p = 0.1056; LYZ p = 0.0609; ORM1 p = 0.1133; no gene has Webb q < 0.05 across 66 genes | Table S1 |
| GLUL-excluded-axis sensitivity | coefficient-vector Pearson r = 0.945; Spearman ρ = 0.944; GLUL remains all-six-criteria robust | Table S2 |

## Portal-side acute-phase response

| Quantity | Reproduced value | Manuscript |
|---|---|---|
| Weak-FDR-only genes | FBLN1, IGKC, MT1H, SAA1, TPSB2 | Fig S5 |
| SAA1 interaction coefficient | beta_2 = +0.015 (q = 6.6e-3); leave-M1 beta_2 = +0.038 (p = 0.021) | Fig S6 |
| SAA1 per-tertile gap | Δ portal = +0.31; Δ central = +0.16 | Fig S6 |
| Supplementary Table 11 calibration, 7 genes | 6/7 sign agreement; signed-magnitude ρ = 0.82 (p = 0.023) | Results |

## Second-tissue validation (intestinal crypt-villus)

`scripts/crypt/` runs the framework on the Moor *et al.* (2018) intestinal
crypt-villus zonation reconstruction, a second tissue independent of the
liver atlas.

| Quantity | Reproduced value | Manuscript |
|---|---|---|
| Canonical crypt-villus markers | 15/15 recover the literature-expected slope direction | Fig S10A |
| Gene filter | 9,241 genes pass (mean zone expression above 25th percentile, q < 0.05) | Methods |
| W effective dimensionality | 2 components capture 95% of the 7-zone trajectory variance | Fig S10B |
| W bootstrap stable-entry fraction | 0.585 (descriptive; no response-label permutation comparison) | Supporting result |

## Figures

The four main figures (`fig1`-`fig4`) and ten supplementary figures
(`figS1`-`figS10`) all render without error. `scripts/16_verify_outputs.py`
checks the complete artifact inventory after rendering.

## cell2location

Step 10 (cell2location) is GPU-bound. cell2location / scvi-tools training is not
bitwise-reproducible across GPU runs, so the per-spot hepatocyte fractions - and
the hepatocyte-composition sensitivity results that depend on them (Fig S4 and
the composition-confound row of Fig S1) - are reported from the cell2location output shipped
alongside the atlas (see [`data/README.md`](data/README.md)). The threshold
sweep applied to that input is deterministic.
