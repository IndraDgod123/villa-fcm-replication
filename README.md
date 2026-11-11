# Replication package — Fuzzy C-Means segmentation of villa booking profiles

Code for the revised manuscript "Fuzzy C-Means Segmentation of Booking Profiles from Integrated
Heterogeneous Records of Four Villas in Bali" (J. Soft Comput. Explor.).

## Data
The real booking table (`master_data.csv`, 17,762 rows) is not public because of guest privacy and
commercial confidentiality. `data/synthetic_master_data.csv` has the same columns and is produced by
`scripts/make_synthetic.py`: joint villa–month–nationality–channel combinations are resampled, while dates,
room units, guest numbers and stay lengths are re-drawn and names are artificial. Results on the synthetic
file are close to, but not identical with, those in the paper.

## Run
```
pip install -r requirements.txt
cp data/synthetic_master_data.csv data/master_data.csv   # or place the real table here
python scripts/a1_data_quality.py        # missingness by villa, residual-duplicate audit
python scripts/a2_dedup_sensitivity.py   # re-runs under stricter deduplication keys
python scripts/a3_cluster_analyses.py    # k and m selection, PCA sensitivity, effect sizes, stability, baselines (~5 min)
python scripts/a4_diagnostics.py         # k = 7 and m = 1.3 profiles, rule surrogates, 2020 split
python scripts/a5_unknown_audit.py       # Unknown/Other sensitivity, audit of 100 collision groups
python scripts/f0_flowchart.py; python scripts/f1_figures.py   # figures
```
Run all scripts from the package root. Outputs are written to `out/` and `fig/`.

## Pipeline
`scripts/pipeline.py` reproduces the feature construction of the study: "Unknown" coding of missing
categories, per-villa median imputation, season mapping (Peak: Jun–Aug, Dec, Jan; Shoulder: Apr, May,
Sep–Nov; Low: Feb, Mar), channel and region mappings, Min–Max scaling plus one-hot encoding (20 columns),
PCA with 8 components, and scikit-fuzzy `cmeans` with k = 3, m = 2.0, error 1e-5, maxiter 1000, seed 42.
The Apache Spark harmonization of the 77 raw source files is not included because the raw files contain
guest names.
