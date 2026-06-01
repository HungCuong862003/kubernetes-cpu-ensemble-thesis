# DATA_MANIFEST

Last updated: 2026-06-01.

Records what data this project uses, where the bytes live, and how to fetch them.
The repo deliberately does NOT contain raw data — only small processed feature
sets that scripts consume directly. Raw traces (~333 GB) are cloud-canonical on
Google Drive and locally mirrored at `E:\Thesis\raw-data\`.

---

## Raw data (NOT in this repo)

Raw cluster traces are cloud-canonical on Google Drive at
`gdrive:kubernetes-cpu-ensemble-thesis/raw-data/`, mirrored locally to
`E:\Thesis\raw-data\` via `rclone copy`. Never edit in place.

### Alibaba 2018 Cluster Trace

- **Source:** https://github.com/alibaba/clusterdata (2018 release)
- **Local path:** `E:\Thesis\raw-data\alibaba\`
- **Drive path:** `gdrive:kubernetes-cpu-ensemble-thesis/raw-data/alibaba/`
- **Files:**
  - `container_usage.csv` — 168 GB
  - `container_usage.tar.gz` — 28 GB
  - `batch_instance.csv` — 106 GB
  - `batch_instance.tar.gz` — 20 GB
  - `machine_usage.csv` — 8.6 GB
  - `machine_usage.tar.gz` — 1.7 GB
  - `batch_task.csv` — 765 MB
  - `batch_task.tar.gz` — 124 MB
  - `container_meta.csv` — 17.8 MB
  - `container_meta.tar.gz` — 2.4 MB
  - `machine_meta.csv` — 0.5 MB
  - `machine_meta.tar.gz` — 0.1 MB
- **Total:** ~333 GB
- **Retrieved:** 2025-11-23
- **Used for:** primary dataset; 4,921 containers after filtering, 48-hour windows, 5-minute intervals

### Bitbrains GWA-T-12

- **Source:** http://gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains
- **Local path:** `E:\Thesis\raw-data\bitbrains\` (or inside the repo at `data\bitbrains\`)
- **Drive path:** `gdrive:kubernetes-cpu-ensemble-thesis/raw-data/bitbrains/`
- **Files:** 156 VM trace files, 5-minute intervals (142 retained post-filter)
- **Used for:** cross-dataset validation. CV_median=1.056, Hurst_median=0.997, ACF@24h=0.116

### ByteDance IaaS

- **Source:** <FILL IN SOURCE URL — internal release / paper supplementary?>
- **Local path:** `E:\Thesis\raw-data\bytedance\` (or inside the repo at `data\bytedance\`)
- **Drive path:** `gdrive:kubernetes-cpu-ensemble-thesis/raw-data/bytedance/`
- **Files:** 93 container traces, 10-minute intervals
- **Used for:** third-dataset validation. Hurst_median=0.956, ACF@24h=0.489

---

## Processed data (IN this repo, under `data/`)

Small feature sets produced from the raw traces by scripts under `src/`.
Committed to git so the repo is self-contained for downstream analysis,
chapter generation, and defence-package export.

(FILL IN — one line per file in `data/` once verified:)

- `data/<filename>` — <one-line description> — produced by `src/<script>.py`
- ...

---

## Reproduction

To rebuild the processed data from raw:

```bash
# 1. Pull raw data from Drive (one-time, ~333 GB)
rclone copy gdrive:kubernetes-cpu-ensemble-thesis/raw-data E:\Thesis\raw-data

# 2. Activate environment
cd E:\Thesis\kubernetes-cpu-ensemble-thesis
.venv\Scripts\activate                     # Windows
# or: source .venv/bin/activate            # Linux/WSL

# 3. Rebuild processed features
make features                              # or the equivalent direct script call

# 4. Train and evaluate
make train
make figures
```

---

## Provenance notes

- Imputation rates (post-Issue-1-fix): Alibaba 7.9–15.6 %, Bitbrains 24.4–25.4 %,
  ByteDance 17.3–17.6 %. Documented in §3 of the manuscript.
- Filtering criteria for each dataset are encoded in `src/<script>.py`
  (FILL IN exact path) and version-controlled.
- Boundary Condition Framework (BCF) predicate: `ACF@24h > 0.2 AND horizon ≥ 30min`.
  See `bcf_pooled_3model.json` in `results/` for the canonical AUC and CI.

---

## Pre-registration

All Phase F thresholds were accepted in writing by Dr. Ho Long Van before
experiments ran. See `phase_f/pre_registration_2026-05-22.md` (or the equivalent
path in the repo) for the signed thresholds. Negative results are reported
as findings, not hidden.
