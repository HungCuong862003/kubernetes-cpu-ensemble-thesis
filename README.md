# Hybrid Ensemble Learning for Proactive Resource Prediction in Kubernetes

Bachelor's Thesis | International University, Vietnam National University Ho Chi Minh City
Author: Jimmy             Supervisor: Dr. Ho Long Van
Defense window: 8-26 June 2026   First-version submission: 4-8 May 2026

## TL;DR
A non-negative least-squares (NNLS) heterogeneous ensemble of XGBoost,
LightGBM, ExtraTrees, and BiLSTM for CPU prediction in Kubernetes
clusters, evaluated across three datasets (Alibaba, Bitbrains, ByteDance)
and four horizons (10/30/60/120 min), benchmarked against the
Chronos-Bolt, TimesFM, and Granite TTM foundation models.

## Repository map
| Path          | Contents                                                  |
|---------------|-----------------------------------------------------------|
| `data/`       | Raw, interim, processed, external datasets                |
| `src/`        | Python package + scripts + post-hoc analyses              |
| `models/`     | Serialised trained artefacts                              |
| `results/`    | All experimental outputs (dataset x horizon x artefact)   |
| `reports/`    | Figures, tables, statistical-test PDFs                    |
| `thesis/`     | LaTeX manuscript, defence slides                          |
| `docs/`       | Data dictionaries, model cards, methodology, repro guide  |
| `submission/` | Frozen archival snapshots (do not edit)                   |

## Reproducing the results
```bash
conda env create -f environment.yml
conda activate kubernetes-cpu-ensemble
pip install -e .
make all
```

## How to cite
See `CITATION.cff`.

## Repository layout philosophy
This project follows the FAIR Guiding Principles (Wilkinson et al., 2016),
the Turing Way reproducible-project template, Cookiecutter Data Science v2
(DrivenData), and the research-compendium pattern of Marwick, Boettiger
and Mullen (2018). Code is licensed MIT; text and figures CC-BY-4.0.
