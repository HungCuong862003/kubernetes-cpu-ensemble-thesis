# D16 Journal — PAR Pivot + Dr. Ho Confirmation
**Date:** 2026-06-07 (D16)

---

## What happened today

**Research synthesis complete (D15/D16):**
Three rounds of deep research covering F1 improvements, F2 improvements, and combined
synthesis identified a single unified root cause for both nulls:

> Cloud workloads are ACF-saturated. Every predictability metric (WPE, CV, ACF@1h,
> ACF@24h) collapses to one dimension. F1's dataset-level features encoded dataset
> identity not a predictability gradient. F2's WPE added nothing beyond ACF because
> positive autocorrelation mechanically depresses WPE on red-noise-like signals.

Anchored by: Şen et al. (2024, MEE) for WPE-ACF mechanism; Wang et al. (2025, arXiv)
for independent validation of the threshold behaviour; Karl et al. (ICML 2024) for the
NMNR framing template; Garland et al. (2014, PRE) for per-series WPE as predictability
indicator; FFORMA (Montero-Manso et al. 2020) for per-series features as meta-learning
standard.

**Dr. Ho Long Van confirmed new direction.**

**DECISION-013 locked:**
- F1 cell-level result stays (NULL, pre-reg complete)
- PAR replaces standalone F3 as primary Phase F contribution
- Per-series features (catch22 + DFA + LZC) replace dataset-level medians
- Regret metric replaces macro-F1 as primary evaluation
- F3 quantile FT demoted to optional ablation

---

## Key insight for the thesis

The original F1 plan (§4 of Phase F Master Plan) was always per-series:
"5155 series × 4 horizons × 4 models × 5 ratios = 412K simulator runs"

What we ran as "F1" was a simplified pre-implementation (n=12 cells). PAR is
implementing the original F1 design at its correct granularity, now motivated
by the F1+F2 diagnosis and using better features (catch22 + DFA + LZC replacing
dataset-level medians that collapsed to dataset identity).

---

## Pending paperwork (D16 priority)

Must complete before PAR Week 1 to keep work record clean:

1. ERRATA-012 — apply in Overleaf per D13 journal substitution text
2. Biblio audit — run f1_biblio_verify.py on Windows with .bib file
3. D3 retro diff — git diff in Overleaf history

---

## PAR Week 1 plan (starting D17 or after paperwork close)

```bash
# Install libraries on Vast.ai
pip install antropy pycatch22 nolds

# Write phase_f/scripts/par_features.py
# Compute catch22 + DFA + LZC + SampEn per series
# Output: phase_f/data/par_features_{alibaba,bitbrains,bytedance}.parquet
# Estimated: 3 days compute, ~5 min total wall time
```

---

## Literature anchors added to thesis bibliography

- Şen et al. (2024) — WPE-ACF substitution mechanism on red-noise signals
- Wang, Quan, Yang & Srivastava (2025, arXiv:2511.08884) — spectral predictability
  for TSFM selection (concurrent, preprint only)
- Garland, James & Bradley (2014, PRE 90:052910) — per-series WPE as predictability
  indicator validated on 120 series
- Karl, Kemeter, Dax & Sierak (ICML 2024, PMLR 235:23256) — NMNR framing template
- Lubba et al. (2019, DAMI 33:1821) — catch22 canonical time series features
- Montero-Manso et al. (2020, IJF 36:86) — FFORMA per-series meta-learning

---

## EOD status

- DECISION-013: locked and documented
- THESIS_STATE.md: refreshed for D16
- D16 journal: this file
- Paperwork: ERRATA-012 + biblio + D3 diff still pending (Jimmy action)
- PAR: direction locked, Week 1 starts after paperwork close
