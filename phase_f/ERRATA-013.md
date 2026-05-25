# ERRATA-013 — Bitbrains test_ensemble_hetero.npy files contain Alibaba data (project storage)

**Logged:** 2026-05-25 (PAR D17 diagnostic, par_diagnose_issues.py Section A)
**Scope:** project storage; does NOT affect submitted PDF or any published number
**Manuscript impact:** none directly; relevant only for PAR Phase F work
**Action:** documented; substitution used in PAR labels; no manuscript change required

## Finding

`/results/bitbrains/h{030,060,120}/predictions/test_ensemble_hetero.npy` contain
Alibaba data, not Bitbrains data. Confirmed via shape and value distribution:

| Cell | Bitbrains shape (expected) | hetero shape (actual) | hetero mean (actual) | y_true mean (Bitbrains) |
|---|---|---|---|---|
| Bitbrains h030 | 202,968 | **1,650,759** (= Alibaba h030) | 10.28 | 30.37 |
| Bitbrains h060 | 202,116 | **1,621,233** (= Alibaba h060) | 10.28 | 30.34 |
| Bitbrains h120 | 200,412 | **1,562,181** (= Alibaba h120) | 10.32 | 30.47 |

The mean ~10.3 matches Alibaba's CPU distribution; Bitbrains y_true mean ~30.4.
Shape matches Alibaba's full-test size at each horizon. The Bitbrains-shaped
file does not exist at this path. The h010 hetero file does not exist at all
(`test_ensemble_hetero.npy` MISSING for h010, consistent with the known
BiLSTM OOF shared-memory failure at h10 documented in `run.log`).

All other Bitbrains ensemble files at the same paths have correct Bitbrains
shapes (~203K) and value distributions: `test_ensemble_homo.npy`,
`test_ensemble_v2.npy`, `test_ensemble_v3.npy`, `test_xgb.npy`, `test_et.npy`.
Issue is isolated to the four hetero `.npy` files.

## Likely cause

Pipeline run-script copied Alibaba's hetero output to the Bitbrains path during
an earlier session (file path templating error). Other prediction files were
written correctly by their respective per-dataset scripts.

## Manuscript impact

**None.** The submitted PDF does not cite the Bitbrains hetero ensemble in any
R² table. `leaderboard_v1.csv` Bitbrains rows use `nnls_agg_method = per_vm_median`
with `nnls_pool = OLD (XGB+LGB+ET+BiLSTM)` — the canonical reference for
Bitbrains NNLS is the per-VM XGBoost median in
`results/bitbrains/diagnostics/bitbrains_per_vm_results.csv`, which was
computed independently and is unaffected.

Figure 4.2 caption already states "computed over N = 156 active VMs using
per-VM XGBoost as the ML reference (the global hetero ensemble fails to
generalise on Bitbrains GWA-T-12; see Table 4.6 caption)". The corrupted
files would have failed any sanity check at the time of writing; the manuscript's
choice of per-VM XGBoost as the Bitbrains ML reference already side-stepped them.

## PAR resolution

PAR per-series R² (Phase F) at the chronos2 K=50 spine needs a per-point
Bitbrains NNLS prediction. The canonical per-VM XGBoost is computed at full
test, not at the spine, and is per-VM-trained (different model class per VM).
For apples-to-apples per-series argmax with the four foundation models at the
spine, PAR substitutes `test_xgb.npy` (global XGBoost trained on all Bitbrains
VMs, predictions at full test, then aligned to the spine).

The substitution is documented in the PAR chapter's methods section:

> *"For Bitbrains, `leaderboard_v1.csv` reports per-VM-XGBoost median R² (full-test scope) as the canonical 'NNLS' reference. The global hetero ensemble file at the project storage path was found to contain corrupted (Alibaba) data; we therefore substitute global XGBoost (`test_xgb.npy`) aligned to the chronos2 K=50 spine, matching the model class used in `bitbrains_per_vm_results.csv`. The cell-level R² differs by ~0.05 from the published leaderboard's per-VM-XGBoost-median scope; the relative ordering of models per cell is preserved. See ERRATA-013."*

## Verification

`par_per_series_r2_v2.py` pre-flight sanity check (ii) confirms `test_xgb.npy`
at the Bitbrains h030 chronos2 spine gives pooled R² = 0.4973, matching the
diagnostic exactly. The substitution alignment is verified.

## Status

- 2026-05-25 — finding logged, substitution applied in PAR labels
- No manuscript edit required
- No further action: corrupted files left in place (not over-writing project storage)
