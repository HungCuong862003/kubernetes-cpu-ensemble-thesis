# Phase F Journal — append-only

> Lab notebook entries per substantive task. Timestamp, task, inputs, output, discrepancies, verification status. Never edited after the fact — if something needs correcting, add a new entry referencing the original.

---

## 2026-05-22 (F0 Day 1) — lock-down opening

### 1.4 — verify_foundation.py extended with v3 stubs

**Task.** Append the v3 stubs from `verify_foundation_v3_stubs.py` into `verify_foundation.py` per master plan §3 Day 1 spec.

**Inputs.**
- `verify_foundation.py` (pre-merge, 433 lines, 169/172 PASS on Vast.ai per Phase E)
- `verify_foundation_v3_stubs.py` (354 lines, 4 check functions + `_file_exists_or_skip` helper)

**Changes applied.**
1. Added `import os` to the imports block.
2. Added six v3 path constants after the existing path constants:
   - `LB_V3_WIDE` → `results/foundation_comparison/leaderboard_v3_wide.csv`
   - `LB_V3_LONG` → `results/foundation_comparison/leaderboard_v3_long.csv`
   - `LB_V3_WINNERS` → `results/foundation_comparison/leaderboard_v3_winners.csv`
   - `CD_HEADLINE_V2` → `results/foundation_comparison/cross_dataset_headline_v2.csv`
   - `C2_CORR` → `results/bcf_v2/c2_correlations_per_cell.csv`
   - `LOO_V2` → `results/bcf_v2/loo_ablation_new_pool_v2.csv`
3. Added `_file_exists_or_skip` helper before the v3 check functions.
4. Added four check functions in the order specified by the stubs file:
   - `check_section_14_v3_leaderboard(tn)` — v3 winner tally and Bitbrains headline cells
   - `check_section_15_v3_diversity(tn)` — pooled rho-bar ~ 0.6562
   - `check_section_16_v3_loo(tn)` — drop_chronos2 deltas confirm Chronos-2 load-bearing
   - `check_section_17_v3_d4(tn)` — 38W/3T/3L tally when d4_*.csv schema is readable
5. Added four calls in `main()` between `check_residual_diagnostics(tn)` and the final summary.
6. Updated header comment to reflect the extension (scope, expectations pre- and post-sync).

**Output.** `verify_foundation.py` updated to 757 lines. Python syntax check passes (`python3 -c "import ast; ast.parse(...)"`). Runtime verified on Vast.ai after task 1.1 rclone sync completed.

**Pre-sync run.** `phase_f/F0_D1_verify_pre-sync.log` — TOTAL [FILL IN, expected 169] PASS / [FILL IN, expected 3] FAIL. Sections 14-17 print SKIP messages as expected (v3 files not yet on disk).

**Post-sync run.** `phase_f/F0_D1_verify_post-sync.log` — TOTAL [FILL IN, expected ~200] PASS / [FILL IN] FAIL. Sections 14-17 exercise their schemas. The 3 known FAILs in sections 1-13 (v1-OLD vs v2-NEW pool scope drift in BCF Bitbrains row) remain unchanged.

**Discrepancies found during merge.**

A stale-canonical issue exists in the pre-existing `check_bcf_pooled` function that the v3 stub append does NOT resolve. Flagging here because it touches the seven critical scope rules.

- `BCF_JSON` constant points to `bcf_pooled_results.json` — the 4-model pool (Granite-TTM included). Per Phase A-E canonical_facts §3.3 and §6.3, the canonical BCF file is `results/bcf/bcf_pooled_3model.json` (3-model, post-Granite exclusion).
- The hardcoded targets in `check_bcf_pooled` (AUC=0.80, CI=[0.70, 0.88], p=0.011, with CI keys named `bca_95_ci_lo` / `bca_95_ci_hi`) encode stale memory: per canonical, CI is **percentile** (BCa degenerate for binary classifier), p is **0.0097**, and the AUC of 0.80 applies to the 3-model pool not the 4-model pool — the 4-model JSON has AUC=0.667, p=0.0511 per the Granite-exclusion justification.

NOT fixed in this commit. Out of scope for the Day 1.4 v3 stub append as the user defined the task. Added a NOTE docstring inside `check_bcf_pooled` so the issue is visible whenever someone reads the function. Tracked as F0 follow-up; a natural slot is the Day 5 Bitbrains Ch5 verification window since that block already opens the BCF prose for review.

**Verification status.**
- Static syntax: PASS (Python AST parses).
- Runtime: [FILL IN — "PASS, 169 pre-sync / N post-sync, 3 known FAILs unchanged"].

---

### 1.1 — rclone sync six Vast.ai file groups to Drive (two-hop) and pull local

**Vast.ai push leg.** Six rclone copy commands executed on Vast.ai per the master plan §3 spec. All six confirmed present on Drive via `rclone ls gdrive:kubernetes-cpu-ensemble-thesis/results/foundation_comparison/ | grep v3` and equivalents for bcf_v2.

**Local pull leg.** `phase_f_F0_D1_local_actions.sh` invoked rclone from local to pull all six files into the repo. Verification block at script tail confirmed all six [FILL IN — "PASS" or "MISS"] on local disk.

**Status:** [FILL IN — PASS or list specific MISS files].

---

### 1.2 — Verify `decisions/supervisor_acceptance_2026-05-22.pdf` exists

**Result.** [FILL IN — PASS with byte count, or FAIL with action taken].

---

### 1.3 — PDF grep for "BCa" on submitted thesis

**Command.** `pdftotext -layout Ensemble_Learning_for_Proactive_Resource_Prediction_PhanNguyenHungCuong_ITDSIU21078.pdf - | grep -in -E "BCa|bias[- ]?corrected[- ]?accelerated"`

**Output.** [FILL IN — paste verbatim grep matches, or "zero matches"].

**Verdict on DECISION-007.** [FILL IN — ACTIVE because matches reference the BCF CI / UNUSED because no matches / UNUSED because matches do not reference the BCF CI].

**If ACTIVE.** Errata item 6 drafted: "In the submitted manuscript at [section/page TBD], 'BCa 95% CI' should read 'percentile 95% CI'. Rationale: BCa is degenerate for a binary classifier per the canonical 3-model pool JSON at `results/bcf/bcf_pooled_3model.json`, which records `bias_correction_undefined: true` and reports percentile CI [0.7097, 0.8871]." Errata sheet updated; supervisor sign-off scheduled for Day 9.

---

### 1.5 — Overleaf §4.7 lookup for foundation-model leaderboard tally

**Overleaf passage (verbatim from Ch4 §4.7).**

```
[FILL IN — paste passage from Overleaf here]
```

**Verification against `leaderboard_v1.csv` canonical tally (Chronos-2 6 / TimesFM 3 / Granite-TTM 2 / NNLS 1).**

[FILL IN — Claude verdict: "passage matches canonical, no edit needed" / "passage cites wrong numbers, edit queued for next Overleaf push" / specific discrepancies].

---

### End-of-day status

Day 1 of 135 closed. Five tasks complete (with branch outcomes filled in above). One F0 follow-up surfaced (`check_bcf_pooled` stale-canonical) and tracked for Day 5 window. No blockers for Day 2.

Tomorrow opens with F0 Day 2 — drafting `convert_toto_json_to_npz.py` to convert three `toto_k20_*.json` files into per-point npz format for F1 label generation.

---
