# Memory snapshot updates — 2026-05-25 F3 Day 1

The following items should appear in next session's userMemories `recent_updates`.
If auto-memory misses them, paste them as memory_user_edits.

---

## Items to ADD (or expect in auto-memory)

1. **2026-05-25 F3 Day 1 probe trail SUCCESS:** Chronos-2 calling convention locked via four probe iterations. Production calling convention: `pipe = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cuda', dtype=torch.bfloat16)`; input is 3D tensor `(n_series, 1, n_context)` float32; `predict_quantiles(inputs, prediction_length, quantile_levels)` returns `tuple[list, list]` of per-series tensors with per-series shape `(1, n_future, n_quantiles)`. Stack, squeeze n_variates axis 1, moveaxis last→first to get `(n_quantiles, n_series, n_future)` for pinball loss.

2. **2026-05-25 F3 Day 1 data layout discovered:** Raw time series at `data/processed/<dataset>/{train,val,test}.parquet` (Alibaba 56.4+12.9+13.0 MB, Bitbrains 3.8+1.0+0.8 MB, ByteDance 0.6+0.2+0.2 MB). Per-horizon aligned spine at `results/<dataset>/h<HHH>/predictions/test_spine.parquet` with 5-column schema `[container_id, time_stamp, cpu_residual, cpu_target, naive_cpu]`. Alibaba test alone has ~340 obs/container — insufficient for N_CONTEXT=512; fix is concatenation of train+val+test per container before sampling.

3. **2026-05-25 F3 Day 1 production script:** `phase_f/scripts/f3_baseline_chronos2.py` (446 lines, patched) uses direct `chronos.BaseChronosPipeline` — NO AutoGluon. Loads train+val+test, auto-detects id column (container_id vs vm_id) and cpu column (cpu_util_percent, cpu_target, cpu, cpu_percent). N_CONTEXT=512, BATCH_SIZE=256, QUANTILE_LEVELS=[0.5,0.7,0.8,0.9,0.95]. Pre-registered F3 success: ≥5% pinball improvement at τ=0.9 h=60min averaged across datasets; PARTIAL ≥2%; FAILURE <2% triggers DECISION-015.

4. **2026-05-25 Vast env state (`C.37705458`):** Python 3.12.13, `/venv/main`, torch 2.9.1+cu128, NO torchvision (intentionally uninstalled to fix `RuntimeError: operator torchvision::nms does not exist` C++ ABI mismatch). chronos-forecasting installed, Chronos-2 weights downloaded (~480 MB). RTX A4000, 16.6/16.8 GB free. AutoGluon 1.5.0 installed but unused by production code.

5. **2026-05-25 F3 Day 1 PENDING:** `f3_baseline_chronos2.py` patched but not yet run successfully. Next action: save patched script to Vast, run with `python3 f3_baseline_chronos2.py 2>&1 | tee ../data/f3_baseline_output.txt`. Expected runtime 30–60 min. Produces `f3_zero_shot_baseline.{json,csv}` + `phase_f/f3_design.md`.

---

## Items to REMOVE / SUPERSEDE (older notes that are now stale)

- "2026-05-25 PAR Step 5 → STRUCTURAL SATURATION at sample-weighted level" — keep
- "DECISION-014 PRE-REGISTERED" — UPDATE to "DECISION-014 LOCKED 2026-05-25 (PAR fails BOTH criteria; STRUCTURAL SATURATION confirmed)"
- "ERRATA-014: F1 router framed as future-positive in submitted PDF Ch3 §3.7 / Ch5 §5.6" — keep, still pending application
- "F3 setup in progress, environment debugged" — SUPERSEDE with item #5 above ("F3 baseline patched, pending run")

---

## Critical reminders for next session

- **Do NOT re-run PAR pipeline** — env was modified for AutoGluon (sklearn 1.7.2, pandas 2.3.3, numpy 2.1.3, xgboost 3.1.3, pyarrow 20.0.0, huggingface_hub 0.36.2). PAR outputs already saved; re-running with these versions may give different numbers.
- **F3 success criterion is LOCKED** — do not adjust thresholds after seeing results. PARTIAL is honest if results land there.
- **DECISION-015 is a placeholder** — only locks if F3 baseline shows FAILURE (<2% improvement). Until baseline + LoRA results land, it remains open.
- **Manuscript chapter rewrite** under Pivot C+D framing (6 chapters: BCF, PAR, Structural Saturation, F3, F4, Discussion) NOT YET STARTED — depends on F3 outcome.
- **Defence timeline:** ~March 2027 (one-semester delay accepted by Dr Ho Long Van).

---

## File paths quick reference (for next session)

| Path | Contents |
|---|---|
| `phase_f/scripts/_paths.py` | Path resolver |
| `phase_f/scripts/f3_probe_v4.py` | API probe (saved on Vast) |
| `phase_f/scripts/inspect_data_layout.py` | Data inspector (saved on Vast) |
| `phase_f/scripts/f3_baseline_chronos2.py` | **Patched baseline (NEEDS RE-SAVE from /home/claude/)** |
| `phase_f/data/par_features.parquet` | PAR features |
| `phase_f/data/par_labels.parquet` | PAR labels |
| `phase_f/data/par_router_results.json` | Step 5 results |
| `phase_f/data/par_router_predictions.csv` | Step 5 predictions |
| `phase_f/data/par_prior_correction_results.json` | Step 5b results |
| `phase_f/data/f3_zero_shot_baseline.{json,csv}` | **Expected after F3 baseline run** |
| `phase_f/f3_design.md` | **Expected after F3 baseline run (pre-reg doc)** |
| `data/processed/<dataset>/{train,val,test}.parquet` | Raw time series |
| `results/<dataset>/h<HHH>/predictions/test_spine.parquet` | Aligned spine |
# Memory snapshot updates — 2026-05-25 F3 Day 1

The following items should appear in next session's userMemories `recent_updates`. If auto-memory doesn't pick them up cleanly, paste them as memory_user_edits.

---

## Items to ADD (or expect in auto-memory)

1. **2026-05-25 F3 Day 1 BASELINE LOCKED:** Zero-shot Chronos-2 pinball loss across 3 datasets × 4 horizons. Primary metric (mean pinball at h=60, τ=0.9 across 3 datasets) = 1.921658. SUCCESS threshold 1.825576 (≥5% better), PARTIAL 1.883225 (≥2%), FAILURE >1.883225 triggers DECISION-015. Per-cell at τ=0.9: Alibaba 0.237/0.347/0.447/0.474; Bitbrains 1.535/2.122/4.859/8.015; ByteDance 0.383/0.396/0.459/0.480 (h=10/30/60/120). Inference time: 14s total for 11 cells on RTX A4000.

2. **2026-05-25 F3 Day 1 Bitbrains scale dominance:** Bitbrains contributes 84% of unweighted-mean primary metric (1.620 of 1.922) due to absolute CPU value differences (Bitbrains cpu_target ~18–25, Alibaba ~3–5, ByteDance ~28–68). Pinball loss not scale-invariant. Option C recommended (RECOMMENDED, not yet locked): keep pre-reg primary AND add secondary = geometric mean of per-dataset % improvements at h=60, τ=0.9, same thresholds. To lock at next session by appending to `phase_f/f3_design.md`.

3. **2026-05-25 F3 Day 1 probe trail SUCCESS:** Chronos-2 calling convention locked via four probe iterations. Production calling convention: `pipe = BaseChronosPipeline.from_pretrained('amazon/chronos-2', device_map='cuda', dtype=torch.bfloat16)`; input is 3D tensor `(n_series, 1, n_context)` float32; `predict_quantiles(inputs, prediction_length, quantile_levels)` returns `tuple[list, list]` of per-series tensors with per-series shape `(1, n_future, n_quantiles)`. Stack, squeeze n_variates axis 1, moveaxis last→first to get `(n_quantiles, n_series, n_future)` for pinball loss.

4. **2026-05-25 F3 Day 1 data layout:** Raw time series at `data/processed/<dataset>/{train,val,test}.parquet` (Alibaba 56.4+12.9+13.0 MB, Bitbrains 3.8+1.0+0.8 MB, ByteDance 0.6+0.2+0.2 MB). Per-horizon aligned spine at `results/<dataset>/h<HHH>/predictions/test_spine.parquet` with 5-column schema `[container_id, time_stamp, cpu_residual, cpu_target, naive_cpu]`. Alibaba test alone has ~340 obs/container — insufficient for N_CONTEXT=512; fix is concatenation of train+val+test per container before sampling. Container length post-concat: Alibaba min 429 median 2296 max 2304; Bitbrains min 7285 median 8615; ByteDance constant 3456.

5. **2026-05-25 F3 Day 1 production script:** `phase_f/scripts/f3_baseline_chronos2.py` (446 lines, patched) uses direct `chronos.BaseChronosPipeline` — NO AutoGluon. Loads train+val+test concat, auto-detects id column (container_id vs vm_id) and cpu column (cpu_util_percent first). N_CONTEXT=512, BATCH_SIZE=256, QUANTILE_LEVELS=[0.5,0.7,0.8,0.9,0.95]. Pre-registered F3 success criterion auto-written to `phase_f/f3_design.md`.

6. **2026-05-25 Vast env state (`C.37705458`):** Python 3.12.13, `/venv/main`, torch 2.9.1+cu128, NO torchvision (intentionally uninstalled to fix `RuntimeError: operator torchvision::nms does not exist` C++ ABI mismatch). chronos-forecasting installed, Chronos-2 weights downloaded (~480 MB). RTX A4000, 16.6/16.8 GB free. AutoGluon 1.5.0 installed but unused by production code. sklearn 1.7.2, pandas 2.3.3, numpy 2.1.3, xgboost 3.1.3, pyarrow 20.0.0, huggingface_hub 0.36.2 — env modified for AutoGluon; do not re-run PAR pipeline with this env.

7. **2026-05-25 F3 within-pattern observations:** Pinball monotonically decreases with τ across ALL 11 cells (predictions slightly biased above truth at high quantiles = over-provisioning behaviour, desirable). Pinball increases with horizon as expected; Bitbrains shows largest horizon degradation (5×, consistent with non-diurnal ACF profile, ACF@24h≈0.12). Chronos-2 already produces moderately calibrated over-provisioning quantiles out of the box; LoRA has finite room to improve.

---

## Items to REMOVE / SUPERSEDE (stale)

- "F3 setup in progress, environment debugged" → SUPERSEDE with item #1 above (baseline LOCKED)
- "f3_baseline_chronos2.py patched but not yet run" → SUPERSEDE with item #5 (run successful, results saved)
- "DECISION-014 PRE-REGISTERED" → SUPERSEDE with "DECISION-014 LOCKED 2026-05-25 (PAR fails BOTH criteria; STRUCTURAL SATURATION confirmed)" (from prior session)

---

## Critical reminders for next session

- **F3 primary metric is LOCKED** (mean pinball at h=60, τ=0.9). Do not change formula or thresholds after seeing fine-tune results. Option C secondary is additive, not a replacement.
- **DECISION-015 placeholder** remains until F3.5 (only locks at SUCCESS/PARTIAL/FAILURE decision).
- **Do NOT re-run PAR pipeline** — env was modified for AutoGluon; PAR outputs already saved.
- **Manuscript chapter rewrite** under Pivot C+D framing NOT YET STARTED — depends on F3 outcome.
- **Defence timeline:** ~March 2027 (one-semester delay accepted by Dr Ho Long Van).

---

## F3 task tracker (post-Day-1)

| # | Task | Status |
|---|---|---|
| F3.1 | Zero-shot baseline | ✅ DONE 2026-05-25 |
| F3.2 | Setup probe + secondary metric lock | ⏸️ NEXT |
| F3.3 | LoRA fine-tune script | Pending |
| F3.4 | Run training | Pending |
| F3.5 | Evaluation + DECISION-015 lock | Pending |
| F3.6 | Robustness ablations | Pending |

Estimated remaining: 4–6 working days.

---

## File paths quick reference

| Path | Contents |
|---|---|
| `phase_f/scripts/_paths.py` | Path resolver |
| `phase_f/scripts/f3_probe_v4.py` | API probe (saved on Vast) |
| `phase_f/scripts/inspect_data_layout.py` | Data inspector (saved on Vast) |
| `phase_f/scripts/f3_baseline_chronos2.py` | Patched baseline, run successfully |
| `phase_f/data/f3_zero_shot_baseline.json` | LOCKED baseline numbers |
| `phase_f/data/f3_zero_shot_baseline.csv` | Flat baseline table |
| `phase_f/data/f3_baseline_output.txt` | Full stdout |
| `phase_f/f3_design.md` | Auto-written pre-registration (secondary metric to be appended) |
| `phase_f/data/par_features.parquet` | PAR features |
| `phase_f/data/par_labels.parquet` | PAR labels |
| `phase_f/data/par_router_results.json` | Step 5 results |
| `phase_f/data/par_prior_correction_results.json` | Step 5b results |
| `data/processed/<dataset>/{train,val,test}.parquet` | Raw time series |
| `results/<dataset>/h<HHH>/predictions/test_spine.parquet` | Aligned spine |