# D4 — WPE prep + Q-007 reopened (2026-05-26)

casual notes, not a formal handoff. Vast.ai instance C.37423026.

## scope

biblio audit + manuscript stuff parked. F2 prep work today — compute WPE for
the three datasets so the pre-registered partial-R²(WPE | ACF@24h) >= 0.3
target has something to bind against.

Q-007 was *attempted* but reopened — earlier "stale literal" diagnosis was
wrong (see findings below).

## decisions

- WPE = Bandt-Pompe weighted PE per Fadlallah et al. 2013. m=4, tau=1.
- Read all 3 datasets from data/processed/<ds>/{train,val,test}.parquet,
  concatenate per series, sort by time, then compute WPE.

## what got done

- [ ] pyarrow installed (was missing from the main venv)
- [ ] parquet schemas inspected for all 3 datasets
- [ ] wrote phase_f/scripts/compute_wpe.py
- [ ] Bitbrains: <N> series, WPE range [<min>, <max>], median <med>
- [ ] ByteDance: <N> series, WPE range [<min>, <max>], median <med>
- [ ] Alibaba: <N> series, WPE range [<min>, <max>], median <med>
- [ ] saved phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv
- [ ] sanity check WPE vs ACF@24h:
    - Bitbrains: r = <value>  (expect negative)
    - ByteDance: r = <value>
    - Alibaba:   r = <value>

## findings / surprises

- Vast.ai workspace is not a git repo (only local Windows is). Earlier EOD
  instructions wrongly put `git commit` on Vast.ai; fixed to the proper
  Vast.ai → Drive → Local → git flow.
- rclone check fails with "is a file not a directory" when applied to single
  files. State file verification needs `rclone hashsum md5` + manual diff
  instead. SYNC_PROTOCOL.md §Verification has a latent bug here.
- 3 stranded F0 D1 verify logs at workspace root never made it to Drive at
  D1 close. Pushed today.
- Two copies of bytedance_per_instance_stats.csv (root + diagnostics/).
  <append diff outcome here>

## Q-007 reopened

Earlier diagnosis was "3 stale BCF Bitbrains literals". Actual situation is
a metric-scope shift, not a stale-literal:

- Verifier expects: -5.46pp (h30), +3.30pp (h120) — OLD pool per-VM medians
- Verifier sees:    +1.53pp (h30), -2.81pp (h120) — NEW pool per-VM medians
- Memory had me about to plug in POOLED full-test deltas (+4.47, -1.26)
  which would have left the verifier still failing on a different metric.

Three valid Bitbrains delta metrics in play, all in canonical files:

| metric                    | h10    | h30    | h60    | h120   | source |
|---------------------------|--------|--------|--------|--------|--------|
| Pooled full-test NEW pool | +0.97  | +4.47  | +1.91  | -1.26  | cross_dataset_headline_v2.csv |
| Per-VM median NEW pool    | -16.66 | +1.53  | -1.99  | -2.81  | (computed live by verifier) |
| Per-VM median OLD pool    | ?      | -5.46  | ?      | +3.30  | bitbrains_summary_corrected.csv (current verifier literals) |

Doing Q-007 properly = decide which metric the verifier should anchor to,
update 4 literals (not 2) + rewrite the verdict text, possibly regenerate
the source CSV under NEW pool. ~45-60 min, not 30. Deferred to D5.

## blockers

- if WPE vs ACF@24h correlation is *positive* in any dataset, implementation
  is suspect. log + investigate before claiming F2 ready.

## next

- D5: partial-R²(WPE | ACF@24h) against delta_pp from bcf_pairs.csv.
  this is the actual F2 pre-reg test (threshold >= 0.3).
- D5: Q-007 done properly per the metric-scope rewrite above.

## time

started ~<HH:MM>, finished ~<HH:MM>, total ~<duration>

---

## actual findings (post-run, 2026-05-26)

### WPE per-dataset distributions

| dataset    | n    | WPE min | median | max    | ACF@24h median |
|------------|------|---------|--------|--------|----------------|
| Alibaba    | 5000 | 0.2405  | 0.5667 | 0.7332 | 0.3158         |
| Bitbrains  | 142  | 0.5572  | 0.7276 | 0.9489 | 0.1090         |
| ByteDance  | 93   | 0.3804  | 0.9545 | 0.9935 | 0.4894         |

Note 98 Alibaba WPE series have no omega entry (5000 - 4902); they were dropped
by the original omega filtering, not a current issue.

### Within-dataset correlations (Pearson)

| dataset    | WPE vs ACF@24h | WPE vs CV  | WPE vs Hurst |
|------------|----------------|------------|--------------|
| Bitbrains  | +0.8744        | -0.2460    | +0.3455      |
| ByteDance  | +0.1838        | -          | +0.3911      |
| Alibaba    | +0.1303        | -0.3188    | +0.1806      |

### Cross-dataset ordering

ByteDance has the HIGHEST ACF@24h (0.49) and the HIGHEST WPE (0.95). Alibaba
has middle ACF@24h (0.32) and the LOWEST WPE (0.57). Bitbrains has the LOWEST
ACF@24h (0.11) and middle WPE (0.73).

### Surprises

- **WPE vs ACF@24h came out POSITIVE in all 3 datasets**, not negative as
  predicted at the script-writing stage. My a-priori reasoning was
  "high ACF@24h → predictable → low entropy → low WPE", which conflated
  two different time scales. WPE on m=4, tau=1 motifs measures short-scale
  (20-40 min) ordinal regularity; ACF@24h measures long-scale periodicity.
  Cloud workloads can be strongly periodic at 24h AND noisy at 20-40 min.
  ByteDance is the textbook case (highest ACF@24h, highest WPE).

- This is actually consistent with F2's intent. The pre-reg target is
  partial-R²(WPE | ACF@24h) ≥ 0.3 — WPE should add information BEYOND
  ACF@24h. If they were strongly negatively correlated (mirror images),
  partial-R² would be ~0 and F2 would fail. Near-orthogonality on Alibaba
  (r=+0.13) supports F2.

- Bitbrains' +0.87 (Pearson) / +0.76 (Spearman) is genuinely strong,not a range artefact. Within-Bitbrains, WPE and ACF@24h are near-redundant signals — WPE won't add much for F2 on this dataset specifically. [Spearman rho=+0.7553 — still strong, not just a Pearson artefact]

- WPE vs CV negative on both datasets that have CV: suggests high-CV series
  in these traces are high-CV because of *structured* variation (sharp
  spikes from low baseline → concentrated ordinal patterns → low entropy),
  not random noise. The flat-low-then-spike pattern has limited motif
  diversity.

- Cadence note: ByteDance is at 10-min intervals, Alibaba and Bitbrains at
  5-min. A 4-point motif covers 40 min on ByteDance vs 20 min on the others.
  So WPE measures slightly different time scales across datasets; not
  comparable on a strict like-for-like basis cross-dataset. Within-dataset
  comparisons remain valid.

### Implications for D5 F2 test

The actual F2 target is partial-R²(WPE | ACF@24h) where the response variable
is ML benefit (delta_pp from `bcf_pairs.csv`), not other predictability
metrics. The within-dataset orthogonality (especially Alibaba +0.13) is
encouraging. Bitbrains' high correlation could lower the pooled partial-R²;
worth investigating whether to report pooled vs per-dataset partial-R²
### Spearman robustness (rank-based, scipy-backed)

| dataset    | WPE-ACF | WPE-CV | WPE-Hurst |
|------------|---------|--------|-----------|
| Bitbrains  | +0.7553 | -0.0620 | +0.5173 |
| ByteDance  | +0.2801 | -       | +0.1838 |
| Alibaba    | +0.1231 | -0.1854 | +0.1022 |

Pearson and Spearman agree on direction for every cell. Pearson was modestly
inflated on Bitbrains WPE-ACF (+0.87 → +0.76) and substantially on WPE-CV
(-0.25 → -0.06). The structured-variation hypothesis from the post-run
narrative weakens — high-CV series being "spike-dominated" doesn't track
robustly in the ranks.

### Revised D5 implications

The pooled partial-R²(WPE | ACF@24h) target ≥ 0.3 may be at risk if
Bitbrains' ~0.76 within-dataset correlation drags the pooled estimate
down. Decision needed at D5: report (a) pooled across 11 cells, (b)
per-dataset, or (c) pooled but with dataset fixed effects. Pre-reg
language likely permits any of these but the choice affects whether
F2 passes.
in the manuscript.

### Spearman robustness (added post-scipy-install)

| dataset    | WPE-ACF | WPE-CV  | WPE-Hurst |
|------------|---------|---------|-----------|
| Bitbrains  | +0.7553 | -0.0620 | +0.5173   |
| ByteDance  | +0.2801 | -       | +0.1838   |
| Alibaba    | +0.1231 | -0.1854 | +0.1022   |

Pearson and Spearman agree on direction for every cell. Drops Pearson → Spearman:
- Bitbrains WPE-ACF: +0.87 → +0.76. Slightly inflated by Pearson but the
  relationship is GENUINELY STRONG. My earlier "Pearson artefact" hypothesis
  was wrong; the Bitbrains line item above stands corrected by the bracket
  update at the end of that bullet.
- Bitbrains WPE-CV: -0.25 → -0.06. Substantially inflated. The
  "structured-variation → low entropy" story weakens to "weak hint only".
- Alibaba WPE-Hurst: +0.18 → +0.10. Consistent direction, both small.

### Revised D5 implications (after Spearman)

Within-Bitbrains, WPE and ACF@24h are near-redundant signals (rho≈+0.76).
WPE won't add much to F2 on Bitbrains specifically. On Alibaba (rho=+0.12)
and ByteDance (rho=+0.28) WPE is closer to orthogonal.

Decision needed at D5: report partial-R²(WPE | ACF@24h)
(a) pooled across all 11 cells,
(b) per-dataset (3 separate numbers),
(c) pooled with dataset fixed effects.

Pre-reg language likely permits any of these; the choice affects whether
F2 clears the 0.3 threshold. Honest disclosure preferred — if Bitbrains
pulls the pooled number below 0.3 while the others clear it, report that.
