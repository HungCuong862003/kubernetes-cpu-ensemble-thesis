# Memory Snapshot — 2026-05-31 (Phase F D9)

## Metadata

- Snapshot date: 2026-05-31 (Sunday, D9 of Phase F)
- Previous snapshot: `phase_f/memory_snapshots/memory_snapshot_2026-05-23.md`
- Coverage: deltas D2 through D9 (8 days)
- userMemories slots before snapshot: 30/30 full
- F0 lockdown: 9 of 10 days complete; D10 is F0 close
- Last commit at snapshot time: `6ea32e0` on `feature/live-demo`

## Project context (stable, see previous snapshot for baseline)

Jimmy (Phan Nguyen Hung Cuong, ITDSIU21078), CS bachelor's at International
University HCMIU, VNU-HCM. Supervisor Dr. Ho Long Van. Thesis: "Hybrid
Ensemble Learning for Proactive Resource Prediction in Kubernetes."
Defence target October 2026 (one-semester delay accepted from original
June 30, 2026).

Phase F is the multi-month extension post-first-submission. F0 lockdown
(D1-D10) is paperwork + pre-registered tests + design work. F1-F5
implementation starts D11+. Pre-registration thresholds (Dr. Ho written
acceptance 2026-05-22): F1 macro-F1 ≥ 0.55, F2 partial-R²(WPE|ACF@24h)
≥ 0.30, F3 Spearman ρ ≥ 0.6 AND |DFL-Pinball-τ| ≤ 5%.

## What happened D2-D9

**D2 (2026-05-24):** Phase F setup. Q-001 / Q-004 / Q-006 closed. New
question Q-007 (boundary condition verifier scope) raised.

**D3 (2026-05-25):** Q-008 / Q-009 closed via SSH read of HPA dominance
metrics; ERRATA-011 opened, DECISION-008 logged. Overleaf batch applied
for ERRATA-001 through 011 in one commit. Biblio audit started at
`phase_f/journal/biblio_audit_d3_notes.md` (schema + 3 placeholder rows
only; actual per-entry verification did NOT happen).

**D4 (2026-05-26):** WPE compute job ran on Vast.ai. Fadlallah 2013
weighted PE, m=4, τ=1, on train+val+test concatenation per series.
Outputs at `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv` (5000 /
142 / 93 series). ByteDance WPE median 0.95 (highest). DECISION-009
locked. Cadence asymmetry noted (ByteDance 10-min vs Alibaba/Bitbrains
5-min) — same WPE motif measures different time scales. D4 ALSO chose
F2 prep over biblio audit continuation; defensible reallocation of F0
slack but means biblio audit is still at schema-only.

**D5 (2026-05-27):** F2 pre-registered partial-R²(WPE | ACF@24h, horizon)
test executed. Both framings BELOW threshold:
- Headline per-cell (n=12): partial-R² = 0.0790, percentile CI [0.0000, 0.0790]
- Bitbrains per-VM (n=568): partial-R² = 0.0558, CI [0.0007, 0.2126]

Null reported per DECISION-005 (no post-hoc adjustment). Structural
diagnosis: R²_reduced (ACF@24h + horizon, no WPE) = 0.903. ACF@24h +
horizon already explain 90% of cell-level delta_pp variance; WPE has
nothing left to predict. WPE coefficient positive cross-dataset (+3.45
headline) — inverts the Pennekamp 2019 / Ponce-Flores 2020 prior that
PE captures predictability ACF misses. Honest finding: WPE and ACF@24h
are partial substitutes on cloud traces, not complements.

Q-007 closed via DECISION-011 (Anchor A): 3-key update to
`reports/tables/thesis_numbers.json` aligning verifier expected values
to NEW pool per-VM medians (delta_30min +1.53, delta_120min -2.81,
verdict reframed). Verifier audit 211/3 → 214/0. ByteDance stats
duplicate diff resolved byte-identical (md5 `a5e25823a28e10c6970c46e022e32ad4`).

DECISION-010 locked (F2 framing: per-cell pooled + Bitbrains per-VM,
A/B per-series deferred). DECISION-011 locked. ERRATA-012 added for
Ch4 BCF Bitbrains row reframing (NEW pool semantics), PENDING D6+
Overleaf application.

**D6 (2026-05-28):** Paperwork-only day. ERRATA-012 Overleaf application
and biblio audit resume both deferred to D10 F0-close batch per
`phase_f/journal/d6_deferred.md`. Rationale: both items batch well with
D10's existing Overleaf batch slot.

**D7 (2026-05-29):** F2 chapter outline sketch produced at
`phase_f/journal/f2_chapter_outline.md` (v3.1, 8 sections, claim +
anchor per section, no prose). Refined through 4 self-examination
passes producing 5/3/1/0 substantive-issue diminishing-returns curve.
Locked at v3.1 with 7 TBV items flagged.

**D8 (2026-05-30):** F2 outline promoted v3.1 → v4 defensible structure
at `phase_f/journal/f2_chapter_outline.md`. Two TBV items cleared:
- §5 h10 ordering: bcf_pairs.csv NNLS shows monotonic ByteDance > Alibaba >
  Bitbrains at ALL FOUR horizons (h10: +6.97/+0.25/-8.68; h30: +5.52/+0.43/-5.46;
  h60: +6.09/+1.33/-0.79; h120: +11.31/+4.64/+3.30 pp at F2 input scope)
- §5 M4-literature citation: Pennekamp et al. 2019 Ecological Monographs
  + Ponce-Flores et al. 2020 Entropy 22(1):89

Critical scope correction in v4: h30 monotonicity exception (Bitbrains
> Alibaba) is at NEW pool full-test scope (`cross_dataset_headline_v2.csv`),
NOT F2's input scope (`bcf_pairs.csv` OLD pool). v3.1 had conflated
these. v4 discloses both aggregations explicitly. 5 TBV items remain
for D11+ source reads (all method-detail).

DECISION-012 locked: F1 router design at cell-level granularity, 4
features (ACF@24h + horizon_min + CV + ACF@1h), training on 12 NNLS
cells from `bcf_pairs.csv` with labels from `leaderboard_v1.csv`
winner tally (Chronos-2:6/TimesFM:3/Granite-TTM:2/NNLS:1), evaluation
LOO-cell CV with macro-F1 ≥ 0.55 per DECISION-005. Baseline
always-predict-Chronos-2 yields macro-F1 = 0.167 (hand-computed:
per-class F1 = [0, 0.667, 0, 0]; macro = 0.167). Classifier architecture
deferred to F1 implementation D11+ among multinomial logistic regression
with L2, k-NN k=3, shallow tree depth ≤ 3.

Sequence-cleanup commit at end of D8: removed two stale journal files
(`d6_errata012_biblio_resume.md` pre-deferral draft asserting work that
didn't happen, `d7_f2_out.md` byte-identical duplicate of d7_f2_outline).
Commit `6ea32e0` is canonical D6-D8 end-state.

**D9 (2026-05-31):** This snapshot.

## Decisions logged through D8

| # | Date | Lock | Topic |
|---|---|---|---|
| 001 | early Phase F | LOCKED | (see previous snapshot) |
| 002 | early Phase F | LOCKED | (see previous snapshot) |
| 003 | early Phase F | LOCKED | (see previous snapshot) |
| 004 | early Phase F | LOCKED | (see previous snapshot) |
| 005 | 2026-05-22 | LOCKED | Pre-registration thresholds F1/F2/F3 (Dr. Ho written acceptance) |
| 006 | early Phase F | LOCKED | (see previous snapshot) |
| 007 | early Phase F | LOCKED | (see previous snapshot) |
| 008 | 2026-05-25 | LOCKED | Full retirement of 533/640 figure with dual-metric replacement |
| 009 | 2026-05-26 | LOCKED | WPE = Fadlallah weighted PE, m=4, τ=1 |
| 010 | 2026-05-27 | LOCKED | F2 framing: per-cell pooled (n=12) + Bitbrains per-VM (n=568); A/B per-series deferred |
| 011 | 2026-05-27 | IMPLEMENTED | Q-007 Anchor A: align verifier to NEW pool per-VM medians via 3-key JSON update |
| 012 | 2026-05-30 | LOCKED | F1 router design lock: 4-feature classifier at cell level, LOO-cell CV, baseline 0.167; implementation D11+ |

## ERRATA status through D8

- 001 through 011: APPLIED 2026-05-25 (Overleaf batch commit)
- **012: PENDING** (Ch4 BCF Bitbrains row OLD → NEW pool per-VM medians; D10 target application)
- 013+: candidates from D10 biblio audit (may open if audit produces findings; biblio audit may itself be deferred to F1+ given current schema-only state)

## Open questions

| Q-ID | Description | Blocking? |
|---|---|---|
| Q-002 | Vast.ai C.37124280 fate | No |
| Q-003 | Public + MIT repo supervisor approval | No |

Q-001/004/006 closed D2. Q-007 closed D5. Q-008/009 closed D3.

## Canonical files post-D8

**Phase F state files:**
- `phase_f/THESIS_STATE.md` (D8 close, refreshed D9)
- `phase_f/DECISIONS.md` (12 decisions logged)
- `phase_f/ERRATA.md` (12 rows, 11 applied, 1 pending)
- `phase_f/SYNC_PROTOCOL.md` (Revisions 4-6 active)

**F2 chapter outline:**
- `phase_f/journal/f2_chapter_outline.md` (v4 D8 finalised, 5 D11+ TBV items)

**F1 design:**
- `phase_f/journal/f1_prep_scope.md` (DECISION-012 design, baseline 0.167)

**F2 data:**
- `phase_f/data/wpe_{alibaba,bitbrains,bytedance}.csv` (D4 WPE per series)
- `phase_f/data/f2_partial_r2_results.csv` (D5, 2 rows: headline + Bitbrains)
- `phase_f/data/per_series_deltas_bitbrains.csv` (D5 intermediate, 568 rows)

**F2 scripts:**
- `phase_f/scripts/compute_wpe.py` (D4)
- `phase_f/scripts/f2_partial_r2.py` (D5; source reads needed at D11+ to clear remaining TBVs)

**Biblio audit:**
- `phase_f/journal/biblio_audit_d3_notes.md` — schema + 3 placeholder rows.
  Substantive audit work has NOT happened; status at D9 is schema-only.

**Inherited canonical (unchanged D2-D9):**
- `results/bcf/bcf_pairs.csv` (12 NNLS cells, F2 input)
- `results/bcf/bcf_pooled_3model.json` (BCF backbone, AUC 0.80, p=0.0097)
- `results/foundation_comparison/leaderboard_v1.csv` (§4.7 canonical, 4-model tally)
- `results/foundation_comparison/cross_dataset_headline_v2.csv` (NEW pool full-test scope)
- `results/bcf_v2/hpa_v4_dominance_per_dataset.csv` (HPA v4 canonical, max_replicas=1000)
- `reports/tables/thesis_numbers.json` (NEW pool Bitbrains BCF anchors as of Q-007 fix)

## Critical scope distinctions to remember

**Two scopes for cross-dataset deltas — do not conflate:**
- **OLD pool / F2 input scope** = `bcf_pairs.csv` NNLS rows. Bitbrains uses per-VM median. Cross-dataset ordering monotonic at ALL four horizons (ByteDance > Alibaba > Bitbrains).
- **NEW pool full-test scope** = `cross_dataset_headline_v2.csv`. Bitbrains pooled. h30 exception exists (Bitbrains +4.47 > Alibaba +3.39 pp despite lower ACF@24h).
- F2 partial-R² result used OLD pool scope. The h30 exception lives in a different aggregation and does NOT affect F2's null.

**Two scopes for §4.7 winner tally — also do not conflate:**
- `leaderboard_v1.csv` at K=20/K=50 subsample scope: Chronos-2:6, TimesFM:3, Granite-TTM:2, NNLS:1 (canonical for §4.7 manuscript, used for F1 labels in DECISION-012)
- `cross_dataset_headline_v2.csv` at full-test scope: NNLS:7, Granite-TTM:2, Chronos-2:2, TimesFM:1 (for §5 disclosure; not §4.7)

**F1 baseline math:**
- Always-predict-Chronos-2 yields:
  - Accuracy = 6/12 = 0.500 (would be marginal vs threshold)
  - Macro-F1 = 0.167 (per-class F1 = [0, 0.667, 0, 0]; informative vs 0.55 threshold)
- Macro-F1 chosen for DECISION-012 specifically because accuracy would make the test marginal.

**ERRATA-012 substitution text (for D10 application):**
- Table 4.10 Bitbrains row:
  - delta @30min: -5.46pp → +1.53pp
  - delta @120min: +3.30pp → -2.81pp
  - verdict: "ML wins only @120min" → per-VM median negative at h120; per-VM win rates 61% / 35% at h30 / h120
- §4.X prose: ~3 sentences disclosing all 4 per-VM medians (-16.66 / +1.53 / -1.99 / -2.81), per-VM win rates (61% h30, 35% h120), OLD→NEW aggregation scope shift with production-relevance rationale.

## Lessons learned D2-D9

1. **Verify cell counts from canonical inputs, not from prior notes** (D5 close lesson #4). The "11 cells" framing was carried wrong from earlier planning; bcf_pairs.csv has 12 NNLS cells. Generalises to: verify ALL numerical claims at point of citation, not from memory.
2. **Read canonical CSVs at finalisation pass even if sketch appeared clean** (D8 lesson). Four self-examination passes at D7 missed the §5 scope conflation between OLD pool bcf_pairs and NEW pool cross_dataset_headline. The bcf_pairs.csv read at D8 caught it.
3. **Diminishing returns curve for self-examination passes** (D8 lesson). Substantive-issue counts: pass 1 = 5, pass 2 = 3, pass 3 = 1, pass 4 = 0. Marginal value of pass 5+ is near-zero on a structurally clean draft.
4. **Pre-registration discipline applies to operationalisations, not just metrics** (D8 lesson). DECISION-005 locked the F2 metric; DECISION-010 locked the operationalisations (per-cell + per-VM framings). Both are pre-registration-honour material.
5. **Macro-F1 vs accuracy is a methodology choice that affects test informativeness** (D8 lesson). On imbalanced 4-class problems, macro-F1 makes baselines harder to clear and tests more discriminative.
6. **Schema-only progress isn't audit progress** (D9 lesson, surfaced reading state for snapshot). The biblio audit at D3 set up the schema but the four days since (D4-D8) chose F2 prep, F2 execution, paperwork, F2 outline, F1 design. Honest D10 framing must acknowledge biblio audit hasn't substantively started, not market the schema as "audit in progress."

## Pre-mortem for D10 and D11+

**D10 (F0 close batch):**
- ERRATA-012 application may surface Ch4 prose mismatches not covered by the original errata. Treat as ERRATA-013+ candidates; do not absorb silently.
- Biblio audit is schema-only at D9. D10 will need a real decision: do it (paste-by-paste verification, several hours) or defer with honest "audit deferred to F1+" framing. Half-finished is the worst option — defence can't market a half-audited bibliography as audited.

**D11+ (F1 implementation):**
- F1 honest expectation range: macro-F1 ∈ [0.30, 0.65]. Could null analogously to F2.
- If F1 lands in [0.30, 0.55], the chapter contribution is structural ("dataset-constant features dominate; router signal limited by n=12") parallel to F2's "ACF saturates predictability axis".
- F1 implementation also clears the 5 remaining F2 outline TBVs (script-source reads of f2_partial_r2.py during the F1 work give incidental access).

**F3 (future):**
- Cost-asymmetric quantile fine-tune of Chronos-2. Needs Vast.ai GPU compute (vs F1's local-pandas option).
- Pre-reg thresholds: Spearman ρ ≥ 0.6 AND |DFL-Pinball-τ| ≤ 5%.

## Forward agenda

- **D10 (2026-06-01, Mon):** F0 close batch — ERRATA-012 Overleaf application, biblio audit completion or final deferral, F0→F1 transition state file refresh, optional D6-D10 combined handoff.
- **D11-D20 (2026-06-02 onward):** F1 implementation per DECISION-012. Classifier architecture picked at start. LOO-cell CV runs. Macro-F1 reported. F2 outline TBVs cleared incidentally.
- **F3 future:** quantile fine-tune of Chronos-2, integration with OptScaler + AHPA per Phase F scope lock.
- **Defence:** ~October 2026.