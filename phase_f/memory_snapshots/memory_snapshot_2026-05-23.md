# Claude Memory Snapshot — kubernetes-cpu-ensemble-thesis project

**Snapshot date:** 2026-05-23 (Phase F Day 1 close)
**Project scope:** kubernetes-cpu-ensemble-thesis (original project)
**Slot status:** 30/30 (at limit — adds blocked, only replacement possible)
**Source:** `memory_user_edits view` against the project's memory store
**Format:** numbered list matching slot indices

---

## Why this file exists

Project memory has a hard cap of 30 slots. The project is at the cap. Future
canonical updates require replacing existing slots, which silently destroys
previous content. This file is a recoverable archive of the current memory
state so individual slot contents can be restored manually if the wrong slot
is replaced, or if the project's memory is ever cleared.

---

## Restoration instructions

**If memories are lost or corrupted:**

1. Open this file in a chat with Claude in the same project.
2. Paste the numbered entry as the desired slot's content.
3. Use the `memory_user_edits` tool with `command="replace"`, the target
   `line_number`, and the original text as `replacement`.

**Per-entry length limit (write):** ~500 characters. Some entries below
exceed 500 chars (they predate that limit). If restoring those, trim to
fit while preserving canonical facts (file names, AUC values, file paths,
dataset names, and numerical anchors).

**If the project is at 30/30 and a deleted memory needs restoring:**
Identify the least load-bearing current entry (e.g., a process record
that's already documented in `phase_f/decisions/`), replace that slot
with the restored content.

---

## Today's memory edits (2026-05-23 audit trail)

Three slots were modified during Phase F Day 1 work:

### Slot #4 — touched twice, net unchanged
- **Original:** V4 HPA CANONICAL dominance counts (the entry now stored)
- **Mid-session:** briefly replaced with Day 1 audit findings (mistake)
- **Restored:** V4 HPA CANONICAL content (recovered from the prior `replace` operation's diff output)

### Slot #17 — replaced
- **Previous content (overwritten 2026-05-23):**
  > "2026-05-23 OVERLEAF PENDING from Phase C: (1) §6.1 HPA errata template
  > (a/b/c) — replace submitted PDF '533/640' with v4 numbers; (2) §3.8 +
  > §4.11 cite bcf_v2/hpa_simulation_*_v4.csv not v2; (3) §4.10 add
  > residual-pool disclosure paragraph (CQR=v1, AgACI/split=v2); (4) §5.4
  > optional sharpening with et-chronos2 pairwise pattern; (5) optional
  > CQR re-run on v2 residuals (~4hr Vast.ai) to close §4.10 mixing concern."
- **Rationale for replacement:** Items (1)-(3) are now captured by the Day 1
  10-Overleaf-edits queue; items (4)-(5) are deferred and tracked in journal
  rather than memory. Day 1 audit findings displaced this entry because they
  are operationally hotter for the next 1-2 weeks of Overleaf work.

### Slot #30 — replaced
- **Previous content (overwritten 2026-05-23):**
  > "2026-05-22 PHASE A MANUSCRIPT INTEGRATION COMPLETE. LaTeX drafts: Ch4
  > (§4.7 + §4.8 Toto inserts), Ch5 (§5.3 corpus-specificity rejection +
  > leaderboard limitations 3 paragraphs), Ch6 (§6.1 + §6.2 limitation 4 +
  > §6.3 item 3 Toto inserts), new Appendix C ~1800 words Toto sensitivity
  > check. Existing Comp Cost appendix renumber from C to E. Citation
  > datadog-2024-toto verified at arXiv:2407.07874 (salvaged from removed
  > bib memory)."
- **Rationale for replacement:** Phase A integration history is now reflected
  in the current chapter LaTeX drafts (memory #20). Discovery of the thesis
  git repo is operationally critical for Phase F dual-canonical store and
  was not recorded anywhere else.

---

## Current 30 memories (verbatim)

### 1. Thesis timeline (IU Semester II 2025-2026)

> Thesis timeline UPDATED per IU academic schedule for Semester II 2025-2026:
> first-version submission to department is May 4-8, 2026 (not April 24-29).
> After May 8 Jimmy has a ~30-day additional-work window before defense prep.
> Defense review period is May 11-25. Committee CV submission June 1. Defense
> itself is June 8-26, 2026. Results submission deadline June 30, 2026.
> Supersedes the earlier April 24-29 submission window stored in thesis context.

### 2. BCF canonical 3-model statistics

> BCF canonical 3-model (NNLS+Chronos-2+TimesFM) at results/bcf/bcf_pooled_3model.json:
> AUC=0.80, PERCENTILE CI [0.7097, 0.8871] (NOT BCa — JSON notes "BCa degenerate
> for binary classifier"), n=36, p=0.0097 (memory previously 0.011 was wrong).
> Per-model AUC: NNLS 0.833, Chronos-2 0.773, TimesFM 0.800, Granite 0.500.
> Granite EXCLUDED. 4-model justification at bcf_pooled_results.json: AUC=0.667
> BCa CI [0.486, 0.793] n=48 p=0.0511. Predicate: ACF@24h>0.2 AND h≥30min.

### 3. HPA grid v4 canonical structure

> HPA grid v4 canonical at bcf_v2/hpa_simulation_*_v4.csv (max_replicas=1000):
> target_util ∈ {0.5,0.6,0.7,0.8} (max 0.8 NOT 0.9); safety_margin
> ∈ {1.0,1.067,...,1.6} MULTIPLIER (not fractional offset). Per cell: 200 rows
> = 4 target_util × 10 safety_margin × 5 strategies (1 ML-Proactive + 4
> Reactive lag=1..4). Reactive lag variants differ materially. v2/v3 at
> max_replicas=100 superseded (c3_saturation_verdict.md: 12-40% saturation).

### 4. V4 HPA canonical dominance counts (per-dataset)

> 2026-05-23 V4 HPA CANONICAL at max_replicas=1000. ML strict per horizon
> (h10/30/60/120): ali 100%/87.5%/37.5%/2.5% = 40/35/15/1; bb 100%/0/0/0
> = 40/0/0/0; byt n/a, 82.5%/80%/72.5% = 33/32/29. Reactive dom %: ali
> 75/34.4/13.1/0.6; bb 73.1/0/0/0; byt 78.1/77.5/70. 11-cell totals: 225
> ML strict / 675 react dom. Source: hpa_v4_dominance_per_dataset.csv.
> Submitted PDF 533/640 (v1-sprint) superseded.

### 5. Boundary condition table aggregation footgun

> boundary_condition_table_corrected.csv aggregation is MIXED by design
> (per bcf_generalization.py line 23): Alibaba and ByteDance use pooled
> (ensemble_R2 - naive_R2); Bitbrains uses per-VM median (n=156). Caption
> must surface this. ByteDance row updated 2026-05-01 to pooled hetero
> +5.52pp (30min) / +11.31pp (120min); prior +5.26/+11.45 were stale.
> Footgun: task0_data_corrections.py only writes Alibaba+Bitbrains rows
> — ByteDance is hand-maintained, re-running task0 silently deletes it.

### 6. NNLS weights — two sources, different scopes

> NNLS weights have TWO sources, different scopes. (1) run.log = PRODUCTION
> canonical for §3.6 ensemble defn + §5.1 R² (matches comparison_table.csv
> hetero R²: 0.9213/0.8404/0.8011/0.7642). 10min: ET=0.976 BiLSTM=N/A (OOF
> shared-mem fail); 30/60/120: ET=0.728/0.784/0.759 BiLSTM=0.220/0.216/0.240.
> (2) meta_learner_comparison.csv = POST-HOC reconstructed BiLSTM OOF,
> canonical ONLY for §3.7 method comparison (NNLS vs Ridge vs Bates-Granger
> vs Simple-Avg). 10min CSV: ET=0.339 BiLSTM=0.654.

### 7. Chronos-2 win-count framing (manuscript citations)

> Chronos-2 win-count framing (manuscript MUST label every citation): 11/12
> vs naive (only Alibaba 10min is -0.10pp delta), 6/12 vs all 4 foundation
> models. Per-cell winners across NNLS+Chronos-2+TimesFM+Granite-TTM:
> Chronos-2 6, TimesFM 3, Granite-TTM 2, NNLS 1. Source: leaderboard_v1.csv.
> Also: Alibaba 120min actual best is TimesFM (0.7726), not the ensemble
> (NNLS 0.7584); Chronos-2 0.7612 also beats NNLS by +0.28pp. Old
> '+1.21pp ensemble win' line is dead.

### 8. Win rate — two metrics, both valid

> Win rate = TWO metrics, both valid. (1) Timestep (stratified_skill.csv):
> % test points ensemble<naive — 17.26/31.53/38.64/43.45 across 10/30/60/120min.
> (2) Container (cv_stratified_skill.csv, N-weighted n=4637): % containers
> ensemble has lower MAE — 0.00/19.34/41.66/52.19. Best ML cell: CV 0.3-0.5
> bin (53.1% at 60min, 62.3% at 120min). §5.3 cites timestep as headline;
> cv per-bin in subsection. Earlier '0/18.3/39.3/49.2' is stale.

### 9. Chronos-2 scale bug fix (Phase B B6, 2026-05-19)

> 2026-05-19 CHRONOS-2 SCALE BUG FIX (Phase B B6): MEMBER_SCALE['chronos2']
> was 'CPU' but Chronos-2 outputs residuals everywhere. Fixed in
> B6_apply_nnls_test_v3.py via sed to 'residual'. Re-ran B6 for all 3 datasets.
> Canonical: results/foundation_comparison/cross_dataset_headline_v2.csv.
> NEW vs naive (pp): Bitbrains +0.97/+4.47/+1.91/-1.26; Alibaba
> +0.55/+3.39/+5.65/+9.34; ByteDance n/a/+19.91/+21.86/+25.43. NEW beats
> OLD on every cell.

### 10. Leaderboard scope distinction

> SCOPE: leaderboard_v1.csv NNLS = OLD-pool at 98k-spine subset;
> cross_dataset_headline_v2.csv NEW NNLS = full test (~200k Bitbrains,
> 1.65M Alibaba, 64k ByteDance). Both valid, different scopes. TimesFM
> 0.7726 winning Alibaba 120min holds at leaderboard scope. To add NEW pool
> to leaderboard apples-to-apples requires applying NEW NNLS weights to
> 98k-subset predictions.

### 11. ACF@24h monotonicity exception

> ACF@24h monotonicity exception (post-fix cross-dataset): ML benefit ordered
> by ACF@24h holds at h60 and h120 (ByteDance > Alibaba > Bitbrains) but
> INVERTS at h30 (Bitbrains +4.47 > Alibaba +3.39 by 1.08pp despite lower
> ACF). Refined claim for §6: "at horizons ≥ 60min, ACF@24h orders ML
> benefit monotonically across all three datasets; at h30 the ordering
> between Alibaba and Bitbrains inverts modestly". Do NOT bring the
> unqualified version to supervisor.

### 12. Per-VM aggregation finding (Bitbrains NEW pool, 2026-05-19)

> 2026-05-19 PER-VM AGGREGATION FINDING (Bitbrains NEW pool): per-VM
> median NEW Δ vs naive = h10:-16.66, h30:+1.53, h60:-1.99, h120:-2.81pp
> from 142 VMs. POOLED full-test gave +0.97/+4.47/+1.91/-1.26pp — opposite
> signs at h10/h60. Pooled is dominated by high-variance VMs; per-VM
> median is the production-relevant aggregation. NEW vs OLD per-VM: NEW
> beats OLD only at h30 (+7pp). At h10/h60/h120 OLD pool is BETTER per-VM
> by 1-8pp. "NEW beats OLD everywhere" claim was pooled-only. Report
> BOTH in §5.

### 13. Manuscript structure committed (6 chapters, 2026-05-22)

> 2026-05-22 MANUSCRIPT STRUCTURE COMMITTED: 6 chapters (not 8). Ch1 Intro
> 6pp / Ch2 Related 11pp / Ch3 Methodology 13pp / Ch4 Impl&Results 24pp /
> Ch5 Discussion&Eval 25pp / Ch6 Conclusion 6pp = 85 body pp. Bound
> ~110-120pp. SCSE cap unconfirmed (75 anchor from Business School).
> LaTeX: chapters/01-06.tex, appendices/A-C, frontmatter/, preamble/.
> Phase D in: Ch3 §3.7, Ch5 §5.7, Ch5 §5.6. chapter_04_phaseD_complete.tex
> (13kw, /mnt/user-data/outputs/) now reference-only.

### 14. C2 canonical error correlation (2026-05-22)

> 2026-05-22 C2 CANONICAL from c2_verdict.md (verified Vast.ai SSH): pooled
> ρ̄=0.6562 container-clustered bootstrap (1000 resamples), CI [0.6499,0.6620],
> verdict HOLDS. 10/11 cells HOLD + 1/11 PARTIAL (Alibaba h10 at 0.8434,
> driven by et-chronos2 pair at 0.9051). Per-dataset ranges: Alibaba 0.71-0.84,
> Bitbrains 0.61-0.74, Bytedance 0.44-0.48. Per-horizon stratified:
> 0.761/0.692/0.652/0.617 (h10/30/60/120). Prior 'overall median 0.69 mean
> 0.65' framing was per-cell summary, not pooled headline.

### 15. LOO + pairwise mechanism (2026-05-22)

> 2026-05-22 LOO + PAIRWISE MECHANISM. LOO ablation
> (loo_ablation_new_pool_v2.csv) Alibaba: drop_chronos2 -2.09/-3.57/-3.43pp,
> drop_et and drop_nhits ~0pp, only_chronos2 within -0.33pp of ALL —
> Chronos-2 load-bearing. Mechanism via pairwise (c2_correlations_per_cell.csv):
> et-chronos2 corr 0.91→0.76 across Alibaba h10→h120 (dominant);
> decorrelates Bytedance (0.38-0.40); intermediate Bitbrains (0.59-0.74).
> Reframe §5.4 'diverse ensemble' → 'Chronos-2 + BCF selector'.

### 16. leaderboard_v1.csv canonical scope

> leaderboard_v1.csv (NNLS+Chronos-2+TimesFM+Granite-TTM, K=20/K=50
> subsample scope) is canonical §4.7. Per-cell winner tally: Chronos-2 6,
> TimesFM 3, Granite-TTM 2, NNLS 1. Verified against submitted PDF Table
> 4.8, OUTLINE.md §5.5, THESIS_HANDOFF.md, finalize_leaderboard_v1.py.
> leaderboard_v3_wide.csv (Toto in, Granite out) was a planning artefact
> that did NOT enter the chapter. Toto in Appendix C as K=20 robustness check.

### 17. Day 1 audit findings (2026-05-23)

> 2026-05-23 D1 AUDIT data-verified Vast.ai: BB30min CI [+3.54,+5.42]
> excludes Ali +3.39 (Ch5 §5.1 'CI crosses' false). 533/640 verified strict
> ML-Pro vs Reactive on c3_pareto_points (Ali pair-level) — per-h
> 157/158/127/91 NOT Ch6 128/156/157/92. DECISION-007 UNUSED, BCa footnote
> enhancement Ch3 §3.8+Ch4 §4.8. 10 Overleaf edits: Ch3 p+JSON+percentile+
> HPAfile, Ch4 row-count qualifier, Ch5 CI+lag-1, Ch6 per-h.

**NOTE:** Per other-Claude review on 2026-05-23, the "157/158/127/91" numbers
are themselves from a _SUPERSEDED file (c3_pareto_points). Both the
submitted PDF's 128/156/157/92 AND this entry's 157/158/127/91 need re-derivation
from canonical v4 files. Edit #10 should not land until v4 numbers are
computed.

### 18. Phase F scope locked (2026-05-22)

> 2026-05-22 PHASE F SCOPE LOCKED with Dr. Ho Long Van WRITTEN acceptance
> (received 2026-05-22): F1 router, F2 predictability WPE/Ω/LZ/DFA, F3
> cost-asymmetric quantile FT of Chronos-2 + F4 integration with
> OptScaler+AHPA + F5 6 new chapters. ~135 days from F0 start, defense
> ~Oct 2026 (one-sem delay). Pre-registration: F1 macro-F1≥0.55, F2
> partial-R²(WPE|ACF@24h)≥0.3, F3 Spearman ρ≥0.6 AND |DFL-Pinball-τ|≤5%.
> F0 Day 1 unblocked.

### 19. BCa DECISION-007 UNUSED (2026-05-23)

> 2026-05-23 BCa DECISION-007 UNUSED. Manuscript writes percentile bootstrap
> in §3.8 with one-sentence justification: BCa attempted but degenerate
> (acceleration term unreliable from binary classifier × binary outcome
> producing few unique AUC values across resamples). Cite Efron-Tibshirani
> 1993. No errata needed — manuscript not yet written with BCa label;
> canonical JSON bcf_pooled_3model.json declares
> ci_method='percentile (binary classifier; BCa degenerate)'.

### 20. All 6 chapters draft state (2026-05-22)

> 2026-05-22 ALL 6 CHAPTERS DRAFT (/mnt/user-data/outputs/01-06-*.tex).
> Words: Ch1 2975/Ch2 3619/Ch3 8625/Ch4 13393/Ch5 4843/Ch6 2694. Resolved
> in drafts: §4.11 HPA→hpa_simulation_*_v4.csv (v3 superseded 2026-05-23);
> §4.10 residual disclosure (CQR=v1, AgACI/split=v2); 'Chronos-2 + BCF
> selector' Ch3-Ch6 reframing; v2 bias→residual_diagnostics_new.csv.
> BCF percentile CI [0.7097,0.8871] p=0.0097. §3.9 cluster-bootstrap DM added.

### 21. HPA canonical state Vast.ai post-audit (2026-05-23)

> 2026-05-23 HPA CANONICAL state Vast.ai post-audit. Cite from:
> c3_hpa_v4_verdict.md, c3_saturation_verdict.md,
> hpa_v4_dominance_per_dataset.csv, hpa_simulation_*_v4.csv (3 datasets),
> hpa_pareto_*_v4.pdf. Renamed _SUPERSEDED: c3_hpa_dominance_verdict,
> c3_hpa_full_verdict, c3_pareto_points, hpa_v3_dominance_per_dataset,
> hpa_simulation_bytedance_v3, hpa_pareto_bytedance_v3.
> README_HPA_CANONICAL.md added. All in results/bcf_v2/.

### 22. Imputation rates post-Issue-1-fix

> IMPUTATION RATES post-Issue-1-fix (corrects 'below 5%' stale memory):
> Alibaba 7.9-15.6%, Bitbrains 24.4-25.4%, ByteDance 17.3-17.6%.
> Bitbrains/ByteDance reflect NeuralForecast internal CV baseline (~1 in
> 4 / ~1 in 5 predictions are residual-mean fallback), not Issue-1 bug
> residue. Issue-1 fix only addressed Alibaba (was 92-98% pre-fix).
> Manuscript §3.5 + Appendix D must use these corrected rates. Errata
> sheet must note correction vs submitted draft.

### 23. Toto Alibaba full coverage (2026-05-21)

> 2026-05-21 TOTO ALIBABA full coverage (4921 containers K=20 seed=42,
> n~98K/horizon, 185.5min RTX4090). R² h10/30/60/120:
> 0.9173/0.8513/0.8205/0.7586. Vs NNLS: NNLS wins h10/h120 (+0.40/+0.56pp);
> Toto wins h30/h60 (+1.09/+1.94pp). Vs other foundations: C2 wins h30,
> TFM wins h60/h120, NNLS wins h10. leaderboard_v3 rebuilt; tally
> unchanged (C2:6 TFM:3 NNLS:2 Toto:1). Asymmetry n=20K→98K resolved.

### 24. Verifier repointed v4 HPA (2026-05-23)

> 2026-05-23 VERIFIER REPOINTED v4 HPA. Score 211/3 (prior 184/3 had dead
> HPA path). Line 37 HPA=hpa_simulation_v2.csv (DEAD) →
> bcf_v2/hpa_v4_dominance_per_dataset.csv. check_section_9_hpa rewritten
> via /tmp/new_check_hpa.txt + bcf_v2/hpa_v4_anchors.json. 52 new HPA PASS
> (3 struct + 5 aggregates + 44 per-cell). 3 FAILs = Bitbrains BCF v1/v2
> errata, documented. Backup: verify_foundation.py.bak_pre_v4_repair.

**NOTE:** Day 1 summary reported 184/3, which contradicts this 211/3.
Re-run verify_foundation.py on C.37423026 to settle which is current.

### 25. BCF v2 AUC (3-member pool, 2026)

> BCF v2 AUC = 0.800 CI [0.65, 0.95] (post-Phase-B-redesign 3-member pool
> ExtraTrees+N-HiTS+Chronos-2, 11 cells). Distinct from memory #2 (v1
> 3-model NNLS+Chronos-2+TimesFM, 36 pairs, AUC=0.80 CI [0.70,0.88]) —
> same predicate, different pools/n. Submitted PDF cites NNLS AUC=0.833
> (v1, 12 cells). All three statistically indistinguishable. Errata sheet
> treats v2 as routine robustness. Threshold sweep [0.15, 0.25] stable
> at 0.800. Source: bcf_per_model_auc_v2.csv.

### 26. Phase-C C3 re-opened on v4 supersession (2026-05-23)

> 2026-05-23 PHASE-C C3 RE-OPENED on v4 supersession. Earlier 'citation
> right' was incomplete — manuscript cites hpa_simulation_v2.csv (v1-sprint,
> max_replicas=100, saturating 12-40% of grid). c3_hpa_v4_verdict.md
> directs 'Cite v4 as canonical; v2/v3 superseded'. Errata required: §6.1
> 533/640 superseded. Canonical Alibaba v4: 91/160 ML strict (40/35/15/1)
> OR 197/640 reactive dom (120/55/21/1). C2/C4/C5 unchanged.

### 27. Two §4.7 tallies — different scopes

> 2026-05-22 TWO §4.7 TALLIES — different scopes, do not conflate:
> (1) 'Chronos-2 6/TimesFM 3/Granite-TTM 2/NNLS 1' = canonical,
> leaderboard_v1.csv at K=20/K=50 same-sample subsample scope, matches
> submitted PDF Table 4.8 and Phase A chapter draft.
> (2) 'NNLS 7/Granite 2/Chronos-2 2/TimesFM 1' (memory 15) = NEW-pool
> apples-to-apples at full-test scope after MEMBER_SCALE fix. Manuscript
> §4.7 uses (1).

### 28. Phase E Path B closed (2026-05-22)

> 2026-05-22 PHASE E PATH B CLOSED. Three cells, three positive proactive
> lead times: alibaba c_64693 off248 +82s, bitbrains bb_276 off35 +70s,
> bytedance bd_instance_38 off5 +146s, mean +99.3s. Three procedural
> fixes: pre-declared forecast-cm with empty data (race eliminated),
> warmup-zero protocol (HPA t0 = metrics-ready, not kubectl-apply),
> Autopilot temporal floor α=0.8 over recent_max. Warmup tax measured:
> 58.6s/70.9s/69.7s, within published 60-90s envelope.

### 29. Phase D D4 completed on TEST (2026-05-22)

> 2026-05-22 PHASE D D4 COMPLETED on TEST: cluster-bootstrap DM (1000
> resamples container-clustered) + Friedman v2 + Holm correction, 44
> tests (ensemble vs naive/ET/N-HiTS/Chronos-2 × 11 cells). Headline
> 38W/3T/3L. Predicate-stratified: 23/1/0 in positive (6 cells: Ali+BD
> h≥30), 15/2/3 in negative (5 cells: Ali h10 + 4 BB). 3 losses at BB
> long h vs N-HiTS/Chronos-2. Friedman ranks 1.87/1.68/1.67/1.65 (ens #1).
> Source: results/bcf_v2/d4_*_TEST.{csv,md}.

### 30. Thesis git repo discovery (2026-05-23)

> Thesis git repo at github.com/HungCuong862003/kubernetes-cpu-ensemble-thesis
> (Public, MIT). Initial commit ~2026-05-02 = one-shot Drive dump
> (data/docs/logs/models/notebooks/references/reports/results/src/submission/
> tests/thesis folders). Dormant until Phase F. Distinct from dashboard repo
> (HungCuong862003/hpa-thesis-dashboard). Drive remains canonical for DATA;
> git canonical for CODE + work-record
> (phase_f/decisions/journal/states/deliverables).

---

## Memory hygiene recommendations for Phase F

The project is at 30/30. Without active curation, slot churn over 135
days will silently delete operationally critical context. Adopt the
following:

### 1. Daily memory snapshots

At end of each Phase F day, export `memory_user_edits view` output to:

```
phase_f/memory_snapshots/day_NNN.md
```

Commit to the thesis git repo with the day's journal entry. Diffs
between snapshots show what changed and why. Cheap insurance against
accidental overwrites.

### 2. Two-tier rule for what enters memory

| Tier | Belongs in memory | Belongs in git/files instead |
|------|-------------------|------------------------------|
| Canonical numbers | AUC, R², dominance counts, NNLS weights | Per-cell metric tables (in result CSVs) |
| File paths | Path to current canonical version | History of which path was canonical when |
| Decisions | The current outcome | Why it was decided, alternatives considered |
| Status flags | "v4 canonical, v3 superseded" | The supersession history and rationale |

Memory holds *what is true now*. Git holds *what was true and why*.

### 3. Replacement protocol

Before calling `memory_user_edits replace`:

1. Identify the slot most safely evictable (process records duplicated
   in `phase_f/decisions/` are top candidates).
2. Snapshot the slot's current content to today's `day_NNN.md` before
   overwriting (the diff will appear in the next commit anyway, but
   capturing it explicitly in the journal avoids confusion later).
3. State the rationale for replacement in the journal.

### 4. Memory budget for Phase F sub-phases

A rough plan for the next 135 days assuming 30/30 stays the cap:

- **F0 (Days 1-10):** 0-2 net new memory entries (audit findings, key
  state). Mostly snapshots, no churn.
- **F1 router (Days 11-40):** ~3 new entries (router architecture
  decisions, headline macro-F1, pre-registration result). 3 evictions
  needed.
- **F2 predictability (Days 41-70):** ~3 new entries (WPE/Ω/LZ/DFA
  canonical values, partial-R² result). 3 more evictions.
- **F3 cost-asymmetric FT (Days 71-100):** ~2 entries (FT hyperparameters,
  Spearman ρ result). 2 evictions.
- **F4 integration (Days 101-120):** ~2 entries (integration result,
  OptScaler/AHPA versions used). 2 evictions.
- **F5 chapters (Days 121-135):** ~2 entries (final chapter structure,
  defense slide canonical state). 2 evictions.

Total churn: ~12 evictions across 135 days. With a daily snapshot
discipline, this is recoverable. Without it, you'd lose ~12 entries
worth of historical context that can't be reconstructed from the
manuscript alone.

---

## How to use this file going forward

1. **Tonight:** Push to git at `phase_f/memory_snapshots/2026-05-23.md`
   and to Drive at
   `gdrive:kubernetes-cpu-ensemble-thesis/phase_f/memory_snapshots/`.
2. **Each subsequent Phase F day:** Generate a new
   `day_NNN.md` snapshot using the same format.
3. **Before defense:** Concatenate all snapshots into a single
   chronological memory history, useful for committee questions about
   when specific findings were established.
4. **If memory is ever lost:** Use the most recent snapshot to restore.

End of snapshot.
