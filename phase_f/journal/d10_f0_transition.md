# D10 — F0 transition (2026-06-01)

casual notes. local Windows session, no Vast.ai today, no Overleaf
today. paperwork day: F0 work product transitions to F1 implementation,
with two paperwork items deferred to F1+ slot.

## scope

D10 task per the D9-D10 sequence, adjusted at session-start per Jimmy's
"manuscript will let for later" instruction:
- task A (Overleaf ERRATA-012 application): DEFERRED to F1+ slot.
- task B (biblio audit): DEFERRED to F1+ slot, with D10 decision
  section appended to biblio_audit_d3_notes.md.
- task C (F0 transition state file refresh + this journal + optional
  handoff): DONE today.

Honest framing: F0 lockdown is partially closing. F0 intellectual
work product (decisions, F2 test, F1 design, F2 outline, memory
snapshots) is complete. F0 manuscript-side propagation (ERRATA-012
application) is not. F1 implementation can start D11+ in parallel
with the deferred manuscript work.

## what got done

- [x] biblio audit deferred to F1+ via deferral section appended to
  biblio_audit_d3_notes.md. F1+ work order specified: 5 memory-flagged
  entries first, then method references, then stat methods, then
  peripheral. Bonus scoping hint added re: _citation_keys_used.txt
  and _unused_bib_keys.txt at repo root.
- [x] F0 transition state file refresh: THESIS_STATE.md rewritten
  for D10 close. Phase = F1 implementation start; F0 work product
  complete; F0 manuscript application carries to F1+. ERRATA-012
  row remains PENDING in ERRATA.md (no transition today since
  Overleaf untouched). D11 plan = F1 classifier architecture pick
  + feature matrix + LOO-cell CV implementation per DECISION-012.
- [x] this journal written.
- [x] optional D6-D10 combined handoff written at
  handoffs/2026-06-01_d10-close.md for audit consistency.

## what got deferred

- ERRATA-012 Overleaf application. Substitution text fully specified
  in ERRATA-012 Correction column + DECISION-011 anchor. F1+ paperwork
  slot will batch this with biblio audit findings.
- Bibliography audit per-entry verification per schema in
  biblio_audit_d3_notes.md.

## findings / surprises

- **Working tree contains substantial off-scope demo-branch
  work-in-progress** (app.py changes, demo_build/ directory,
  models/demo/precomputed/cell__bd_instance_93__*.json files, etc.).
  Per SYNC_PROTOCOL Revision 4 surgical-add discipline, today's D10
  commit stages only the 4 phase_f paths. Demo-branch state stays as-is.
- **_citation_keys_used.txt and _unused_bib_keys.txt exist at repo
  root** (untracked). These bound the F1+ biblio audit scope and
  may shrink it significantly. Surfaced in biblio_audit_d3_notes.md
  deferral section.
- **F0 close framing pivoted at session-start** from "F0 close batch"
  (all three tasks today) to "F0 transition with manuscript carry-over"
  (paperwork side today, Overleaf later). Updated state file framing
  accordingly. Honest split — F0 work product is genuinely done;
  manuscript propagation is genuinely deferred.

## F0 work product summary (at transition)

F0 lockdown (D1-D10) work product is complete:
- **12 DECISIONS** logged (DECISION-001 through DECISION-012).
- **12 ERRATA** opened. 11 APPLIED at D3 batch (Overleaf commit
  covering ERRATA-001 through 011). 1 still PENDING (ERRATA-012,
  carries to F1+ Overleaf slot).
- **Substantive new files (5+):** F2 chapter outline v4
  (f2_chapter_outline.md), F1 design lock (f1_prep_scope.md),
  F2 WPE data per series (3 CSVs), F2 partial-R² results CSV,
  F2 compute scripts (compute_wpe.py, f2_partial_r2.py),
  two memory snapshots (2026-05-23, 2026-05-31).
- **F2 pre-registered test:** NULL reported (headline 0.0790,
  Bitbrains per-VM 0.0558; both below 0.30 threshold; structural
  diagnosis R²_reduced = 0.903).
- **F1 router design locked** at DECISION-012, implementation
  ready at D11+.
- **Verifier audit:** 211/3 → 214/0 (D5 cleared via Q-007 fix).

**F0 manuscript-side carry-over to F1+:**
- ERRATA-012 Overleaf application (Ch4 §4.X prose + Table 4.10
  Bitbrains row; substitution text fully specified).
- Bibliography audit per-entry verification (schema set up, scope
  bounded by _citation_keys_used.txt).

**Audit status at F0 transition:** thesis manuscript Overleaf source
agrees with all canonical CSVs / JSONs / log files for the 11 ERRATA
applied at D3. The Bitbrains BCF row in Ch4 still uses OLD pool
per-VM semantics in the manuscript; canonical CSV
(boundary_condition_table_corrected.csv) is already at NEW pool, so
the divergence is one row pending Overleaf application.

## decisions today

No DECISION-013 today. The biblio audit and Overleaf deferrals are
process decisions (when to do the work), not methodology decisions
(what work to do); both schemas are preserved. Captured in this journal
+ biblio_audit_d3_notes.md instead of DECISIONS.md.

## next steps

- D11+ (2026-06-02 onward): F1 implementation per DECISION-012.
  Classifier architecture pick at start. LOO-cell CV runs.
  Macro-F1 reported. 5 remaining F2 outline TBVs cleared
  incidentally.
- F1+ paperwork slot (TBD, likely D15+): ERRATA-012 Overleaf
  application + biblio audit per schema, batched together for one
  Overleaf commit covering ERRATA-012 + any ERRATA-013+ rows from
  biblio findings.
- D16 (2026-06-07, Sun): next weekly memory snapshot.

## time

started ~10:00, finished ~12:00, total ~2 h (biblio deferral writeup
+ F0 transition writeup + state file refresh + handoff). Lighter
than the projected ~4h because Overleaf application was pivoted out.