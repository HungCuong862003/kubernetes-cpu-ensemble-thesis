# Bibliography Audit — Phase F Day 3 (2026-05-25)

**Scope:** Cross-reference all `\citep` / `\citet` keys in Ch1–Ch6 against the `.bib` file, then verify each .bib entry against authoritative sources (publisher DOI page, arXiv abstract, conference proceedings, official GitHub).

**Priority order:** (1) Three flagged-in-memory entries (Fremer, AAPA, OptScaler), (2) Phase F method references (foundation models + ensembles), (3) Statistical methods cites, (4) Older/peripheral entries.

**Status values:** `OK`, `CORRECT_FIELD_X`, `FABRICATED_REPLACE_ENTRY`, `UNVERIFIABLE_DEFER`.

---

## Per-entry findings

| .bib key | Claimed lead author | Verified lead author | Claimed venue | Verified venue | Claimed year | Verified year | Status | Notes |
|---|---|---|---|---|---|---|---|---|
| fremer-pvldb | [fill from .bib] | Hengyu Ye | [fill from .bib] | PVLDB | [fill from .bib] | [verify] | [TBD] | Memory flag: lead author must be Hengyu Ye, not whatever earlier draft claimed |
| fremer-arxiv | [fill from .bib] | Jiadong Chen | [fill from .bib] | arXiv | [fill from .bib] | [verify] | [TBD] | Memory flag: separate arXiv preprint with different lead author Jiadong Chen |
| aapa | [fill from .bib] | Guilin Zhang | [fill from .bib] | [verify] | [fill from .bib] | [verify] | [TBD] | Memory flag: authors are Guilin Zhang et al., affiliations GWU + Workday + Youngstown State |
| optscaler | [fill from .bib] | [verify] | [fill from .bib] | [verify] | [fill from .bib] | [verify] | [TBD] | Memory flag: no public code exists. If .bib cites a GitHub URL or code release, the citation is fabricated and must be flagged. |

*[Append rows below for every additional .bib entry inventoried today. Use the same schema.]*

---

## Cluster summary (fill at end of audit)

- Total entries inventoried: [N]
- `OK`: [count]
- `CORRECT_FIELD_X`: [count] — list keys
- `FABRICATED_REPLACE_ENTRY`: [count] — list keys
- `UNVERIFIABLE_DEFER`: [count] — list keys with reason

## Carry to Day 4

- Apply `.bib` corrections for the `FABRICATED_REPLACE_ENTRY` cluster.
- For any entry whose correction changes meaning in-chapter (not just a name spelling), open ERRATA-012+ in `ERRATA.md`.
---

## D10 deferral decision (2026-06-01)

**Status at D9 state read:** schema + 3 placeholder rows. No per-entry
verification work has happened across D3-D9. The D3 close handoff
implied "audit in progress" but the file shows otherwise.

**Decision (D10):** defer biblio audit work to F1+ phase, with explicit
F1+ slot allocation. F0 close does NOT mark biblio audit complete.

**Rationale:**
- Forcing the audit into D10 risks half-finished output (worst case:
  half-audited bibliography marketed as audited at defence).
- F0 close batch is higher-priority and time-bound to D10.
- F1+ phase has natural paperwork days (e.g. D15+ between F1
  implementation milestones) where biblio audit batches well.
- The four memory-flagged anchors (Fremer-pvldb / Fremer-arxiv /
  AAPA / OptScaler / HCMIU QĐ719) are the highest-risk entries. These
  five can be verified at the start of the F1+ slot in ~1 hour
  combined.

**Bonus scoping hint:** repo root contains `_citation_keys_used.txt`
and `_unused_bib_keys.txt` (untracked) from a prior session. These
likely bound the audit scope — entries in `_unused_bib_keys.txt` can
be deleted outright (no audit needed), and `_citation_keys_used.txt`
defines the actual verification surface. Check these at F1+ audit
start before committing to a per-entry pass.

**Carry to F1+:**
- Audit all .bib entries against the schema in this file, scoped by
  `_citation_keys_used.txt` if usable.
- Priority order: (1) the five memory-flagged entries first, (2)
  Phase F method references (foundation models + ensembles), (3)
  statistical methods cites, (4) older/peripheral entries.
- Any FABRICATED_REPLACE_ENTRY findings open ERRATA-013+ rows
  for Overleaf batch application in the same F1+ paperwork day
  that handles ERRATA-012 application.
- Any UNVERIFIABLE_DEFER entries stay in the schema with reason
  noted; defence Q&A can address them.

**Status at F0 transition (D10):** biblio audit NOT complete.
Schema-only. Deferred to F1+ together with ERRATA-012 Overleaf
application.