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