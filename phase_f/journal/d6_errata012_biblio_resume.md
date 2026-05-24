# D6 — ERRATA-012 Overleaf application + biblio audit resume (2026-05-28)

casual notes. local Windows session (no Vast.ai today).

## scope

today's task: apply ERRATA-012 in Overleaf (Ch4 BCF Bitbrains row from
OLD pool to NEW pool per-VM medians). resume biblio audit per
biblio_audit_d3_notes.md schema, candidate ERRATA-013+ rows as PENDING
for next Overleaf pass (NOT applied today).

## decisions

no new decisions today. ERRATA-012 substitution was fully specified by
DECISION-011 + the canonical NEW pool per-VM medians; today is mechanical
application.

## what got done

- [x] Located Ch4 §4.X + Table 4.10 Bitbrains row in chapter_04 .tex
      source via grep on "ML wins only @120min" + "5.46pp" + "3.30pp".
- [x] Applied Table 4.10 Bitbrains row substitution: delta @30min
      -5.46pp → +1.53pp, delta @120min +3.30pp → -2.81pp, verdict
      "ML wins only @120min" → "Per-VM median negative at h120; 61% /
      35% of VMs win at h30 / h120".
- [x] Applied §4.X prose substitution: ~3 sentences disclosing per-VM
      medians at four horizons (-16.66 / +1.53 / -1.99 / -2.81 pp),
      per-VM win rates (61% h30, 35% h120), OLD→NEW aggregation scope
      shift with production-relevance rationale.
- [x] PDF rebuild clean. Visual diff confirmed: only Ch4 BCF section +
      Table 4.10 row changed; §3.8 BCF backbone (AUC 0.80, p=0.0097)
      and Table 4.13 / §6.1 HPA framing untouched.
- [x] Overleaf commit pushed: "Apply ERRATA-012 — Ch4 BCF Bitbrains row
      from OLD pool to NEW pool per-VM medians, Phase F D6 (2026-05-28)"
- [x] ERRATA.md row 012 status PENDING → APPLIED, Applied = 2026-05-28.

## biblio audit progress

[USER FILLS IN AFTER COMMIT — for each .bib entry inventoried today,
note the row schema entries + status. Candidate ERRATA-013+ rows
appended to ERRATA.md as PENDING.]

n entries inventoried today: [N]
- OK: [count]
- CORRECT_FIELD_X: [count] — keys: [list]
- FABRICATED_REPLACE_ENTRY: [count] — keys: [list]
- UNVERIFIABLE_DEFER: [count] — keys: [list, with reason]

candidate ERRATA-013+ rows opened today (PENDING):
- ERRATA-XXX: [.bib key, what's wrong, what gets corrected in chapter]
- [...]

## findings / surprises

[USER FILLS IN any unexpected mismatches found during Overleaf
application — e.g. pooled-scope Bitbrains text elsewhere in Ch4/Ch5
that needs reconciling with the new per-VM scope, or .bib entries
that were correct but cited inconsistently in chapter.]

## blockers / forward implications

- ERRATA-013+ rows from today's biblio findings batch for D7+ Overleaf
  application (not applied today per the D6-D8 sequence plan).
- D7: biblio audit continues + F2 chapter outline begins.

## off-plan items today (lessons)

[USER FILLS IN if anything went sideways during Overleaf application
or biblio verification — e.g. .tex anchor moved since D3 audit, grep
patterns needed adjustment, PDF rebuild had unexpected warning, etc.]

## time

started ~HH:MM, finished ~HH:MM, total ~X h