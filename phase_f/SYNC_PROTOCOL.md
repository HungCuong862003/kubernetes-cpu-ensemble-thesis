# SYNC_PROTOCOL.md

**Purpose:** Canonical end-of-day sync workflow across Vast.ai, Google Drive,
and the local Windows git checkout. Followed at every day-close and
every Vast.ai session-start to prevent content drift, credential leaks,
and sequencing bugs. Verification at the end of every transit is mandatory.

**Update protocol:** edits to this file should be rare. If the workflow
changes, append a "## Revision N (YYYY-MM-DD)" section rather than
rewriting inline. Same append-only discipline as DECISIONS.md and ERRATA.md.

---

## Three environments

| Env | Path | Role |
|---|---|---|
| Vast.ai | `/workspace/kubernetes-cpu-ensemble-thesis/` | Compute + scratch; source of new scripts, logs, day-of state edits when work happens on the instance |
| Drive | `gdrive:kubernetes-cpu-ensemble-thesis/` | Bridge between Vast.ai and local; canonical for DATA + `phase_f/` artefacts |
| Local Windows | `E:\thesis\kubernetes-cpu-ensemble-thesis\` | Git checkout; source of git commits |

**Path asymmetry to remember:** state files (`THESIS_STATE.md`, `DECISIONS.md`,
`ERRATA.md`) and `handoffs/` live at **WORKSPACE ROOT** on Vast.ai but under
`phase_f/` on Drive and in `phase_f/` in the git repo. Everything else
(`logs/`, `scripts/`, `journal/`, `SYNC_PROTOCOL.md` itself) lives under
`phase_f/` in all three environments. rclone handles the path mapping with
explicit per-file destinations; don't try to "fix" the asymmetry by changing
Vast.ai layout, it would break the prior six months of relative paths in
existing scripts.

## Branch + path constants

- Local repo path: `E:\thesis\kubernetes-cpu-ensemble-thesis` (the
  `kubernetes-cpu-ensemble-thesis` subdirectory IS the repo; `E:\thesis`
  is just a container directory).
- Branch: `feature/live-demo` (NOT `main`; D1 work landed here and
  Phase F continues on this branch).
- Remote: `origin = https://github.com/HungCuong862003/kubernetes-cpu-ensemble-thesis`
  (PUBLIC, MIT licence pending Q-003 supervisor approval).

## Sync direction policy

Three valid directions, each with a triggering condition:

| Direction | When | Initiated from |
|---|---|---|
| Vast.ai → Drive → Local | Day's work happened on Vast.ai (scripts ran, logs produced) | Vast.ai SSH at EOD |
| Local → Drive | Day's work happened in chat / Overleaf / locally (no Vast.ai compute) | Local PowerShell |
| Drive → Vast.ai | Vast.ai session-start to pick up state changes from prior day | Vast.ai SSH at session-start |

Drive is canonical in all three cases. Never edit on two sides simultaneously
without resolving via Drive first.

## Sync order (strict)

The bug that produced D2's `0a5186b` + `05edef6` fixup pair was inverting
step 1 and step 2 — syncing Drive before pasting Vast.ai content was complete.

### Standard EOD flow (Vast.ai work on the day)

1. **Vast.ai**: finalise all .md edits, verify with `head -3 <file>`.
2. **Vast.ai → Drive**: `rclone copy` each path with explicit destinations.
3. **Verify Drive received**: `rclone check` (mandatory; see §Verification).
4. **Drive → Local**: `rclone copy gdrive:.../phase_f/ .\phase_f\`.
5. **Verify Local matches Drive**: `rclone check` (mandatory).
6. **Local**: `git add phase_f/`, verify with `git diff --cached --name-only`,
   commit via Out-File-based message, push.
7. **Verify GitHub received**: spot-check commit log URL.
8. **Vast.ai web UI**: Stop (NOT Destroy).

### Local-first EOD flow (no Vast.ai work on the day)

1. **Local**: save all .md files at correct local paths under `phase_f/`,
   verify with `Get-Content .\phase_f\<file> -TotalCount 3`.
2. **Local → Drive**: `rclone copy .\phase_f\ gdrive:.../phase_f/`.
3. **Verify Drive received**: `rclone check` (mandatory).
4. **Local**: `git add phase_f/`, verify, commit, push.
5. **Verify GitHub received**: spot-check commit log URL.
6. **Vast.ai**: stays stopped. Next session-start pulls from Drive.

### Vast.ai session-start flow

1. **Start Vast.ai instance** via web UI (Stop preserved disk; ~3 min boot).
2. **SSH in**, navigate to workspace root.
3. **Pull state files** from Drive `phase_f/` to Vast.ai workspace ROOT
   (per-file due to path asymmetry).
4. **Pull `handoffs/`** from Drive `phase_f/handoffs/` to Vast.ai workspace
   `./handoffs/`.
5. **Pull `phase_f/`** subdirectories (`journal/`, `logs/`, `scripts/`,
   `SYNC_PROTOCOL.md`) from Drive `phase_f/` to Vast.ai `./phase_f/`.
6. **Verify Vast.ai matches Drive**: `rclone check` per path (mandatory).
7. Proceed with day's work.

---

## Verification — mandatory at every transit

Verification is non-optional. Every sync direction must end with an
`rclone check` confirming the two sides agree on every file under scope.
A successful `rclone check` reports `0 differences found` (or equivalent
zero-counts). Any non-zero output is a failure that must be resolved
before moving to the next step.

`rclone check` compares MD5 hashes when both sides support them; Google
Drive does. It is a true content comparison, not a size-only check.

### Verify Local ↔ Drive (after Local → Drive push or Drive → Local pull)

```powershell
cd E:\thesis\kubernetes-cpu-ensemble-thesis

rclone check .\phase_f\ gdrive:kubernetes-cpu-ensemble-thesis/phase_f/ --one-way
# --one-way: report files present on local but absent on Drive (and content
# diffs); ignore files present on Drive but not local (those would be from
# environments we don't manage locally, e.g. data_snapshots that live on
# Drive only)

# Expected:
#   2026/MM/DD HH:MM:SS NOTICE: <directory>: 0 differences found
#   2026/MM/DD HH:MM:SS NOTICE: <directory>: <N> matching files

# If output reports differences:
#   - Examine which files differ (the lines that start with NOTICE :
#     "<path>: <reason>")
#   - Re-run rclone copy on the divergent paths
#   - Re-run rclone check until clean
```

For full bidirectional check (no `--one-way`), use:

```powershell
rclone check .\phase_f\ gdrive:kubernetes-cpu-ensemble-thesis/phase_f/
# Reports files-only-on-A, files-only-on-B, and content-differs
```

The bidirectional form is the right answer for the canonical day-close
verification. Use `--one-way` only when you know Drive has additional
content (e.g. data_snapshots/) that local doesn't track.

### Verify Drive ↔ Vast.ai (after Vast.ai → Drive push or Drive → Vast.ai pull)

Path asymmetry means the check must be done per-path, not as a single
directory check. On Vast.ai SSH:

```bash
cd /workspace/kubernetes-cpu-ensemble-thesis

# State files: Vast.ai root vs Drive phase_f/
for f in THESIS_STATE.md DECISIONS.md ERRATA.md; do
  echo "=== $f ==="
  rclone check ./$f gdrive:kubernetes-cpu-ensemble-thesis/phase_f/$f 2>&1 | tail -2
done

# handoffs/: Vast.ai root/handoffs/ vs Drive phase_f/handoffs/
echo "=== handoffs/ ==="
rclone check ./handoffs/ gdrive:kubernetes-cpu-ensemble-thesis/phase_f/handoffs/ 2>&1 | tail -3

# phase_f/ subdirectories: same path both sides, but exclude root state files
# and handoffs/ which were checked above
echo "=== phase_f/ (subdirectories only) ==="
rclone check ./phase_f/ gdrive:kubernetes-cpu-ensemble-thesis/phase_f/ \
  --filter "- THESIS_STATE.md" \
  --filter "- DECISIONS.md" \
  --filter "- ERRATA.md" \
  --filter "- handoffs/**" \
  2>&1 | tail -3
```

Expected: every block ends with `0 differences found`. Any other output
identifies a path that needs re-pushing or re-pulling.

### Verify git ↔ GitHub (after every push)

```powershell
cd E:\thesis\kubernetes-cpu-ensemble-thesis

# Local HEAD SHA
$localHead = git rev-parse HEAD
Write-Host "Local HEAD: $localHead"

# Remote HEAD SHA (fetches metadata without modifying working tree)
git fetch origin feature/live-demo
$remoteHead = git rev-parse origin/feature/live-demo
Write-Host "Remote HEAD: $remoteHead"

# Must match
if ($localHead -ne $remoteHead) {
    Write-Warning "Local and remote HEAD disagree. Push did not land or remote moved."
}
else {
    Write-Host "OK: local and remote agree at $localHead"
}

# Working tree must be clean (no uncommitted changes)
$status = git status --porcelain
if ($status) {
    Write-Warning "Working tree has uncommitted changes:"
    Write-Output $status
}
else {
    Write-Host "OK: working tree clean"
}
```

Expected output:
## Revision 4 (2026-05-27)

Surgical patches to canonical project files outside `phase_f/` (e.g.,
`reports/tables/`, `results/`, `src/`) can be committed alongside
`phase_f/` when they are the direct output of Phase F work documented
in `DECISIONS.md`. Stage these per-file: `git add path/to/file`. Never
use `git add -A`.

Example: DECISION-011 (Q-007 Anchor A) modified
`reports/tables/thesis_numbers.json`. The D5 commit staged this file
explicitly via `git add reports/tables/thesis_numbers.json` alongside
`git add phase_f/`, and the commit message named the off-scope file so
it traces back to the originating decision.

## Revision 5 (2026-05-27, proposed by D5 retrospective — not yet enforced)

Session-start checklist must include an explicit pull-state-files block
before any other work begins, not just a verify. Today (D5) the verify
step alone surfaced the broken-rclone-hashsum-on-single-files issue, but
the underlying problem — state files absent from Vast.ai workspace root
— was missed until EOD. Recommended automation: a
`phase_f/scripts/phase_f_session_start.sh` wrapper that does steps 3–6
of the existing Vast.ai session-start flow non-interactively.
## Revision 6 (2026-05-27)

**Codifies the Local-first state-file workflow** used for the D4 handoff
post-hoc add and going forward. This is a refinement of "Local-first
EOD flow", not a replacement — it specifies the human-loop sequence
when state-file content is being generated in chat rather than typed
directly into a local editor.

### When to use this workflow

For any human-authored `.md` file under `phase_f/` whose content is
being drafted by Claude in chat:
- Handoffs (`handoffs/*.md`)
- Journals (`journal/*.md`)
- THESIS_STATE.md refresh
- DECISIONS.md amendments
- ERRATA.md additions
- SYNC_PROTOCOL.md revisions (like this one)

For compute artefacts produced on Vast.ai (scripts, data CSVs, run
logs), the existing "Standard EOD flow" still applies — those stay
Vast.ai-first.

### Sequence

1. **Claude provides the complete file content** as a single fenced
   markdown block in chat. For wholesale replacements, the full file;
   for append-only files (DECISIONS, ERRATA, SYNC_PROTOCOL), only the
   new section.

2. **User: create or edit on Local Windows.** Open the file in notepad
   or VS Code, paste content, save at the correct path under
   `E:\thesis\kubernetes-cpu-ensemble-thesis\phase_f\<file>`.

3. **User: upload same content to Vast.ai.** Path destination:
   - State files (`THESIS_STATE.md`, `DECISIONS.md`, `ERRATA.md`) and
     `SYNC_PROTOCOL.md`: maintain the workspace-root asymmetry per the
     §"Three environments" rule — these live at workspace root on
     Vast.ai.
   - `handoffs/*.md` and `journal/*.md`: may live at `phase_f/handoffs/`
     or `phase_f/journal/` on Vast.ai (aligned with Local + Drive). The
     workspace-root asymmetry for `./handoffs/` is preserved as
     legacy convention but is not load-bearing — no existing script
     reads files in `./handoffs/`, so the deviation is tolerated.
     If a workspace-root copy is wanted for consistency, the next
     Vast.ai session-start can pull from Drive `phase_f/handoffs/`
     to `./handoffs/`.

4. **Local → Drive**: from Local PowerShell:
```powershell
   rclone copy .\phase_f\<file> gdrive:kubernetes-cpu-ensemble-thesis/phase_f/<destination>/
```
   (Use the bulk directory form `rclone copy .\phase_f\ gdrive:.../phase_f/`
   for multi-file pushes; single-file destinations must include the
   parent path on Drive.)

5. **Md5 verification across all populated layers.** Single-file
   `rclone check` fails with "is a file not a directory"; use
   hashsum diff instead:

```powershell
   # Local
   $localMd5 = (Get-FileHash .\phase_f\<file> -Algorithm MD5).Hash.ToLower()
   # Drive
   $driveMd5 = (rclone hashsum md5 gdrive:.../phase_f/<file>).Split(' ')[0]
```
```bash
   # Vast.ai SSH
   md5sum phase_f/<file>   # or ./<file> if state file at workspace root
```

   All populated layers must print the same 32-char hex. Mismatch
   means re-push the divergent layer from the canonical source
   (Local in this workflow).

6. **Git commit + push from Local.** Always from Local PowerShell
   (git lives nowhere else):

```powershell
   git add phase_f/<file>
   git diff --cached --name-only

   @"
   <commit subject and body — ASCII hyphens OR Encoding UTF8 below>
   "@ | Out-File -Encoding UTF8 -NoNewline -FilePath .\_commit_msg.txt

   git commit -F .\_commit_msg.txt
   Remove-Item .\_commit_msg.txt

   git push origin feature/live-demo

   # Verify push landed
   $localHead = git rev-parse HEAD
   git fetch origin feature/live-demo
   $remoteHead = git rev-parse origin/feature/live-demo
   if ($localHead -ne $remoteHead) { Write-Warning "Divergence" } else { Write-Host "OK pushed: $localHead" }
```

### Commit message encoding (corrects D4-era bug)

`Out-File -Encoding ASCII` silently strips non-ASCII characters from
commit messages — the em-dash `—` in commit `f2bf193`'s subject became
`?` in `git log`. Always use `-Encoding UTF8 -NoNewline` for commit
message files. Past commits not amended (cosmetic only); going
forward this is the rule.

### Single-file rclone gotchas (re-anchored)

Two failure modes observed at D4 + D5 close, important enough to
restate here even though Revision 5 mentioned them:

1. `rclone copy gdrive:path/file.md ./` (single-file source, current
   directory destination) silently no-ops on some rclone versions —
   returns successfully without producing the file. Workaround:
   use directory copy `rclone copy gdrive:path/ ./` with filters, or
   `rclone copyto gdrive:path/file.md ./file.md` for explicit file
   copy.

2. `rclone check` fails with "is a file not a directory" when given
   single-file source or destination. Workaround: hashsum diff via
   `rclone hashsum md5` on both sides as in step 5 above.

### Vast.ai stop discipline (unchanged but worth restating)

When Vast.ai work concludes for the day, stop the instance via the
web UI's **Stop** button (not Destroy). Stop preserves disk and
installed packages; Destroy wipes everything. Confirm the instance
status shows "stopped" before closing the browser. D4 close added
scipy 1.17.1 to `/venv/main/` which would be lost on Destroy.