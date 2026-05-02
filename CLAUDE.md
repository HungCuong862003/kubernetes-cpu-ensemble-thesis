# Live Demo — Build Context for Claude Code

## What we're building

A new Streamlit page `pages/0_Live_Demo.py` for the existing thesis dashboard. Single live-demo page that lets a viewer:

- See pre-computed forecasts for 4 curated Alibaba containers across 4 horizons (10/30/60/120min)
- Compare ML vs naive predictions with CQR uncertainty bands
- See actual outcomes overlaid (test set ground truth)
- View a BCF traffic light explaining when ML helps and when it doesn't
- See the naive + ML decomposition explicitly
- Toggle between compare-grid mode (2x2 of all 4 containers) and detail mode (single container)

This page is the FRONT DOOR of the dashboard. The existing 7 research pages become supporting evidence behind it.

## Critical constraints

1. **NO model code in the page.** All predictions are precomputed and stored as JSON. The page loads JSON and renders. No joblib loading, no `sprint1_Main.py` import, no live inference.
2. **Mobile responsive but desktop-first.** The defense projector is the primary target.
3. **Keep prose minimal.** Hero plot does the work; one-line explanations only.
4. **Code style:** junior-programmer voice. Print statements OK during development. No over-clever constructs. See the dedicated "Code Style" section below for details.

## Code Style (humanized, junior-programmer voice)

Write code as a real junior CS student would, not as an AI assistant producing polished output. Specifically:

- Comments are sparse and only where genuinely useful. Don't comment what the code obviously says ("# loop through items" above a for loop). DO comment WHY a non-obvious choice was made ("# h_steps=12 because 60min @ 5min cadence; sprint1's convention").
- No docstrings on every function. Add one only for functions whose contract isn't obvious from the signature. One-liner descriptions are fine.
- Variable names are practical, sometimes terse. `cells` not `precomputed_forecast_cells`. `df` is fine for a local pandas DataFrame. Don't over-explain.
- Section comments using `# === short label ===` banners are acceptable when a script has multiple distinct phases. Don't decorate every block.
- Print statements during development are OK. Leave a few for debugging visibility. Don't wrap everything in logging.
- No type hints unless the function is genuinely confusing without them. Streamlit page code rarely needs them.
- Imports grouped: stdlib, third-party, local. Blank line between groups. No alphabetical zealotry.
- f-strings for formatting. Don't mix .format() and f-strings.
- Error handling only where things actually fail. Don't wrap every read in try/except defensively.
- If you reach for a "clever" pattern (decorators, metaclasses, complex comprehensions), step back and use the obvious version instead. We're optimizing for a reviewer who'll read this for 30 seconds at defense.
- One blank line between functions, two between major sections. Standard PEP 8 spacing, not aggressively spaced.
- File should look like ~80% of it was written in one sitting by someone who was thinking about the problem, not refining for elegance.

What we want to avoid: emoji decorations in print statements, exhaustive type annotations, defensive try/except wrappers, three-line docstrings on five-line functions, alphabetized imports for their own sake, and the general "this was clearly produced by an LLM" feel.

What we want to keep: working code, comments where they earn their space, occasional inline TODO or FIXME if you genuinely punt on something.

## Data already in place — do not regenerate

All inputs are precomputed. Locations:

### Precomputed forecast cells (16 files, ~300 KB total)
`models/demo/precomputed/`
- `index.json` — metadata for all 16 cells (load this first)
- `demo_containers.json` — the 4 picked containers and their slot/label
- `cell__<container_id>__<horizon>.json` × 16 — per-cell prediction data

### The 4 picked containers
- **Slot A — "ML wins big"**: `c_41072` (ACF@24h=0.71, CV=0.43, +54pp at 120min, diurnal) — hero shot
- **Slot B — "Marginal"**: `c_12798` (ACF@24h=0.26, CV=0.34, +1.5pp at 120min) — small ML benefit
- **Slot C — "Bursty"**: `c_9956` (ACF@24h=0.02, CV=1.44, +6.6pp at 120min, irregular spikes) — wide CQR band
- **Slot D — "ML loses (BCF red)"**: `c_41212` (ACF@24h=0.11, CV=0.19, -36pp at 120min) — honesty signal

### Cell JSON structure
Each `cell__<cid>__<hz>.json` has:

```json
{
  "metadata": {
    "container_id": "c_41072",
    "slot": "A",
    "label": "ML wins big",
    "horizon": "60min",
    "h_steps": 12,
    "cadence_min": 5,
    "acf_24h": 0.7144,
    "cv": 0.4257,
    "hurst": 0.8701,
    "delta_pp": 16.64,
    "et_r2": 0.8483,
    "naive_r2": 0.6819,
    "bcf_zone": "green",   // "green" or "red"
    "bcf_reason": "ACF@24h (0.71) > 0.2 AND horizon (60min) >= 30min",
    "n_test_points": 341,
    "n_predictions": 329,
    "has_cqr": true
  },
  "history": {
    "timestamps": [int, int, ...],         // length n_test_points
    "cpu_actual": [float, float, ...]      // length n_test_points
  },
  "predictions": {
    "issue_timestamps":  [int, ...],       // length n_predictions; when forecast made
    "target_timestamps": [int, ...],       // length n_predictions; what time predicted
    "naive_pred":  [float, ...],
    "ml_pred":     [float, ...],
    "cqr_lower":   [float, ...],           // null if has_cqr=false
    "cqr_upper":   [float, ...],           // null if has_cqr=false
    "y_true":      [float, ...]            // actual outcome at target_timestamp
  }
}
```

### Other inputs needed
- `reports/tables/hpa_simulation.csv` — for the HPA reactive vs proactive panel
- `results/bcf/bcf_pairs.csv` — per-cell BCF predicate data (already encoded in cell metadata, but the global table is in case you need it)
- `reports/tables/shap_importance.csv` — for the optional drill-down "how was this forecast made" expander

## Critical CQR labeling

The CQR bands are EMPIRICALLY ~80% intervals (uncalibrated CQR offsets, not the 90% recalibrated version). Label them as "Prediction interval" not "90% prediction interval". Hover/tooltip should say "Empirical ~80% interval (uncalibrated CQR offset). Width grows with horizon and decreases with workload predictability."

## Page layout

### Top
- Title: "Kubernetes Workload Forecaster"
- Tagline: *"Predicts when ML helps, and tells you when it doesn't."*

### Mode toggle
- Radio button: "Compare all 4" (default) | "Detail (single container)"

### Compare mode
- 2x2 grid, one tile per container, all at the user-selected horizon (chosen via a horizon selector at top)
- Each tile shows:
  - Slot label and container ID at the top
  - A condensed plot: actual CPU history (last ~24h) + ML forecast trajectory + CQR band
  - Border color = BCF zone (green or red)
  - One-line summary: "ML helps (+54pp)" or "Naive wins (-36pp)" etc.

### Detail mode
- Container dropdown (4 options labeled by slot)
- Horizon radio (10 / 30 / 60 / 120 min)
- "Now" cursor slider (lets viewer scrub through the test window)
- **Hero plot** (full-width):
  - Solid blue: actual CPU history
  - Gray dotted: rolling naive forecast
  - Red dashed: rolling ML forecast
  - Red shaded band: CQR prediction interval (use plotly fill_between; handle inverted bounds gracefully — sometimes lower > upper at sharp transitions, that's data not bug)
  - Vertical line at "now" cursor
  - At cursor + horizon: red dot (ML), gray dot (naive), green star (actual)
  - Horizontal threshold line (slider-adjustable, default = max(history) * 1.2)
  - Red shading where ML upper band crosses threshold
- **Three info cards below hero plot** (in row):
  1. **BCF traffic light** — green or red based on `metadata.bcf_zone`. Show ACF@24h, horizon, predicate that fires
  2. **Naive + ML decomposition** — at the cursor: "Persistence forecast: X.XX% / ML correction: ±Y.YY% / Final: Z.ZZ%"
  3. **Recommended action** — if BCF green AND ml_pred > threshold: "Scale up by N at T+horizon"; if BCF red: "Use naive baseline (ML offers no improvement on this workload)"; else "No action needed"
- **HPA panel** — bar chart from `reports/tables/hpa_simulation.csv`. For the selected horizon, show "Reactive HPA" vs "ML-Proactive" violation rates. Use canonical config (target_util=0.7, safety_margin=1.2 — verify in CSV)
- **Drill-down expander** (collapsible, closed by default):
  - "How was this forecast made?" — shows ET R² vs naive R² for this cell, top SHAP features for this horizon (read shap_importance.csv)
  - Link/button to "Open the full research dashboard" → existing pages 1-7

## Build pieces, in order

Don't try to build everything at once. Each piece must render correctly before moving to the next.

### Piece 1 — Skeleton + data loading + basic chart (1-2 hours)
- Create `pages/0_Live_Demo.py`
- Sidebar: container dropdown, horizon radio
- Load `index.json` and the selected `cell__*__*.json`
- Render just one thing: a Streamlit `st.line_chart` of `cpu_actual` for the picked container at the picked horizon
- Verify all 4 containers x 4 horizons load without errors
- Verify it appears as page 0 (first) in the sidebar

### Piece 2 — Hero plot with predictions + CQR + cursor (2-3 hours)
- Replace `st.line_chart` with a Plotly figure
- Add ML/naive forecast traces, CQR band (use go.Scatter with fill="tonexty"), threshold line, violation shading
- Cursor slider lets user scrub through the prediction range
- Three forecast markers at cursor+horizon: red dot (ML), gray dot (naive), green star (y_true)

### Piece 3 — Three info cards (BCF / decomposition / action) (1-2 hours)
- Three st.columns below the hero plot
- BCF light: colored container with ACF/horizon/reason
- Decomposition: pull naive_pred and ml_pred at cursor, format as text card
- Action: simple if/else based on bcf_zone and threshold crossing

### Piece 4 — Compare mode + HPA panel (2-3 hours)
- Mode toggle radio at top
- Compare: 2x2 grid using st.columns; each tile is a smaller Plotly figure
- HPA panel: bar chart from hpa_simulation.csv (verify column names, the file is at reports/tables/hpa_simulation.csv)

### Piece 5 — Polish + drill-down + final review (2-3 hours)
- Drill-down expander with R² comparison + SHAP top features
- Title, tagline, footer with caveats
- Mobile responsive sanity check
- Walk through all 4 containers x 4 horizons in both modes

## Things to verify yourself, not assume

1. The dashboard repo's existing `app.py` and `pages/` directory structure — adapt naming if needed
2. CQR band rendering — at sharp transitions, lower > point > upper can happen; use plotly fill_between which handles this gracefully (don't use errorbar)
3. `hpa_simulation.csv` column names — read the first row to confirm before referencing
4. The existing dashboard's color/style conventions — match them so this page doesn't look bolted on

## Commit discipline

Each piece ends with a git commit on a feature branch. Don't push to main until all 5 pieces are done and reviewed.

## What NOT to do

- Don't try to load joblibs at runtime
- Don't import sprint1_Main.py or anything that triggers feature engineering
- Don't claim "90% prediction interval" anywhere — it's empirically ~80%
- Don't add a CSV upload feature in v1 (out of scope; cluster_id imputation problem)
- Don't refactor the existing 7 dashboard pages
- Don't restructure the repo

## When uncertain, ask

If something is ambiguous (column name, file location, design choice), pause and ask. Don't guess.