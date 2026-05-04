# Kubernetes Workload Forecaster - Live Demo (v5 — dual-fragment architecture)
# Loads precomputed forecast cells (models/demo/precomputed/*.json) and renders
# them. No model loading, no inference. See CLAUDE.md for build plan.
#
# v5 vs v4 — fixes the cursor-not-moving bug diagnosed via the v4-debug panel.
# v4 cursor advance: every fragment tick called st.rerun(scope="app"). At ~1Hz
# this raced the main-pane render — sidebar (lightweight) updated each tick,
# main pane (Plotly hero + 4-panel HPA + drill-down) couldn't finish a render
# in <1s on a typical machine, so it got cancelled and the previous chart
# persisted. Counters showed cursor_adv=802 in 974 ticks (mechanically firing)
# but visually the cursor stayed still. Architecture cost, not a bug.
#
# v5 splits the work into two fragments:
#   - demo_tick_and_caption: caption + view-advance only. Calls
#     st.rerun(scope="app") rarely (once per ~14s, at view transitions).
#   - cursor_block_fragment: owns the cursor-dependent rendering block
#     (verdict pill, hero plot, cursor readout, info cards). Runs every 1s,
#     advances cursor in session state, re-renders inline. NEVER calls
#     st.rerun. The fragment's own re-render is sufficient.
#
# Cursor key (cursor_<cid>_<hz>) is shared between the slider and the fragment.
# When tour is active+unpaused, the slider is hidden (so no widget binding)
# and the fragment writes the key freely. When tour is off or paused, the
# slider is visible and owns the key normally. The slider position picks up
# from wherever the fragment last wrote on resume; manual scrub during pause
# propagates to the next fragment tick on resume.
#
# Layout change: verdict pill moved from above headline metrics to between
# sliders and hero plot, so the cursor-dependent block is contiguous (the
# fragment must own a contiguous region).
#
# Stripped: pending_cursor_change drain, _slider_version, versioned slider
# key — all dead with the new architecture. apply_demo_view now writes the
# cursor key directly when seeding start_pct (it's called from inside the
# caption fragment but the slider hasn't rendered on a fragment-only tick,
# so direct write is safe).
#
# Kept: pending_view_change drain for the radio (still widget-bound),
# diagnostic panel (toggle "Show debug" in sidebar) for verifying v5.
# Counter labels updated: caption_ticks, cursor_ticks (separate fragment),
# cursor_adv (only counts ticks where tour was active+unpaused), view_adv.
#
# v4 vs v3 — fixes from "examine carefully" pass before tackling the cursor:
# 1. HPA panel now loads hpa_simulation_v2.csv (800 rows, with target_util)
#    instead of v1 (200 rows) which had the known reactive-invariance bug
#    (reactive bars don't move when safety_margin changes). v2 schema adds a
#    target_util dimension; we surface it as a select_slider next to the
#    safety_margin slider. variant strings in v2 encode tu as a suffix
#    ("lag=1_tu=0.5", "ML_tu=0.5") — strip via split("_tu=")[0].
# 2. HPA caption now uses conditional phrasing. v1 never broke (avg_dv/dw
#    always positive there); v2 has 37 cells where avg_dw flips negative
#    (ML uses LESS capacity than reactive avg). Hardcoded "lowers ... uses
#    more" would render "lowers by -X" / "uses -X more". Now picks
#    "lowers"/"raises" and "more"/"less" based on sign.
# 3. Decomp card alignment. v3 used `:6.2f` padding inside <b>...</b> which
#    HTML collapses, plus hand-counted &nbsp; runs that didn't actually align.
#    Now uses `white-space: pre` on .info-decomp + `:<16` left-pad on labels +
#    `:>+7.2f` on the refinement (auto-sign, no leading-space glitch like
#    "+ 3.45"). Other values use `:>7.2f` for consistent column width.
# 4. Removed debug `st.code(_last_drain)` block and its write site in the
#    pending_cursor_change drain. Was visible above the page title.
# 5. HPA panel: removed the dead `for trace in fig.data: ... textposition =
#    "auto"` loop in build_diverging_hpa — bars were already created with
#    textposition="auto".
# 6. Slot tile "Selected" button now `disabled=is_sel` so clicking the
#    already-selected tile is a no-op (was unnecessarily rerunning + pausing
#    the tour).
# 7. Stripped `transition: transform/box-shadow` from .tile CSS — there were
#    no :hover rules to drive it, so it was dead.
# 8. Dropped `show_cqr` from session_state defaults (no UI toggled it; was
#    always True). Hardcoded True at the build_hero call site.
#
# v3 vs v2 (preserved for context) — tour-driven cursor advance via the
# existing demo_tick_and_caption fragment plus pending_cursor_change drain;
# Compare-4 desync fix via key="view"+pending_view_change drain; per-view
# start_pct in DEMO_VIEWS.

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import streamlit as st

# keyboard shortcuts are nice-to-have, demo works without them
try:
    import streamlit_shortcuts
    HAS_SHORTCUTS = True
except ImportError:
    HAS_SHORTCUTS = False


PRECOMPUTED_DIR = Path("models/demo/precomputed")
HPA_PATH = Path("reports/tables/hpa_simulation_v2.csv")
SHAP_PATH = Path("reports/tables/shap_importance.csv")
COVERAGE_PATH = Path("reports/tables/uq_recalibration_table.csv")

# how many cursor indices to advance per tour tick. 1s tick rate, so this sets
# the speed: 6 indices/sec means a ~316-index window sweeps in ~52 seconds.
PLAY_STEPS_PER_TICK = 6

# === colors (saturated for projector contrast) ===

ZONE_COLOR = {"green": "#15803d", "yellow": "#a16207", "red": "#b91c1c"}
ZONE_BG    = {"green": "#f0fdf4", "yellow": "#fefce8", "red": "#fef2f2"}

ML_RED        = "#b91c1c"
NAIVE_GRAY    = "#475569"
ACTUAL_BLUE   = "#1e40af"
ACTUAL_GREEN  = "#15803d"
TEXT_DARK     = "#0f172a"
TEXT_MUTED    = "#475569"
GRID_COLOR    = "#e2e8f0"
AXIS_COLOR    = "#475569"
BORDER_COLOR  = "#cbd5e1"

ZONE_VERDICT = {
    "green":  ("Use ML-Proactive",
               "ACF@24h crosses 0.2 and horizon >= 30min - the predicate holds. "
               "ML reduces SLO violations measurably."),
    "yellow": ("Use ML cautiously",
               "Mixed signal. ML helps on this horizon but margin is narrow - "
               "validate against a rolling window before deploying."),
    "red":    ("Stick with reactive",
               "BCF predicate fails. ML overfits noise and degrades vs naive - "
               "reactive HPA is the safer choice for this workload."),
}

# Demo Mode auto-tour. Each Single-container view has start_pct in [0,1)
# specifying where the cursor parks at view entry. Compare-4 views have
# start_pct = None.
DEMO_VIEWS = [
    {"cid": "c_41072", "hz": "120min", "view": "Single container", "duration": 14,
     "start_pct": 0.10,
     "caption": "1 / 5 — Slot A, 120min. Strong diurnal cycle (ACF@24h=0.71). "
                "Watch the cursor walk forward — verdict pill stays GREEN through the morning ramp."},
    {"cid": "c_41212", "hz": "120min", "view": "Single container", "duration": 14,
     "start_pct": 0.15,
     "caption": "2 / 5 — Slot D, 120min. Bursty, no temporal structure (ACF@24h=0.11). "
                "Verdict pill flickers RED on every spike — ML's noise hurts vs persistence."},
    {"cid": "c_41072", "hz": "120min", "view": "Compare 4", "duration": 10,
     "start_pct": None,
     "caption": "3 / 5 — All four archetypes side by side at 120min, sorted by ML lift. "
                "The trade-off in one frame."},
    {"cid": "c_41072", "hz": "30min", "view": "Single container", "duration": 14,
     "start_pct": 0.15,
     "caption": "4 / 5 — Same Slot A at 30min. R^2 gap shrinks vs 120min — "
                "ML's value is horizon-dependent."},
    {"cid": "c_41072", "hz": "10min", "view": "Single container", "duration": 14,
     "start_pct": 0.15,
     "caption": "5 / 5 — Slot A at 10min. BCF predicate fails (horizon < 30min) — "
                "even the strongest container goes RED. The rule, illustrated."},
]


# === data loading ===

@st.cache_data
def load_index():
    with open(PRECOMPUTED_DIR / "index.json") as f:
        return json.load(f)


@st.cache_data
def load_demo_containers():
    with open(PRECOMPUTED_DIR / "demo_containers.json") as f:
        return json.load(f)


@st.cache_data
def load_cell(container_id, horizon):
    fname = f"cell__{container_id}__{horizon}.json"
    with open(PRECOMPUTED_DIR / fname) as f:
        return json.load(f)


@st.cache_data
def load_hpa():
    # v2: 800 rows, schema: Horizon, Strategy, variant, target_util,
    # safety_margin, violation_rate, waste_rate, viol_severity. The variant
    # column encodes target_util as a suffix ("lag=1_tu=0.5", "ML_tu=0.5").
    return pd.read_csv(HPA_PATH)


@st.cache_data
def load_shap():
    df = pd.read_csv(SHAP_PATH)
    return df.rename(columns={df.columns[0]: "feature"})


@st.cache_data
def load_coverage():
    df = pd.read_csv(COVERAGE_PATH)
    return {row["Horizon"]: dict(row) for _, row in df.iterrows()}


def cell_zone(index, container_id, horizon):
    for entry in index["cells"]:
        if entry["container_id"] == container_id and entry["horizon"] == horizon:
            return entry["bcf_zone"]
    return "red"


def feature_group(name):
    if name in ("hour_sin", "hour_cos"): return "hour-of-day"
    if name in ("dow_sin", "dow_cos", "business_hours"): return "day/business"
    if "roll" in name: return "rolling"
    if name.startswith("cpu_lag") or name.startswith("mem_lag"): return "lag"
    if name.startswith("mem_") or name.startswith("disk_"): return "cross-resource"
    if name == "cluster_id": return "cluster"
    return "other"


def strip_tu_suffix(variant):
    """v2 encodes target_util as a "_tu=X" suffix on the variant string.
    "lag=1_tu=0.5" -> "lag=1", "ML_tu=0.5" -> "ML". For strings without the
    suffix this is a no-op (returns the original)."""
    return variant.split("_tu=")[0]


def cursor_verdict(cell, cursor_idx):
    m = cell["metadata"]
    p = cell["predictions"]
    zone = m["bcf_zone"].lower()

    ml_at = p["ml_pred"][cursor_idx]
    naive_at = p["naive_pred"][cursor_idx]
    actual = p["y_true"][cursor_idx]
    ml_err = abs(ml_at - actual)
    naive_err = abs(naive_at - actual)

    if ml_err < naive_err:
        moment = "ML beats persistence"
        delta_at_cursor = naive_err - ml_err
        moment_color = ZONE_COLOR["green"]
    elif ml_err > naive_err:
        moment = "ML loses to persistence"
        delta_at_cursor = ml_err - naive_err
        moment_color = ZONE_COLOR["red"]
    else:
        moment = "ML ties persistence"
        delta_at_cursor = 0
        moment_color = TEXT_MUTED

    overall_pp = m["delta_pp"]
    return {
        "zone": zone,
        "zone_color": ZONE_COLOR.get(zone, "#888"),
        "moment_text": moment,
        "moment_delta": delta_at_cursor,
        "moment_color": moment_color,
        "overall_pp": overall_pp,
        "overall_text": f"overall +{overall_pp:.1f}pp R^2" if overall_pp >= 0
                        else f"overall {overall_pp:.1f}pp R^2",
        "acf": m["acf_24h"],
        "horizon": m["horizon"],
    }


# === Plotly template ===

def setup_plotly():
    pio.templates["thesis"] = go.layout.Template(
        layout=dict(
            font=dict(family="Inter, -apple-system, Segoe UI, sans-serif",
                      size=14, color=TEXT_DARK),
            paper_bgcolor="white",
            plot_bgcolor="white",
            xaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=BORDER_COLOR,
                       linecolor=AXIS_COLOR, linewidth=1.5,
                       showline=True, ticks="outside",
                       ticklen=5, tickcolor=AXIS_COLOR,
                       title_font=dict(size=13, color=TEXT_DARK)),
            yaxis=dict(gridcolor=GRID_COLOR, zerolinecolor=BORDER_COLOR,
                       linecolor=AXIS_COLOR, linewidth=1.5,
                       showline=True, ticks="outside",
                       ticklen=5, tickcolor=AXIS_COLOR,
                       title_font=dict(size=13, color=TEXT_DARK)),
            colorway=[ACTUAL_BLUE, ML_RED, ACTUAL_GREEN, "#a16207", "#7c3aed"],
            margin=dict(l=60, r=20, t=50, b=50),
        )
    )
    pio.templates.default = "thesis"


# === plot helpers ===

def build_hero(cell, cursor_idx, threshold, show_cqr=True):
    ts_hist = cell["history"]["timestamps"]
    t0 = ts_hist[0]
    hist_x = np.array([(t - t0) / 60 for t in ts_hist])
    hist_y = np.array(cell["history"]["cpu_actual"])

    pred = cell["predictions"]
    issue_x = np.array([(t - t0) / 60 for t in pred["issue_timestamps"]])
    target_x = np.array([(t - t0) / 60 for t in pred["target_timestamps"]])
    ml_y = np.array(pred["ml_pred"])
    naive_y = np.array(pred["naive_pred"])
    y_true = np.array(pred["y_true"])
    cqr_lo = np.array(pred["cqr_lower"]) if pred.get("cqr_lower") else None
    cqr_hi = np.array(pred["cqr_upper"]) if pred.get("cqr_upper") else None

    fig = go.Figure()

    if show_cqr and cqr_lo is not None:
        fig.add_trace(go.Scatter(
            x=target_x, y=cqr_hi,
            line=dict(width=0), showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=target_x, y=cqr_lo,
            fill="tonexty", fillcolor="rgba(185, 28, 28, 0.18)",
            line=dict(width=0), name="Empirical ~80% interval",
            hovertemplate="lower: %{y:.1f}%<extra></extra>",
        ))

    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines", name="Actual CPU",
        line=dict(color=ACTUAL_BLUE, width=2.8),
        hovertemplate="t=%{x:.0f}min  CPU=%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=target_x, y=naive_y, mode="lines", name="Naive forecast",
        line=dict(color=NAIVE_GRAY, width=2, dash="dot"),
        hovertemplate="naive=%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=target_x, y=ml_y, mode="lines", name="ML forecast",
        line=dict(color=ML_RED, width=2.8, dash="dash"),
        hovertemplate="ml=%{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=threshold, line=dict(color=TEXT_MUTED, width=1.5, dash="dashdot"),
        annotation_text=f"threshold {threshold:.0f}%",
        annotation_position="top right",
        annotation_font=dict(size=12, color=TEXT_DARK),
    )

    if show_cqr and cqr_hi is not None:
        viol_mask = cqr_hi > threshold
        if viol_mask.any():
            in_region = False
            region_start = None
            for i in range(len(viol_mask)):
                if viol_mask[i] and not in_region:
                    region_start = target_x[i]
                    in_region = True
                elif not viol_mask[i] and in_region:
                    fig.add_vrect(x0=region_start, x1=target_x[i],
                                  fillcolor="rgba(185, 28, 28, 0.10)",
                                  layer="below", line_width=0)
                    in_region = False
            if in_region:
                fig.add_vrect(x0=region_start, x1=target_x[-1],
                              fillcolor="rgba(185, 28, 28, 0.10)",
                              layer="below", line_width=0)

    cursor_issue_x = issue_x[cursor_idx]
    cursor_target_x = target_x[cursor_idx]
    fig.add_vline(x=cursor_issue_x, line=dict(color=TEXT_DARK, width=2.5),
                  annotation_text="now", annotation_position="top",
                  annotation_font=dict(size=12, color=TEXT_DARK))

    fig.add_trace(go.Scatter(
        x=[cursor_target_x], y=[ml_y[cursor_idx]],
        mode="markers", name="ML @ cursor+h",
        marker=dict(color=ML_RED, size=14, symbol="circle",
                    line=dict(color="white", width=3)),
        hovertemplate="ML at t+h: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[cursor_target_x], y=[naive_y[cursor_idx]],
        mode="markers", name="Naive @ cursor+h",
        marker=dict(color=NAIVE_GRAY, size=12, symbol="circle",
                    line=dict(color="white", width=3)),
        hovertemplate="Naive at t+h: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[cursor_target_x], y=[y_true[cursor_idx]],
        mode="markers", name="Actual @ cursor+h",
        marker=dict(color=ACTUAL_GREEN, size=16, symbol="star",
                    line=dict(color="white", width=3)),
        hovertemplate="Actual at t+h: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        height=500,
        xaxis_title="minutes from test-window start",
        yaxis_title="CPU %",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(size=12)),
        margin=dict(l=60, r=20, t=70, b=50),
    )
    return fig


def compact_tile(cell, rank=None):
    p = cell["predictions"]
    targets = p["target_timestamps"]
    t0 = targets[0]
    x = [(t - t0) / 60 for t in targets]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=p["y_true"], mode="lines",
                             line=dict(color=ACTUAL_BLUE, width=2), name="Actual"))
    fig.add_trace(go.Scatter(x=x, y=p["naive_pred"], mode="lines",
                             line=dict(color=NAIVE_GRAY, width=1.4, dash="dot"),
                             name="Naive"))
    fig.add_trace(go.Scatter(x=x, y=p["ml_pred"], mode="lines",
                             line=dict(color=ML_RED, width=1.6, dash="dash"),
                             name="ML"))

    if rank is not None:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.95,
            text=f"<b>#{rank}</b>", showarrow=False,
            font=dict(size=15, color=TEXT_DARK),
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor=BORDER_COLOR, borderwidth=1, borderpad=4,
        )

    fig.update_layout(
        height=200,
        margin=dict(l=10, r=10, t=10, b=24),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, title=None),
        yaxis=dict(tickfont=dict(size=10, color=TEXT_DARK),
                   gridcolor=GRID_COLOR, title=None),
        plot_bgcolor="white",
    )
    return fig


def build_diverging_hpa(sub):
    ml = sub[sub["label"] == "ML-Proactive"].iloc[0]
    react = sub[sub["label"] != "ML-Proactive"].copy()

    react["dv"] = (react["violation_rate"] - ml["violation_rate"]) * 100
    react["dw"] = (react["waste_rate"] - ml["waste_rate"]) * 100
    react["ds"] = react["viol_severity"] - ml["viol_severity"]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=(
            f"<b>SLO violations vs ML</b><br><span style='font-size:11px;color:{TEXT_MUTED}'>ML baseline: {ml['violation_rate']*100:.1f}%</span>",
            f"<b>Resource waste vs ML</b><br><span style='font-size:11px;color:{TEXT_MUTED}'>ML baseline: {ml['waste_rate']*100:.1f}%</span>",
            f"<b>Violation severity vs ML</b><br><span style='font-size:11px;color:{TEXT_MUTED}'>ML baseline: {ml['viol_severity']:.3f}</span>",
        ),
        horizontal_spacing=0.10,
    )

    def color_for(delta):
        return ZONE_COLOR["red"] if delta > 0 else ZONE_COLOR["green"]

    for col, (col_name, fmt) in enumerate(
        [("dv", "{:+.2f}pp"), ("dw", "{:+.2f}pp"), ("ds", "{:+.3f}")], start=1
    ):
        colors = [color_for(v) for v in react[col_name]]
        fig.add_trace(go.Bar(
            y=react["label"], x=react[col_name],
            orientation="h", marker=dict(color=colors,
                                          line=dict(color=TEXT_DARK, width=0.5)),
            text=[fmt.format(v) for v in react[col_name]],
            textposition="auto", showlegend=False,
            textfont=dict(size=12, color="white"),
        ), row=1, col=col)
        fig.add_vline(x=0, line=dict(color=TEXT_DARK, width=2.5), row=1, col=col)

    for col in [1, 2, 3]:
        fig.update_xaxes(automargin=True, row=1, col=col)

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=80, b=40),
        plot_bgcolor="white",
    )
    fig.update_xaxes(zeroline=False, gridcolor=GRID_COLOR,
                     tickfont=dict(size=11, color=TEXT_DARK))
    fig.update_yaxes(autorange="reversed", gridcolor=GRID_COLOR,
                     tickfont=dict(size=12, color=TEXT_DARK))
    return fig


def build_reliability(coverage):
    horizons_list = ["10min", "30min", "60min", "120min"]
    nominal = [coverage[h]["Nominal"] for h in horizons_list]
    empirical = [coverage[h]["Coverage"] for h in horizons_list]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0.5, 1.0], y=[0.5, 1.0],
        mode="lines", line=dict(color=TEXT_MUTED, dash="dash", width=1.5),
        name="perfect calibration",
    ))
    fig.add_trace(go.Scatter(
        x=nominal, y=empirical,
        mode="markers+text",
        marker=dict(color=ML_RED, size=14, line=dict(color="white", width=3)),
        text=horizons_list, textposition="top right",
        textfont=dict(size=11, color=TEXT_DARK),
        name="empirical coverage",
        hovertemplate="%{text}: %{y:.4f}<extra></extra>",
    ))
    fig.update_layout(
        height=340,
        xaxis=dict(title="nominal coverage", range=[0.75, 0.85], dtick=0.025,
                   tickfont=dict(size=11)),
        yaxis=dict(title="empirical coverage", range=[0.75, 0.85], dtick=0.025,
                   tickfont=dict(size=11)),
        legend=dict(orientation="h", y=-0.2, font=dict(size=12)),
        margin=dict(l=60, r=20, t=20, b=60),
    )
    return fig


# === render functions ===

def render_verdict_pill(cell, cursor_idx):
    v = cursor_verdict(cell, cursor_idx)
    moment_arrow = "▲" if v["moment_text"].startswith("ML beats") else (
                   "▼" if v["moment_text"].startswith("ML loses") else "—")
    st.markdown(
        f"<div class='verdict-pill' style='border-left-color:{v['zone_color']};'>"
        f"<span class='verdict-zone' style='color:{v['zone_color']}'>"
        f"{v['zone'].upper()}</span>"
        f"<span class='verdict-main' style='color:{v['moment_color']}'>"
        f"{moment_arrow} {v['moment_text']} by {v['moment_delta']:.1f}% CPU at this cursor"
        f"</span>"
        f"<span class='verdict-meta'>"
        f"horizon {v['horizon']} · ACF@24h={v['acf']:.2f} · {v['overall_text']}"
        f"</span></div>",
        unsafe_allow_html=True,
    )


def render_coverage_badge(hz):
    cov = load_coverage()
    row = cov[hz]
    emp = row["Coverage"] * 100
    nom = row["Nominal"] * 100
    gap = row["Coverage_Gap_pp"]
    width = row["Mean_Width"]
    method = row["Method"]
    direction = "over-covers" if gap > 0 else "under-covers"
    st.markdown(
        f"<div class='coverage-badge'>"
        f"<b>Empirical {nom:.0f}% prediction interval</b> · "
        f"actual coverage <b>{emp:.2f}%</b> on test set "
        f"(<i>{direction}</i> by {abs(gap):.2f}pp · "
        f"mean width {width:.2f}% CPU · {method})"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_compare_grid_sorted(hz):
    by_slot = {c["slot"]: c for c in demo["selected"]}
    enriched = []
    for slot in ["A", "B", "C", "D"]:
        c = by_slot[slot]
        cell = load_cell(c["container_id"], hz)
        enriched.append({
            "slot": slot, "cell": cell,
            "delta_pp": cell["metadata"]["delta_pp"],
            "label": cell["metadata"]["label"],
            "container_id": c["container_id"],
        })
    enriched.sort(key=lambda e: e["delta_pp"], reverse=True)

    st.caption(
        f"Sorted by ML lift over persistence at {hz}. "
        f"Best on the upper-left, worst on the lower-right."
    )

    rows = [enriched[:2], enriched[2:]]
    for rank_offset, row in enumerate(rows):
        cols = st.columns(2, gap="medium")
        for col_idx, (col, e) in enumerate(zip(cols, row)):
            rank = rank_offset * 2 + col_idx + 1
            with col:
                m = e["cell"]["metadata"]
                zone = m["bcf_zone"].lower()
                border = ZONE_COLOR.get(zone, "#888")
                delta = e["delta_pp"]
                arrow = "+" if delta >= 0 else ""
                st.markdown(
                    f"<div class='tile-header' style='border-left-color:{border}'>"
                    f"<b>#{rank} · Slot {e['slot']} - {e['label']}</b><br>"
                    f"<span class='tile-meta'>"
                    f"&Delta; {arrow}{delta:.1f}pp &middot; "
                    f"ACF@24h={m['acf_24h']:.2f} &middot; "
                    f"BCF {zone}</span></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    compact_tile(e["cell"], rank=rank),
                    use_container_width=True,
                    key=f"tile_{e['container_id']}_{hz}",
                )


def render_hpa_panel(hz):
    """v2 schema: two policy knobs (target_util, safety_margin). Each
    (horizon, target_util, safety_margin) tuple yields exactly 5 rows:
    1 ML-Proactive + 4 reactive lags. variant column encodes target_util as
    a "_tu=X" suffix; we strip it for display labels."""
    df = load_hpa()
    df = df[df["Horizon"] == hz].copy()

    # --- target_util slider (new in v4) ---
    tu_options = sorted(df["target_util"].unique())
    default_tu = 0.7 if 0.7 in tu_options else tu_options[len(tu_options) // 2]
    tu = st.select_slider(
        "Target utilization (HPA setpoint — fraction of capacity at which to scale)",
        options=tu_options,
        value=default_tu,
        key=f"hpa_tu_{hz}",
    )
    df = df[df["target_util"] == tu].copy()

    # --- safety_margin slider ---
    margins = sorted(df["safety_margin"].unique())
    margin_options = [round(m, 3) for m in margins]
    default_sm = 1.0 if 1.0 in margin_options else margin_options[0]
    sm = st.select_slider(
        "Safety margin (multiplier on requested capacity)",
        options=margin_options,
        value=default_sm,
        key=f"hpa_sm_{hz}",
    )

    sub = df[df["safety_margin"] == sm].copy()
    if sub.empty:
        st.warning(
            f"No HPA rows for horizon={hz}, target_util={tu}, "
            f"safety_margin={sm}. Check {HPA_PATH}."
        )
        return

    sub["label"] = sub.apply(
        lambda r: "ML-Proactive" if r["Strategy"] == "ML-Proactive"
                  else strip_tu_suffix(r["variant"]),
        axis=1,
    )
    order = ["ML-Proactive", "lag=1", "lag=2", "lag=3", "lag=4"]
    have = [x for x in order if x in sub["label"].values]
    sub = sub.set_index("label").loc[have].reset_index()

    fig = build_diverging_hpa(sub)
    st.plotly_chart(fig, use_container_width=True, key=f"hpa_{hz}_{tu}_{sm}")

    # --- caption with conditional phrasing for sign flips (v4) ---
    ml = sub[sub["label"] == "ML-Proactive"].iloc[0]
    react = sub[sub["label"] != "ML-Proactive"]
    avg_dv = (react["violation_rate"].mean() - ml["violation_rate"]) * 100
    avg_dw = (ml["waste_rate"] - react["waste_rate"].mean()) * 100

    if avg_dv >= 0:
        viol_phrase = f"lowers violations by {avg_dv:.1f}pp"
    else:
        viol_phrase = f"raises violations by {abs(avg_dv):.1f}pp"
    if avg_dw >= 0:
        waste_phrase = f"uses {avg_dw:.1f}pp more capacity"
    else:
        waste_phrase = f"uses {abs(avg_dw):.1f}pp less capacity"

    st.caption(
        f"At target_util={tu}, safety margin={sm}, ML-Proactive {viol_phrase} "
        f"on average and {waste_phrase} vs the average reactive lag. "
        f"Red bars = reactive worse than ML; green bars = reactive better than ML."
    )


def render_info_cards(cell, cursor_idx):
    m = cell["metadata"]
    p = cell["predictions"]
    zone = m["bcf_zone"].lower()
    color = ZONE_COLOR.get(zone, "#888")
    verdict, why = ZONE_VERDICT.get(zone, ("Unknown", ""))

    naive_v = p["naive_pred"][cursor_idx]
    ml_v = p["ml_pred"][cursor_idx]
    actual = p["y_true"][cursor_idx]
    refinement = ml_v - naive_v

    horizon_min = m["h_steps"] * m["cadence_min"]
    acf_pass = m["acf_24h"] >= 0.2
    hz_pass = horizon_min >= 30
    acf_op = "&ge;" if acf_pass else "&lt;"
    hz_op = "&ge;" if hz_pass else "&lt;"

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(
            f"<div class='info-card info-card-l' style='border-left-color:{color}'>"
            f"<div class='info-label'>BCF BOUNDARY CONDITION</div>"
            f"<div class='info-headline' style='color:{color}'>{zone.upper()}</div>"
            f"<div class='info-body'>"
            f"ACF@24h = <b>{m['acf_24h']:.2f}</b> ({acf_op} 0.2)<br>"
            f"horizon = <b>{m['horizon']}</b> ({hz_op} 30min)"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with c2:
        # v4: white-space:pre on .info-decomp + left-padded labels (:<16)
        # + right-padded values (:>7.2f) so the column actually aligns.
        # Refinement uses :>+7.2f for auto-sign and consistent width.
        # Previous version used :6.2f-padded values inside <b>...</b>,
        # which HTML collapsed, plus manual nbsp counts that didn't match
        # the label widths.
        st.markdown(
            f"<div class='info-card'>"
            f"<div class='info-label'>FORECAST DECOMPOSITION</div>"
            f"<div class='info-decomp'>"
            f"{'Naive baseline:':<16}<b>{naive_v:>7.2f}</b>\n"
            f"{'ML refinement:':<16}<b>{refinement:>+7.2f}</b>\n"
            f"<span style='color:{ML_RED}'>{'ML forecast:':<16}<b>{ml_v:>7.2f}</b></span>\n"
            f"<span style='color:{ACTUAL_BLUE}'>{'Actual:':<16}<b>{actual:>7.2f}</b></span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"<div class='info-card info-card-l' style='border-left-color:{color}'>"
            f"<div class='info-label'>RECOMMENDED ACTION</div>"
            f"<div class='info-action'>{verdict}</div>"
            f"<div class='info-action-body'>{why}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )


def render_drilldown(cell):
    m = cell["metadata"]
    cid = m["container_id"]
    hz = m["horizon"]

    with st.expander("Drill-down: per-horizon R-squared, SHAP, calibration",
                     expanded=False):
        all_hz = sorted(horizons, key=lambda h: int(h.replace("min", "")))
        ml_r2s, naive_r2s = [], []
        for h in all_hz:
            cc = load_cell(cid, h)
            ml_r2s.append(cc["metadata"]["et_r2"])
            naive_r2s.append(cc["metadata"]["naive_r2"])

        st.markdown(f"**Per-horizon R-squared for {cid}**")
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(x=all_hz, y=naive_r2s, name="Naive",
                                marker=dict(color=NAIVE_GRAY,
                                             line=dict(color=TEXT_DARK, width=0.5)),
                                text=[f"{v:.3f}" for v in naive_r2s],
                                textposition="auto",
                                textfont=dict(size=12, color="white")))
        fig_r2.add_trace(go.Bar(x=all_hz, y=ml_r2s, name="ML (ExtraTrees)",
                                marker=dict(color=ML_RED,
                                             line=dict(color=TEXT_DARK, width=0.5)),
                                text=[f"{v:.3f}" for v in ml_r2s],
                                textposition="auto",
                                textfont=dict(size=12, color="white")))
        fig_r2.update_layout(
            height=300, barmode="group", plot_bgcolor="white",
            margin=dict(l=20, r=20, t=10, b=30),
            legend=dict(orientation="h", y=-0.18, font=dict(size=12)),
            yaxis=dict(title="R-squared", gridcolor=GRID_COLOR),
            xaxis=dict(tickfont=dict(size=12, color=TEXT_DARK)),
        )
        st.plotly_chart(fig_r2, use_container_width=True, key=f"drill_r2_{cid}")

        st.markdown(f"**Top 10 SHAP features at {hz} (Alibaba ensemble, global)**")
        shap_df = load_shap()
        top = shap_df.nlargest(10, hz)[["feature", hz]].copy()
        top["group"] = top["feature"].apply(feature_group)
        group_color = {
            "hour-of-day": "#1d4ed8", "day/business": "#3b82f6",
            "rolling": "#a16207", "lag": "#15803d",
            "cross-resource": "#7c3aed", "cluster": "#475569", "other": "#64748b",
        }
        colors = [group_color[g] for g in top["group"]]

        fig_shap = go.Figure(go.Bar(
            x=top[hz][::-1], y=top["feature"][::-1],
            orientation="h",
            marker=dict(color=colors[::-1],
                        line=dict(color=TEXT_DARK, width=0.5)),
            text=[f"{v:.3f}" for v in top[hz][::-1]],
            textposition="outside",
            textfont=dict(size=11, color=TEXT_DARK),
        ))
        fig_shap.update_layout(
            height=360, plot_bgcolor="white",
            margin=dict(l=30, r=30, t=10, b=30),
            xaxis=dict(title="mean |SHAP|", gridcolor=GRID_COLOR),
            yaxis=dict(tickfont=dict(size=12, color=TEXT_DARK)),
        )
        st.plotly_chart(fig_shap, use_container_width=True,
                        key=f"drill_shap_{cid}_{hz}")

        st.markdown("**Calibration: empirical vs nominal coverage across horizons**")
        cov = load_coverage()
        fig_rel = build_reliability(cov)
        st.plotly_chart(fig_rel, use_container_width=True, key=f"drill_rel_{cid}")

        st.caption(
            "All four horizons over-cover by 1.3-2.0pp - the empirical 80% interval "
            "actually contains 81-82% of test points. Recalibrated via AgACI "
            "(Zaffran et al., ICML 2022). Calibration is conservative, not loose."
        )


def render_provenance_footer():
    st.markdown(
        "<div class='provenance-footer'>"
        "Dataset: <b>Alibaba Cluster Trace 2018</b> &middot; 4920 containers &middot; "
        "5-min cadence &middot; chronological train/val/test split &middot; "
        "Persistence baseline: ŷ_{t+h} = y_t &middot; "
        "Ensemble: NNLS over XGB+LGB+ET+BiLSTM (OOF) &middot; "
        "Conformal: AgACI recalibrated to 80% nominal &middot; "
        "<i>All predictions precomputed - no live inference</i><br>"
        "Hybrid Ensemble Learning for Proactive Resource Prediction in Kubernetes "
        "&middot; Bachelor's thesis, IU VNU-HCM 2026"
        "</div>",
        unsafe_allow_html=True,
    )


def section_break():
    st.markdown("<div class='section-break'></div>", unsafe_allow_html=True)


# === cursor block (verdict pill + hero + readout + info cards) ===
# This block is what re-renders during the tour. Static rendering function
# is shared with the non-tour path; the fragment wrapper adds cursor-advance
# logic for the tour case.

def render_cursor_block(cell, active_cid, active_hz, threshold):
    """Render the cursor-dependent UI: verdict pill, hero plot, cursor
    readout (4 metrics), BCF zone caption, info cards. Called once per
    full-app render when tour is off/paused, and once per fragment tick
    via cursor_block_fragment when tour is active+unpaused."""
    cursor_key = f"cursor_{active_cid}_{active_hz}"
    cursor_idx = st.session_state.get(cursor_key, 0)

    # verdict pill (cursor-dependent moment text)
    render_verdict_pill(cell, cursor_idx)

    # hero plot
    fig = build_hero(cell, cursor_idx, threshold, show_cqr=True)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"hero_{active_cid}_{active_hz}")

    # cursor readout (4 metrics)
    pred = cell["predictions"]
    ml_at = pred["ml_pred"][cursor_idx]
    naive_at = pred["naive_pred"][cursor_idx]
    true_at = pred["y_true"][cursor_idx]
    ml_err = abs(ml_at - true_at)
    naive_err = abs(naive_at - true_at)

    r1, r2, r3, r4 = st.columns(4, gap="medium")
    r1.metric("ML forecast", f"{ml_at:.1f}%",
              delta=f"err {ml_err:.1f}%", delta_color="inverse")
    r2.metric("Naive forecast", f"{naive_at:.1f}%",
              delta=f"err {naive_err:.1f}%", delta_color="inverse")
    r3.metric("Actual outcome", f"{true_at:.1f}%")
    r4.metric("ML beats naive by", f"{naive_err - ml_err:+.1f}%")

    meta = cell["metadata"]
    st.caption(f"BCF zone: **{meta['bcf_zone'].upper()}** - "
               f"{meta.get('bcf_reason', '')}")

    section_break()
    render_info_cards(cell, cursor_idx)


@st.fragment(run_every="1s")
def cursor_block_fragment(cell, active_cid, active_hz, threshold):
    """Tour-active wrapper. Each tick: maybe advance cursor, then render the
    block. NEVER calls st.rerun — fragment's own re-render is sufficient,
    avoiding the v4 rerun-per-tick race that froze the main pane.

    Args are captured from the calling context. They refresh on every full
    app rerun (e.g. when user pauses, when view transitions). Between full
    reruns, the fragment uses the cell/cid/hz/threshold from its last
    registration; that's correct because those don't change during a single
    view's lifetime."""
    # DEBUG: count cursor fragment entries
    st.session_state._dbg_cursor_ticks = (
        st.session_state.get("_dbg_cursor_ticks", 0) + 1
    )

    # advance cursor if tour is active and not paused. The slider widget is
    # hidden in this state (see main pane code), so writing the cursor key
    # is safe — no widget claims it on this tick.
    if (st.session_state.get("demo_mode")
            and not st.session_state.get("demo_paused")):
        cursor_key = f"cursor_{active_cid}_{active_hz}"
        n_pred = len(cell["predictions"]["ml_pred"])
        current = st.session_state.get(cursor_key, 0)
        new_val = current + PLAY_STEPS_PER_TICK
        if new_val >= n_pred:
            new_val = 0  # loop back to start of prediction window
        st.session_state[cursor_key] = new_val
        st.session_state._dbg_cursor_advances = (
            st.session_state.get("_dbg_cursor_advances", 0) + 1
        )

    render_cursor_block(cell, active_cid, active_hz, threshold)


# === Demo Mode helpers ===

def apply_demo_view(idx):
    """Apply a tour view. The `view` radio is widget-bound (key="view"), so
    its write is deferred via pending_view_change and drained at top of next
    full-app rerun. The cursor key isn't widget-bound at the moment this
    function is called (we're inside a fragment-only tick; the slider, if
    visible at all, hasn't re-instantiated on this tick), so writing it
    directly is safe — and after the view-advance st.rerun the slider may
    not even render (if tour stays active+unpaused, which it does at view
    transitions)."""
    v = DEMO_VIEWS[idx]

    # plain session_state writes — not widget-bound, always safe
    st.session_state.view_index = idx
    st.session_state.container_id = v["cid"]
    st.session_state.horizon = v["hz"]
    st.session_state.view_started_at = time.time()
    st.session_state.view_elapsed_at_pause = 0.0

    # `view` IS bound to the radio (key="view"). Defer.
    st.session_state.pending_view_change = v["view"]

    # cursor seed for Single container views with start_pct
    if v["view"] == "Single container" and v.get("start_pct") is not None:
        cursor_key = f"cursor_{v['cid']}_{v['hz']}"
        cell = load_cell(v["cid"], v["hz"])  # cached
        n_pred = len(cell["predictions"]["ml_pred"])
        start_idx = int(v["start_pct"] * n_pred)
        st.session_state[cursor_key] = start_idx


def pause_demo_if_active():
    if st.session_state.get("demo_mode") and not st.session_state.get("demo_paused"):
        elapsed = time.time() - st.session_state.get("view_started_at", time.time())
        st.session_state.view_elapsed_at_pause = elapsed
        st.session_state.demo_paused = True


@st.fragment(run_every="1s")
def demo_tick_and_caption():
    """Caption renderer + view-advance only. Runs every 1s. Calls
    st.rerun(scope='app') only on view transitions (every ~14s), not per-tick.
    Cursor advance lives in cursor_block_fragment to avoid the rerun-per-tick
    race that froze the main pane in v4."""
    if not st.session_state.get("demo_mode"):
        return

    # DEBUG: count fragment entries and timestamp this tick
    st.session_state._dbg_caption_ticks = (
        st.session_state.get("_dbg_caption_ticks", 0) + 1
    )
    st.session_state._dbg_last_tick_at = time.time()

    idx = st.session_state.get("view_index", 0)
    v = DEMO_VIEWS[idx]
    duration = v.get("duration", 15)
    paused = st.session_state.get("demo_paused", False)

    if paused:
        elapsed = st.session_state.get("view_elapsed_at_pause", 0.0)
    else:
        elapsed = time.time() - st.session_state.get("view_started_at", time.time())

    # DEBUG: snapshot the fragment's view of the world
    st.session_state._dbg_last_elapsed = elapsed
    st.session_state._dbg_last_paused = paused
    st.session_state._dbg_last_idx = idx
    st.session_state._dbg_last_view_str = v["view"]

    # --- step 2: render caption FIRST so it lands on every tick. previous
    # versions rendered the caption at the bottom of the function, but the
    # reruns triggered by view-advance and cursor-advance abort execution
    # before the caption block, leaving the caption stale. moving it up
    # ensures the caption text and progress bar update visibly each second.
    remaining = max(0, int(duration - elapsed))
    if paused:
        status_line = f"PAUSED at {int(elapsed)}s / {duration}s"
    else:
        status_line = f"advancing in {remaining}s"
    progress_pct = min(100, int((elapsed / duration) * 100))
    bar_color = "#fbbf24" if paused else "#22c55e"

    st.markdown(
        f"<div class='demo-caption'>"
        f"<div class='demo-caption-row'>"
        f"<span class='demo-tag'>DEFENSE TOUR</span>"
        f"<span class='demo-status'>{status_line}</span>"
        f"</div>"
        f"<div class='demo-caption-text'>{v['caption']}</div>"
        f"<div class='demo-progress-track'>"
        f"<div class='demo-progress-fill' style='width:{progress_pct}%;background:{bar_color}'></div>"
        f"</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

    # --- step 3: time-based view advance ---
    # Only path that calls st.rerun. Fires at most once per ~14s. Far from
    # the v4 problem of rerun-per-tick.
    if not paused and elapsed >= duration:
        st.session_state._dbg_view_advances = (
            st.session_state.get("_dbg_view_advances", 0) + 1
        )
        next_idx = (idx + 1) % len(DEMO_VIEWS)
        apply_demo_view(next_idx)
        st.rerun(scope="app")
        return

    # cursor advance lives in cursor_block_fragment — see render_cursor_block
    # and cursor_block_fragment below.


# === page setup ===

st.set_page_config(page_title="Kubernetes Workload Forecaster", layout="wide")
setup_plotly()

st.markdown("""
<style>
.block-container { padding-top: 1.5rem !important; padding-bottom: 4rem !important;
                   max-width: 1240px; }
.block-container h1, .block-container h2, .block-container h3 { color: #0f172a; }
h1 { font-size: 1.75rem !important; font-weight: 700 !important; margin-bottom: 0.4rem; }
h2 { font-size: 1.3rem !important; font-weight: 700 !important; margin-top: 2rem !important; }
h3 { font-size: 1.05rem !important; font-weight: 600 !important; }
[data-testid="stHorizontalBlock"] { gap: 1.25rem; }
[data-testid="stMetricValue"] { font-weight: 600 !important; color: #0f172a !important; }
[data-testid="stMetricLabel"] { color: #475569 !important; }

.section-break { height: 1px; background: #cbd5e1; margin: 36px 0 28px 0; border: none; }
.stApp hr { border: 0; border-top: 1px solid #cbd5e1; margin: 32px 0; }

/* v4: stripped the dead `transition: transform/box-shadow` line; there were
   no :hover rules to drive it, so it had no visible effect. */
.tile {
    border: 2.5px solid #cbd5e1; border-radius: 8px; padding: 12px 14px;
    margin-bottom: 8px; background: #ffffff;
}
.tile-green  { border-color: #15803d; background: #f0fdf4; }
.tile-yellow { border-color: #a16207; background: #fefce8; }
.tile-red    { border-color: #b91c1c; background: #fef2f2; }
.tile-selected { box-shadow: 0 0 0 3px #0f172a, 0 2px 6px rgba(0,0,0,0.08); }
.tile-slot { font-weight: 700; font-size: 14px; color: #0f172a; }
.tile-id { font-size: 11px; color: #475569; font-family: monospace; margin-top: 2px; }

.verdict-pill {
    display: flex; align-items: center; gap: 18px; flex-wrap: wrap;
    padding: 14px 20px; border-left: 8px solid #888;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #ffffff;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin: 8px 0 18px 0;
    font-size: 15px;
}
.verdict-zone { font-weight: 800; letter-spacing: 0.6px; font-size: 14px; }
.verdict-main { font-weight: 700; font-size: 16px; }
.verdict-meta { color: #475569; font-size: 12px; margin-left: auto; font-weight: 500; }

.coverage-badge {
    padding: 12px 16px; background: #f8fafc;
    border: 1.5px solid #cbd5e1; border-radius: 6px;
    font-size: 13px; color: #0f172a; margin: 4px 0 16px 0;
}
.coverage-badge b { color: #0f172a; font-weight: 700; }

.info-card {
    border: 2px solid #e2e8f0; border-radius: 8px; padding: 18px 20px;
    height: 200px; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.info-card-l { border-left-width: 8px; border-left-style: solid; }
.info-label { font-size: 11px; color: #475569; letter-spacing: 0.8px; font-weight: 700; margin-bottom: 6px; }
.info-headline { font-size: 26px; font-weight: 700; margin-top: 6px; letter-spacing: 0.5px; }
.info-body { font-size: 14px; color: #0f172a; line-height: 1.65; margin-top: 10px; }
.info-body b { color: #0f172a; }

/* v4: white-space: pre preserves the leading/trailing spaces emitted by
   Python format specs (:<16, :>7.2f), so the column actually aligns. The
   <b> tags inside are still inline and don't disrupt the layout. */
.info-decomp {
    font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace;
    font-size: 14px; color: #0f172a; line-height: 1.85; margin-top: 8px;
    white-space: pre;
}

.info-action { font-size: 18px; font-weight: 700; color: #0f172a; margin-top: 6px; }
.info-action-body { font-size: 13px; color: #475569; margin-top: 10px; line-height: 1.55; }

.tile-header {
    border-left: 5px solid #888; padding: 6px 12px; margin-bottom: 4px;
    background: #f8fafc; border-radius: 0 4px 4px 0;
}
.tile-header b { color: #0f172a; font-size: 14px; }
.tile-meta { font-size: 12px; color: #475569; }

.demo-caption {
    background: #000000; color: #ffffff;
    padding: 14px 20px; border-radius: 8px;
    margin: 8px 0 18px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.15);
}
.demo-caption-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
.demo-tag {
    background: #fbbf24; color: #000000; padding: 3px 10px; border-radius: 4px;
    font-size: 11px; font-weight: 800; letter-spacing: 0.8px;
}
.demo-status { color: #fbbf24; font-size: 12px; font-weight: 600; font-family: monospace; letter-spacing: 0.5px; }
.demo-caption-text { font-size: 15px; font-weight: 500; line-height: 1.55; color: #f8fafc; margin-bottom: 10px; }
.demo-progress-track { height: 4px; background: #374151; border-radius: 2px; overflow: hidden; }
.demo-progress-fill { height: 100%; transition: width 0.8s linear; }

.provenance-footer {
    margin-top: 36px; padding: 16px 20px;
    border-top: 2px solid #cbd5e1;
    color: #475569; font-size: 12px;
    text-align: center; line-height: 1.8;
    background: #f8fafc; border-radius: 6px;
}
.provenance-footer b { color: #0f172a; }

.keymap {
    background: #f8fafc; border: 1.5px solid #cbd5e1; border-radius: 6px;
    padding: 14px 16px; font-size: 13px; line-height: 1.95;
    margin: 8px 0 16px 0; color: #0f172a;
}
.keymap kbd {
    background: #0f172a; color: white; padding: 3px 8px; border-radius: 4px;
    font-family: monospace; font-size: 12px; font-weight: 600;
    box-shadow: 0 1px 0 rgba(0,0,0,0.3);
}

[data-testid="stSidebar"] {
    background: #f8fafc;
    border-right: 1px solid #cbd5e1;
}
[data-testid="stSidebar"] h3 {
    color: #475569 !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase;
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
}

@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
    .verdict-meta { width: 100%; margin-left: 0; }
    .info-card { height: auto; min-height: 160px; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Kubernetes Workload Forecaster")
st.caption("*Predicts when ML helps, and tells you when it doesn't.* "
           "Bachelor's thesis · IU VNU-HCM · 2026")
# v4: removed the `if "_last_drain" in st.session_state: st.code(...)` debug
# block that was rendering above the page title in v3.


# === sidebar ===

index = load_index()
demo = load_demo_containers()
horizons = index["horizons"]

# session-state defaults — set BEFORE any widget renders.
# v4: dropped "show_cqr" (no UI toggled it; was always True). Hardcoded
# True at the build_hero call site.
for k, v in [
    ("container_id", "c_41072"),
    ("horizon", "120min"),
    ("view", "Single container"),
    ("demo_mode", False),
    ("demo_paused", False),
    ("view_index", 0),
    ("view_started_at", time.time()),
    ("view_elapsed_at_pause", 0.0),
    ("show_keymap", False),
    ("pending_view_change", None),
    # diagnostic counters / state snapshots — see header comment
    ("_dbg_show", False),
    ("_dbg_caption_ticks", 0),
    ("_dbg_cursor_ticks", 0),
    ("_dbg_cursor_advances", 0),
    ("_dbg_view_advances", 0),
    ("_dbg_last_tick_at", 0.0),
    ("_dbg_last_elapsed", 0.0),
    ("_dbg_last_paused", False),
    ("_dbg_last_idx", 0),
    ("_dbg_last_view_str", "?"),
]:
    if k not in st.session_state:
        st.session_state[k] = v

# === pending_view_change drain ===
# The radio's `view` key is widget-bound, so writes from inside the caption
# fragment (during view-advance) are deferred via pending_view_change and
# applied here, BEFORE the radio re-instantiates. v5 dropped the analogous
# pending_cursor_change drain — the cursor key is no longer widget-bound
# during active tour (slider hidden), so the cursor fragment writes it
# directly with no race.

if st.session_state.pending_view_change is not None:
    st.session_state.view = st.session_state.pending_view_change
    st.session_state.pending_view_change = None


with st.sidebar:
    st.markdown("### Container")
    by_slot = {c["slot"]: c for c in demo["selected"]}

    for slot in ["A", "B", "C", "D"]:
        c = by_slot[slot]
        cid = c["container_id"]
        zone = cell_zone(index, cid, st.session_state.horizon)
        is_sel = (cid == st.session_state.container_id)

        classes = ["tile", f"tile-{zone}"]
        if is_sel:
            classes.append("tile-selected")

        st.markdown(
            f'<div class="{" ".join(classes)}">'
            f'<div class="tile-slot">Slot {slot} - {c["label"]}</div>'
            f'<div class="tile-id">{cid}</div></div>',
            unsafe_allow_html=True,
        )
        # v4: disabled=is_sel — the "Selected" button on the active slot is
        # now a no-op, so it doesn't fire pause_demo_if_active() or rerun
        # the page when an examiner clicks it during the tour.
        if st.button(
            "Select" if not is_sel else "Selected",
            key=f"sel_{slot}",
            disabled=is_sel,
        ):
            st.session_state.container_id = cid
            pause_demo_if_active()
            st.rerun()

    st.markdown("### Horizon")
    cols = st.columns(4)
    for i, h in enumerate(horizons):
        is_sel = (h == st.session_state.horizon)
        label = f"**{h}**" if is_sel else h
        if cols[i].button(label, key=f"hz_{h}"):
            st.session_state.horizon = h
            pause_demo_if_active()
            st.rerun()

    st.markdown("### View")
    # v3: radio uses key="view" so its widget state IS st.session_state.view.
    # The drain above seeds .view from pending_view_change before this radio
    # renders, so the tour's view writes propagate to the sidebar correctly.
    st.radio(
        "view_mode",
        ["Single container", "Compare 4"],
        index=["Single container", "Compare 4"].index(st.session_state.view),
        label_visibility="collapsed", key="view",
        on_change=pause_demo_if_active,
    )

    st.markdown("### Defense tour")
    dm_col1, dm_col2 = st.columns(2)
    if dm_col1.button(
        "Stop tour" if st.session_state.demo_mode else "Start tour",
        key="toggle_demo", use_container_width=True,
    ):
        st.session_state.demo_mode = not st.session_state.demo_mode
        st.session_state.demo_paused = False
        if st.session_state.demo_mode:
            apply_demo_view(0)
        st.rerun()

    if st.session_state.demo_mode:
        if dm_col2.button(
            "Resume" if st.session_state.demo_paused else "Pause",
            key="toggle_pause", use_container_width=True,
        ):
            if not st.session_state.demo_paused:
                elapsed = time.time() - st.session_state.view_started_at
                st.session_state.view_elapsed_at_pause = elapsed
                st.session_state.demo_paused = True
            else:
                elapsed = st.session_state.view_elapsed_at_pause
                st.session_state.view_started_at = time.time() - elapsed
                st.session_state.view_elapsed_at_pause = 0.0
                st.session_state.demo_paused = False
            st.rerun()

    if st.button(
        "Hide keymap" if st.session_state.show_keymap else "Show keymap",
        key="show_keymap_btn", use_container_width=True,
    ):
        st.session_state.show_keymap = not st.session_state.show_keymap
        st.rerun()

    # === DIAGNOSTIC PANEL (off by default; not shown at defense) ===
    if st.button(
        "Hide debug" if st.session_state._dbg_show else "Show debug",
        key="dbg_toggle", use_container_width=True,
    ):
        st.session_state._dbg_show = not st.session_state._dbg_show
        st.rerun()

    if st.session_state._dbg_show:
        cid = st.session_state.container_id
        hz = st.session_state.horizon
        cursor_key = f"cursor_{cid}_{hz}"
        cur_val = st.session_state.get(cursor_key, "—")

        last_tick_at = st.session_state._dbg_last_tick_at
        if last_tick_at > 0:
            tick_age = f"{time.time() - last_tick_at:.1f}s"
        else:
            tick_age = "never"

        # expected counts after t seconds of tour at default 14s/view duration:
        #   caption_ticks ≈ t       (caption fragment, 1Hz)
        #   cursor_ticks ≈ t        (cursor fragment, 1Hz, only registered
        #                            when tour active+unpaused on Single view)
        #   cursor_advances ≈ t × (Single-view share, ~80% across DEMO_VIEWS)
        #   view_advances ≈ t / 14
        expected_note = (
            "expected at t=30s: caption≈30, cursor≈24, "
            "cursor_adv≈24, views≈2"
        )

        cap_ticks = st.session_state._dbg_caption_ticks
        cur_ticks = st.session_state._dbg_cursor_ticks
        c_adv = st.session_state._dbg_cursor_advances
        v_adv = st.session_state._dbg_view_advances
        last_idx = st.session_state._dbg_last_idx
        last_elapsed = st.session_state._dbg_last_elapsed
        last_paused = st.session_state._dbg_last_paused
        last_view_str = st.session_state._dbg_last_view_str

        pend_view = st.session_state.pending_view_change

        st.markdown(
            f"<div style='font-family:ui-monospace,Menlo,Consolas,monospace;"
            f"font-size:11px;line-height:1.5;padding:10px 12px;"
            f"background:#0f172a;color:#e2e8f0;border-radius:6px;"
            f"margin-top:6px;border:1px solid #334155'>"
            f"<div style='color:#fbbf24;font-weight:700;letter-spacing:0.5px;"
            f"margin-bottom:6px'>FRAGMENTS</div>"
            f"caption_ticks: <b style='color:#86efac'>{cap_ticks}</b><br>"
            f"cursor_ticks:&nbsp; <b style='color:#86efac'>{cur_ticks}</b><br>"
            f"cursor_adv:&nbsp;&nbsp; <b style='color:#86efac'>{c_adv}</b><br>"
            f"view_adv:&nbsp;&nbsp;&nbsp;&nbsp; <b style='color:#86efac'>{v_adv}</b><br>"
            f"last_idx:&nbsp;&nbsp;&nbsp;&nbsp; {last_idx} ({last_view_str})<br>"
            f"last_elap:&nbsp;&nbsp;&nbsp; {last_elapsed:.1f}s<br>"
            f"last_paus:&nbsp;&nbsp;&nbsp; {last_paused}<br>"
            f"tick_age:&nbsp;&nbsp;&nbsp;&nbsp; {tick_age}<br>"
            f"<div style='color:#94a3b8;font-size:10px;margin-top:4px'>"
            f"{expected_note}</div>"
            f"<div style='color:#fbbf24;font-weight:700;letter-spacing:0.5px;"
            f"margin-top:10px;margin-bottom:6px'>MAIN PANE</div>"
            f"view_idx (ss):&nbsp; {st.session_state.view_index}<br>"
            f"container:&nbsp;&nbsp;&nbsp;&nbsp; {cid}<br>"
            f"horizon:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; {hz}<br>"
            f"view (radio):&nbsp; {st.session_state.view}<br>"
            f"cursor[k]:&nbsp;&nbsp;&nbsp;&nbsp; {cur_val}<br>"
            f"<div style='color:#fbbf24;font-weight:700;letter-spacing:0.5px;"
            f"margin-top:10px;margin-bottom:6px'>PENDING</div>"
            f"view: {pend_view}"
            f"</div>",
            unsafe_allow_html=True,
        )

        if st.button("Reset counters", key="dbg_reset",
                     use_container_width=True):
            st.session_state._dbg_caption_ticks = 0
            st.session_state._dbg_cursor_ticks = 0
            st.session_state._dbg_cursor_advances = 0
            st.session_state._dbg_view_advances = 0
            st.rerun()

if HAS_SHORTCUTS:
    streamlit_shortcuts.add_shortcuts(
        sel_A="a", sel_B="b", sel_C="c", sel_D="d",
        hz_10min="1", hz_30min="2", hz_60min="3", hz_120min="4",
        toggle_demo="m", toggle_pause="p", show_keymap_btn="?",
    )


# === main ===

if st.session_state.demo_mode:
    demo_tick_and_caption()

active_cid = st.session_state.container_id
active_hz = st.session_state.horizon
view = st.session_state.view

cell = load_cell(active_cid, active_hz)
meta = cell["metadata"]

if st.session_state.show_keymap:
    kb_help = ""
    if not HAS_SHORTCUTS:
        kb_help = (" <i>(streamlit-shortcuts not installed - run "
                   "<code>pip install streamlit-shortcuts</code>)</i>")
    st.markdown(
        "<div class='keymap'>"
        f"<b>Keyboard shortcuts</b>{kb_help}<br>"
        "<kbd>A</kbd> <kbd>B</kbd> <kbd>C</kbd> <kbd>D</kbd> &nbsp; container slot &nbsp;&nbsp; "
        "<kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> <kbd>4</kbd> &nbsp; horizon (10/30/60/120) <br>"
        "<kbd>M</kbd> &nbsp; toggle defense tour &nbsp;&nbsp; "
        "<kbd>P</kbd> &nbsp; pause/resume tour &nbsp;&nbsp; "
        "<kbd>?</kbd> &nbsp; toggle this keymap"
        "</div>",
        unsafe_allow_html=True,
    )


if view == "Single container":
    pred = cell["predictions"]
    ml_y = np.array(pred["ml_pred"])
    naive_y = np.array(pred["naive_pred"])
    y_true = np.array(pred["y_true"])
    hist_y = np.array(cell["history"]["cpu_actual"])

    n_pred = len(ml_y)
    lo_idx = int(n_pred * 0.10)
    hi_idx = int(n_pred * 0.80)
    diffs = np.abs(ml_y - naive_y)
    default_cursor = lo_idx + int(np.argmax(diffs[lo_idx:hi_idx]))
    default_threshold = float(min(
        hist_y.max() * 1.2,
        max(hist_y.max(), ml_y.max()) + 5,
    ))

    cursor_key = f"cursor_{active_cid}_{active_hz}"
    threshold_key = f"thr_{active_cid}_{active_hz}"

    # seed if missing
    if cursor_key not in st.session_state:
        st.session_state[cursor_key] = default_cursor
    if threshold_key not in st.session_state:
        st.session_state[threshold_key] = default_threshold

    # --- static block: headline metrics + coverage badge ---
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    c1.metric("Container", meta["container_id"])
    c2.metric("Horizon", meta["horizon"])
    c3.metric("ACF@24h", f"{meta['acf_24h']:.2f}")
    c4.metric("ML R^2 vs Naive R^2",
              f"{meta['et_r2']:.3f} vs {meta['naive_r2']:.3f}",
              delta=f"{meta['delta_pp']:+.2f}pp")

    render_coverage_badge(active_hz)

    section_break()

    # --- sliders: cursor (hidden during active tour) + threshold ---
    # When tour is active+unpaused, cursor slider hides and the cursor
    # fragment owns the cursor key. When tour is off or paused, slider
    # renders normally with key=cursor_key (no version trick — v5 doesn't
    # need it because the fragment doesn't call st.rerun anymore).
    tour_active_unpaused = (
        st.session_state.get("demo_mode")
        and not st.session_state.get("demo_paused")
    )

    if tour_active_unpaused:
        st.caption(
            "▶ Cursor auto-advancing during tour — click **Pause** in the "
            "sidebar to scrub manually."
        )
        st.slider(
            "CPU threshold (%)",
            min_value=0.0,
            max_value=float(max(hist_y.max(), ml_y.max())) + 10,
            step=1.0,
            key=threshold_key,
        )
    else:
        s1, s2 = st.columns([3, 2], gap="medium")
        s1.slider(
            "Now (cursor scrubs through prediction window)",
            min_value=0, max_value=n_pred - 1,
            key=cursor_key,
        )
        s2.slider(
            "CPU threshold (%)",
            min_value=0.0,
            max_value=float(max(hist_y.max(), ml_y.max())) + 10,
            step=1.0,
            key=threshold_key,
        )

    threshold = st.session_state[threshold_key]

    # --- cursor block: fragment when tour active+unpaused, static otherwise ---
    # Both paths share render_cursor_block. The fragment wrapper adds the
    # 1Hz tick + cursor advance; the static path just renders once.
    if tour_active_unpaused:
        cursor_block_fragment(cell, active_cid, active_hz, threshold)
    else:
        render_cursor_block(cell, active_cid, active_hz, threshold)

    section_break()

    render_drilldown(cell)

else:
    st.subheader(f"All 4 demo containers at {active_hz}")
    render_compare_grid_sorted(active_hz)


section_break()

st.subheader(f"HPA simulation at {active_hz}: ML-Proactive vs reactive lags (diverging)")
render_hpa_panel(active_hz)


render_provenance_footer()