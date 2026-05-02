# Kubernetes Workload Forecaster — Live Demo
# Loads precomputed forecast cells (models/demo/precomputed/*.json) and renders
# them. No model loading, no inference. See CLAUDE.md for build plan.

import json
from pathlib import Path

import streamlit as st


PRECOMPUTED_DIR = Path("models/demo/precomputed")


# === data loading ===

@st.cache_data
def load_index():
    with open(PRECOMPUTED_DIR / "index.json") as f:
        return json.load(f)


@st.cache_data
def load_demo_containers():
    # NOTE: this file lives INSIDE precomputed/, not at models/demo/
    with open(PRECOMPUTED_DIR / "demo_containers.json") as f:
        return json.load(f)


@st.cache_data
def load_cell(container_id, horizon):
    fname = f"cell__{container_id}__{horizon}.json"
    with open(PRECOMPUTED_DIR / fname) as f:
        return json.load(f)


def cell_zone(index, container_id, horizon):
    # bcf_zone is per cell, not per container — A is red at 10min, green at 30/60/120
    for entry in index["cells"]:
        if entry["container_id"] == container_id and entry["horizon"] == horizon:
            return entry["bcf_zone"]
    return "red"


# === page setup ===

st.set_page_config(
    page_title="Kubernetes Workload Forecaster",
    layout="wide",
)

# CSS for the sidebar tile picker. Keeping this minimal — color borders driven
# by the BCF zone of the (container, horizon) pair, not the container alone.
st.markdown("""
<style>
.tile {
    border: 2px solid #d0d0d0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    background: #fafafa;
}
.tile-green { border-color: #22c55e; background: #f0fdf4; }
.tile-red   { border-color: #ef4444; background: #fef2f2; }
.tile-selected { box-shadow: 0 0 0 2px #1f2937; }
.tile-slot { font-weight: 700; font-size: 14px; }
.tile-label { font-size: 13px; color: #374151; }
.tile-id { font-size: 11px; color: #6b7280; font-family: monospace; }

div.stButton > button[kind="secondary"] { width: 100%; }
</style>
""", unsafe_allow_html=True)

# Tighten Streamlit's default top padding and render title in a div with margin

# Tighten Streamlit's default top padding and render title in a div with margin

# Tighten Streamlit's default top padding and render title in a div with margin

st.markdown("""
<style>
.block-container { padding-top: 2rem !important; }
</style>
""", unsafe_allow_html=True)
st.markdown("# Kubernetes Workload Forecaster")
st.caption("*Predicts when ML helps, and tells you when it doesn't.*")


# === sidebar ===

index = load_index()
demo = load_demo_containers()
horizons = index["horizons"]

# selection state in session_state so button clicks persist across reruns
if "container_id" not in st.session_state:
    st.session_state.container_id = "c_41072"  # slot A, hero shot
if "horizon" not in st.session_state:
    st.session_state.horizon = "120min"

with st.sidebar:
    st.markdown("### CONTAINER")

    # demo["selected"] is in slot order from the JSON, but let's enforce A,B,C,D
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
            f'<div class="tile-slot">Slot {slot} — {c["label"]}</div>'
            f'<div class="tile-id">{cid}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Streamlit doesn't let us make divs clickable directly without
        # a component, so a small Select button under each tile.
        if st.button("Select" if not is_sel else "✓ Selected", key=f"sel_{slot}"):
            st.session_state.container_id = cid
            st.rerun()

    st.markdown("### HORIZON")
    cols = st.columns(4)
    for i, h in enumerate(horizons):
        is_sel = (h == st.session_state.horizon)
        label = f"**{h}**" if is_sel else h
        if cols[i].button(label, key=f"hz_{h}"):
            st.session_state.horizon = h
            st.rerun()


# === main ===

cell = load_cell(st.session_state.container_id, st.session_state.horizon)
meta = cell["metadata"]

# header strip with the picked cell's headline numbers
c1, c2, c3, c4 = st.columns(4)
c1.metric("Container", meta["container_id"])
c2.metric("Horizon", meta["horizon"])
c3.metric("ACF@24h", f"{meta['acf_24h']:.2f}")
c4.metric("ML R² vs Naive R²", f"{meta['et_r2']:.3f} vs {meta['naive_r2']:.3f}",
          delta=f"{meta['delta_pp']:+.2f}pp")

st.markdown("---")

# === hero plot ===
# Plot in minutes-from-start space. timestamps are seconds (cadence_min*60 apart),
# so dividing by 60 gives minutes. t0 is the first history timestamp.

import pandas as pd
import numpy as np
import plotly.graph_objects as go

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

# default cursor: argmax |ml - naive| within inner 10-80% of window.
# this lands the demo on the most informative moment without scrubbing.
n_pred = len(ml_y)
lo_idx = int(n_pred * 0.10)
hi_idx = int(n_pred * 0.80)
diffs = np.abs(ml_y - naive_y)
default_cursor = lo_idx + int(np.argmax(diffs[lo_idx:hi_idx]))

# threshold default: 1.2x the max history value, but for cells where actual
# stays under 30% it's silly to draw the threshold at 36 — clamp to a useful range
default_threshold = float(min(max(hist_y) * 1.2, max(hist_y.max(), ml_y.max()) + 5))

# === sliders ===

s1, s2 = st.columns([3, 2])
cursor_idx = s1.slider(
    "Now (cursor scrubs through prediction window)",
    min_value=0, max_value=n_pred - 1, value=default_cursor,
    key=f"cursor_{st.session_state.container_id}_{st.session_state.horizon}",
)
threshold = s2.slider(
    "CPU threshold (%)",
    min_value=0.0,
    max_value=float(max(hist_y.max(), ml_y.max())) + 10,
    value=default_threshold,
    step=1.0,
)

# === build the figure ===

fig = go.Figure()

# CQR band (drawn first so it sits behind everything else)
if cqr_lo is not None:
    fig.add_trace(go.Scatter(
        x=target_x, y=cqr_hi,
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=target_x, y=cqr_lo,
        fill="tonexty", fillcolor="rgba(239, 68, 68, 0.15)",
        line=dict(width=0), name="Empirical ~80% interval",
        hovertemplate="lower: %{y:.1f}%<extra></extra>",
    ))

# actual CPU history (the "ground truth" line)
fig.add_trace(go.Scatter(
    x=hist_x, y=hist_y,
    mode="lines", name="Actual CPU",
    line=dict(color="#2563eb", width=2),
    hovertemplate="t=%{x:.0f}min  CPU=%{y:.1f}%<extra></extra>",
))

# naive (persistence) forecast
fig.add_trace(go.Scatter(
    x=target_x, y=naive_y,
    mode="lines", name="Naive forecast",
    line=dict(color="#9ca3af", width=1.5, dash="dot"),
    hovertemplate="naive=%{y:.1f}%<extra></extra>",
))

# ML forecast
fig.add_trace(go.Scatter(
    x=target_x, y=ml_y,
    mode="lines", name="ML forecast",
    line=dict(color="#dc2626", width=2, dash="dash"),
    hovertemplate="ml=%{y:.1f}%<extra></extra>",
))

# threshold line (horizontal)
fig.add_hline(
    y=threshold, line=dict(color="#6b7280", width=1, dash="dashdot"),
    annotation_text=f"threshold {threshold:.0f}%",
    annotation_position="top right",
)

# violation shading: where ML upper band exceeds threshold
if cqr_hi is not None:
    viol_mask = cqr_hi > threshold
    if viol_mask.any():
        # find contiguous regions and shade them
        in_region = False
        region_start = None
        for i in range(len(viol_mask)):
            if viol_mask[i] and not in_region:
                region_start = target_x[i]
                in_region = True
            elif not viol_mask[i] and in_region:
                fig.add_vrect(
                    x0=region_start, x1=target_x[i],
                    fillcolor="rgba(239, 68, 68, 0.08)",
                    layer="below", line_width=0,
                )
                in_region = False
        if in_region:
            fig.add_vrect(
                x0=region_start, x1=target_x[-1],
                fillcolor="rgba(239, 68, 68, 0.08)",
                layer="below", line_width=0,
            )

# cursor: vertical line at issue time, three markers at target time
cursor_issue_x = issue_x[cursor_idx]
cursor_target_x = target_x[cursor_idx]
fig.add_vline(
    x=cursor_issue_x, line=dict(color="#1f2937", width=1.5),
    annotation_text="now", annotation_position="top",
)

# the three forecast markers at cursor + horizon
fig.add_trace(go.Scatter(
    x=[cursor_target_x], y=[ml_y[cursor_idx]],
    mode="markers", name="ML @ cursor+h",
    marker=dict(color="#dc2626", size=12, symbol="circle", line=dict(color="white", width=2)),
    hovertemplate=f"ML at t+h: %{{y:.1f}}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[cursor_target_x], y=[naive_y[cursor_idx]],
    mode="markers", name="Naive @ cursor+h",
    marker=dict(color="#6b7280", size=10, symbol="circle", line=dict(color="white", width=2)),
    hovertemplate=f"Naive at t+h: %{{y:.1f}}%<extra></extra>",
))
fig.add_trace(go.Scatter(
    x=[cursor_target_x], y=[y_true[cursor_idx]],
    mode="markers", name="Actual @ cursor+h",
    marker=dict(color="#22c55e", size=14, symbol="star", line=dict(color="white", width=2)),
    hovertemplate=f"Actual at t+h: %{{y:.1f}}%<extra></extra>",
))

fig.update_layout(
    height=480,
    xaxis_title="minutes from test-window start",
    yaxis_title="CPU %",
    hovermode="x unified",
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
    ),
    margin=dict(l=40, r=20, t=60, b=40),
)

st.plotly_chart(fig, use_container_width=True)

# === cursor readout ===
# small numerical strip showing the cursor moment's actuals so the examiner
# doesnt have to mouse-hover on the markers
ml_at = ml_y[cursor_idx]
naive_at = naive_y[cursor_idx]
true_at = y_true[cursor_idx]
ml_err = abs(ml_at - true_at)
naive_err = abs(naive_at - true_at)

r1, r2, r3, r4 = st.columns(4)
r1.metric("ML forecast", f"{ml_at:.1f}%", delta=f"err {ml_err:.1f}%", delta_color="inverse")
r2.metric("Naive forecast", f"{naive_at:.1f}%", delta=f"err {naive_err:.1f}%", delta_color="inverse")
r3.metric("Actual outcome", f"{true_at:.1f}%")
r4.metric("ML beats naive by", f"{naive_err - ml_err:+.1f}%")

st.caption(
    f"BCF zone: **{meta['bcf_zone'].upper()}** — {meta['bcf_reason']}"
)

# debug strip — leaving for Piece 1; will remove in Piece 5
with st.expander("debug: cell metadata"):
    st.json(meta)