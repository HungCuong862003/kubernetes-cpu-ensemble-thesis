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

# Piece 1 placeholder chart: just the actual CPU history.
# Piece 2 will replace this with a Plotly hero plot (ML, naive, CQR band, cursor).
import pandas as pd
ts = cell["history"]["timestamps"]
t0 = ts[0]
hist = pd.DataFrame({
    "min_from_start": [(t - t0) / 60 for t in ts],
    "CPU %": cell["history"]["cpu_actual"],
}).set_index("min_from_start")
st.line_chart(hist, height=320, x_label="minutes from test-window start", y_label="CPU %")

st.caption(
    f"BCF zone: **{meta['bcf_zone'].upper()}** — {meta['bcf_reason']}"
)

# debug strip — leaving for Piece 1; will remove in Piece 5
with st.expander("debug: cell metadata"):
    st.json(meta)