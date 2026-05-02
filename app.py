# Kubernetes Workload Forecaster - Live Demo
# Loads precomputed forecast cells (models/demo/precomputed/*.json) and renders
# them. No model loading, no inference. See CLAUDE.md for build plan.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st


PRECOMPUTED_DIR = Path("models/demo/precomputed")
HPA_PATH = Path("reports/tables/hpa_simulation.csv")
SHAP_PATH = Path("reports/tables/shap_importance.csv")

ZONE_COLOR = {"green": "#22c55e", "yellow": "#eab308", "red": "#ef4444"}

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


@st.cache_data
def load_hpa():
    return pd.read_csv(HPA_PATH)


@st.cache_data
def load_shap():
    df = pd.read_csv(SHAP_PATH)
    return df.rename(columns={df.columns[0]: "feature"})


def cell_zone(index, container_id, horizon):
    # bcf_zone is per cell, not per container - A is red at 10min, green at 30/60/120
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


# === page setup ===

st.set_page_config(
    page_title="Kubernetes Workload Forecaster",
    layout="wide",
)

st.markdown("""
<style>
.block-container { padding-top: 2rem !important; max-width: 1180px; }
h1 { font-size: 1.6rem !important; margin-bottom: 0.5rem; }
[data-testid="stHorizontalBlock"] { gap: 1rem; }

.tile {
    border: 2px solid #d0d0d0;
    border-radius: 8px;
    padding: 10px;
    margin-bottom: 8px;
    background: #fafafa;
}
.tile-green  { border-color: #22c55e; background: #f0fdf4; }
.tile-yellow { border-color: #eab308; background: #fefce8; }
.tile-red    { border-color: #ef4444; background: #fef2f2; }
.tile-selected { box-shadow: 0 0 0 2px #1f2937; }
.tile-slot { font-weight: 700; font-size: 14px; }
.tile-label { font-size: 13px; color: #374151; }
.tile-id { font-size: 11px; color: #6b7280; font-family: monospace; }

div.stButton > button[kind="secondary"] { width: 100%; }

.info-card {
    border: 1px solid #e5e7eb;
    border-radius: 6px;
    padding: 12px;
    height: 175px;
    background: white;
}
.info-card-l { border-left-width: 4px; border-left-style: solid; }
.info-label { font-size: 11px; color: #6b7280; letter-spacing: 0.5px; }

@media (max-width: 768px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; }
}
</style>
""", unsafe_allow_html=True)

st.markdown("# Kubernetes Workload Forecaster")
st.caption("*Predicts when ML helps, and tells you when it doesn't.*")


# === sidebar ===

index = load_index()
demo = load_demo_containers()
horizons = index["horizons"]

if "container_id" not in st.session_state:
    st.session_state.container_id = "c_41072"  # slot A, hero shot
if "horizon" not in st.session_state:
    st.session_state.horizon = "120min"
if "view" not in st.session_state:
    st.session_state.view = "Single container"

with st.sidebar:
    st.markdown("### CONTAINER")
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
            f'<div class="tile-id">{cid}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("Select" if not is_sel else "Selected", key=f"sel_{slot}"):
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

    st.markdown("### VIEW")
    st.session_state.view = st.radio(
        "view_mode",
        ["Single container", "Compare 4"],
        index=["Single container", "Compare 4"].index(st.session_state.view),
        label_visibility="collapsed",
        key="view_radio",
    )


# === plot helpers ===

def build_hero(cell, cursor_idx, threshold):
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

    # CQR band first so it sits behind everything else
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

    fig.add_trace(go.Scatter(
        x=hist_x, y=hist_y, mode="lines", name="Actual CPU",
        line=dict(color="#2563eb", width=2),
        hovertemplate="t=%{x:.0f}min  CPU=%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=target_x, y=naive_y, mode="lines", name="Naive forecast",
        line=dict(color="#9ca3af", width=1.5, dash="dot"),
        hovertemplate="naive=%{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=target_x, y=ml_y, mode="lines", name="ML forecast",
        line=dict(color="#dc2626", width=2, dash="dash"),
        hovertemplate="ml=%{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(
        y=threshold, line=dict(color="#6b7280", width=1, dash="dashdot"),
        annotation_text=f"threshold {threshold:.0f}%",
        annotation_position="top right",
    )

    # violation shading where CQR upper exceeds threshold
    if cqr_hi is not None:
        viol_mask = cqr_hi > threshold
        if viol_mask.any():
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

    cursor_issue_x = issue_x[cursor_idx]
    cursor_target_x = target_x[cursor_idx]
    fig.add_vline(
        x=cursor_issue_x, line=dict(color="#1f2937", width=1.5),
        annotation_text="now", annotation_position="top",
    )

    fig.add_trace(go.Scatter(
        x=[cursor_target_x], y=[ml_y[cursor_idx]],
        mode="markers", name="ML @ cursor+h",
        marker=dict(color="#dc2626", size=12, symbol="circle",
                    line=dict(color="white", width=2)),
        hovertemplate="ML at t+h: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[cursor_target_x], y=[naive_y[cursor_idx]],
        mode="markers", name="Naive @ cursor+h",
        marker=dict(color="#6b7280", size=10, symbol="circle",
                    line=dict(color="white", width=2)),
        hovertemplate="Naive at t+h: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[cursor_target_x], y=[y_true[cursor_idx]],
        mode="markers", name="Actual @ cursor+h",
        marker=dict(color="#22c55e", size=14, symbol="star",
                    line=dict(color="white", width=2)),
        hovertemplate="Actual at t+h: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        height=480,
        xaxis_title="minutes from test-window start",
        yaxis_title="CPU %",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig, ml_y, naive_y, y_true


def compact_tile(cell):
    p = cell["predictions"]
    targets = p["target_timestamps"]
    t0 = targets[0]
    x = [(t - t0) / 60 for t in targets]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=p["y_true"], mode="lines",
        line=dict(color="#1e40af", width=1.5), name="Actual",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p["naive_pred"], mode="lines",
        line=dict(color="#9ca3af", width=1, dash="dot"), name="Naive",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=p["ml_pred"], mode="lines",
        line=dict(color="#dc2626", width=1.2, dash="dash"), name="ML",
    ))
    fig.update_layout(
        height=180,
        margin=dict(l=8, r=8, t=8, b=20),
        showlegend=False,
        xaxis=dict(showticklabels=False, showgrid=False, title=None),
        yaxis=dict(tickfont=dict(size=9), gridcolor="#f1f5f9", title=None),
        plot_bgcolor="white",
    )
    return fig


# === Piece 3: info cards ===

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
    sign = "+" if refinement >= 0 else ""

    horizon_min = m["h_steps"] * m["cadence_min"]
    acf_pass = m["acf_24h"] >= 0.2
    hz_pass = horizon_min >= 30
    acf_op = "&ge;" if acf_pass else "&lt;"
    hz_op = "&ge;" if hz_pass else "&lt;"

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"<div class='info-card info-card-l' style='border-left-color:{color}'>"
            f"<div class='info-label'>BCF BOUNDARY CONDITION</div>"
            f"<div style='font-size:22px;font-weight:600;color:{color};margin-top:4px'>"
            f"{zone.upper()}</div>"
            f"<div style='font-size:13px;margin-top:8px;color:#374151'>"
            f"ACF@24h = <b>{m['acf_24h']:.2f}</b> "
            f"({acf_op} 0.2)<br>"
            f"horizon = <b>{m['horizon']}</b> "
            f"({hz_op} 30min)"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            f"<div class='info-card'>"
            f"<div class='info-label'>FORECAST DECOMPOSITION</div>"
            f"<div style='font-family:monospace;font-size:13px;margin-top:8px;line-height:1.7'>"
            f"Naive baseline: <b>{naive_v:6.2f}</b><br>"
            f"ML refinement: <b>{sign}{refinement:5.2f}</b><br>"
            f"<span style='color:#dc2626'>ML forecast:&nbsp;&nbsp;&nbsp;<b>{ml_v:6.2f}</b></span><br>"
            f"<span style='color:#1e40af'>Actual:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<b>{actual:6.2f}</b></span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            f"<div class='info-card info-card-l' style='border-left-color:{color}'>"
            f"<div class='info-label'>RECOMMENDED ACTION</div>"
            f"<div style='font-size:16px;font-weight:600;margin-top:4px;color:#111827'>"
            f"{verdict}</div>"
            f"<div style='font-size:12px;color:#4b5563;margin-top:8px;line-height:1.5'>"
            f"{why}</div></div>",
            unsafe_allow_html=True,
        )


# === Piece 4: compare grid + HPA panel ===

def render_compare_grid(hz):
    by_slot = {c["slot"]: c for c in demo["selected"]}
    rows = [["A", "B"], ["C", "D"]]
    for row in rows:
        cols = st.columns(2)
        for col, slot in zip(cols, row):
            with col:
                c = by_slot[slot]
                cid = c["container_id"]
                cell = load_cell(cid, hz)
                m = cell["metadata"]
                zone = m["bcf_zone"].lower()
                border = ZONE_COLOR.get(zone, "#888")
                delta = m["delta_pp"]
                arrow = "+" if delta >= 0 else ""
                st.markdown(
                    f"<div style='border-left:4px solid {border};padding:4px 0 4px 10px;"
                    f"margin-bottom:2px;'>"
                    f"<b>Slot {slot} - {m['label']}</b><br>"
                    f"<span style='font-size:12px;color:#6b7280'>"
                    f"&Delta; {arrow}{delta:.1f}pp &middot; "
                    f"ACF@24h={m['acf_24h']:.2f} &middot; "
                    f"BCF {zone}</span></div>",
                    unsafe_allow_html=True,
                )
                st.plotly_chart(
                    compact_tile(cell),
                    use_container_width=True,
                    key=f"tile_{cid}_{hz}",
                )


def render_hpa_panel(hz):
    df = load_hpa()
    df = df[df["Horizon"] == hz].copy()

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
    sub["label"] = sub.apply(
        lambda r: "ML-Proactive" if r["Strategy"] == "ML-Proactive" else r["variant"],
        axis=1,
    )
    order = ["ML-Proactive", "lag=1", "lag=2", "lag=3", "lag=4"]
    have = [x for x in order if x in sub["label"].values]
    sub = sub.set_index("label").loc[have].reset_index()

    colors = ["#16a34a" if x == "ML-Proactive" else "#94a3b8" for x in sub["label"]]

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=("SLO violations (%)", "Resource waste (%)", "Violation severity"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(go.Bar(
        x=sub["label"], y=sub["violation_rate"] * 100, marker_color=colors,
        text=[f"{v*100:.1f}" for v in sub["violation_rate"]],
        textposition="outside", showlegend=False,
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        x=sub["label"], y=sub["waste_rate"] * 100, marker_color=colors,
        text=[f"{v*100:.1f}" for v in sub["waste_rate"]],
        textposition="outside", showlegend=False,
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=sub["label"], y=sub["viol_severity"], marker_color=colors,
        text=[f"{v:.3f}" for v in sub["viol_severity"]],
        textposition="outside", showlegend=False,
    ), row=1, col=3)

    fig.update_layout(
        height=340,
        margin=dict(l=20, r=20, t=50, b=40),
        plot_bgcolor="white",
    )
    fig.update_yaxes(gridcolor="#f1f5f9")
    st.plotly_chart(fig, use_container_width=True, key=f"hpa_{hz}")

    ml_row = sub[sub["label"] == "ML-Proactive"].iloc[0]
    react_rows = sub[sub["label"] != "ML-Proactive"]
    dv = (react_rows["violation_rate"].mean() - ml_row["violation_rate"]) * 100
    dw = (ml_row["waste_rate"] - react_rows["waste_rate"].mean()) * 100
    st.caption(
        f"At safety margin {sm}, ML-Proactive lowers violations by {dv:.1f}pp "
        f"and uses {dw:.1f}pp more capacity vs the average reactive lag "
        f"(lower is better for both)."
    )


# === Piece 5: drill-down ===

def render_drilldown(cell):
    m = cell["metadata"]
    cid = m["container_id"]
    hz = m["horizon"]

    with st.expander("Drill-down: per-horizon R-squared and SHAP feature importance",
                     expanded=False):
        # R^2 across horizons for this container
        all_hz = sorted(horizons, key=lambda h: int(h.replace("min", "")))
        ml_r2s, naive_r2s = [], []
        for h in all_hz:
            cc = load_cell(cid, h)
            ml_r2s.append(cc["metadata"]["et_r2"])
            naive_r2s.append(cc["metadata"]["naive_r2"])

        st.markdown(f"**Per-horizon R-squared for {cid}**")
        fig_r2 = go.Figure()
        fig_r2.add_trace(go.Bar(
            x=all_hz, y=naive_r2s, name="Naive", marker_color="#94a3b8",
            text=[f"{v:.3f}" for v in naive_r2s], textposition="outside",
        ))
        fig_r2.add_trace(go.Bar(
            x=all_hz, y=ml_r2s, name="ML (ExtraTrees)", marker_color="#dc2626",
            text=[f"{v:.3f}" for v in ml_r2s], textposition="outside",
        ))
        fig_r2.update_layout(
            height=280, barmode="group", plot_bgcolor="white",
            margin=dict(l=20, r=20, t=10, b=20),
            legend=dict(orientation="h", y=-0.2),
            yaxis=dict(title="R-squared", gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig_r2, use_container_width=True, key=f"drill_r2_{cid}")

        # SHAP top features at active horizon
        st.markdown(f"**Top 10 SHAP features at {hz} (Alibaba ensemble, global)**")
        shap_df = load_shap()
        top = shap_df.nlargest(10, hz)[["feature", hz]].copy()
        top["group"] = top["feature"].apply(feature_group)
        group_color = {
            "hour-of-day": "#3b82f6", "day/business": "#60a5fa",
            "rolling": "#f59e0b", "lag": "#84cc16",
            "cross-resource": "#a855f7", "cluster": "#64748b", "other": "#9ca3af",
        }
        colors = [group_color[g] for g in top["group"]]

        fig_shap = go.Figure(go.Bar(
            x=top[hz][::-1], y=top["feature"][::-1],
            orientation="h", marker_color=colors[::-1],
            text=[f"{v:.3f}" for v in top[hz][::-1]],
            textposition="outside",
        ))
        fig_shap.update_layout(
            height=340, plot_bgcolor="white",
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(title="mean |SHAP|", gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig_shap, use_container_width=True,
                        key=f"drill_shap_{cid}_{hz}")

        st.caption(
            "SHAP is global across the Alibaba ensemble - same values for all "
            "four demo containers. Feature group (color) tells the story: "
            "hour-of-day dominates at 120min, rolling stats matter most at 10min."
        )


# === main ===

active_cid = st.session_state.container_id
active_hz = st.session_state.horizon
view = st.session_state.view

cell = load_cell(active_cid, active_hz)
meta = cell["metadata"]


if view == "Single container":
    # header strip with the picked cell's headline numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Container", meta["container_id"])
    c2.metric("Horizon", meta["horizon"])
    c3.metric("ACF@24h", f"{meta['acf_24h']:.2f}")
    c4.metric(
        "ML R^2 vs Naive R^2",
        f"{meta['et_r2']:.3f} vs {meta['naive_r2']:.3f}",
        delta=f"{meta['delta_pp']:+.2f}pp",
    )
    st.markdown("---")

    pred = cell["predictions"]
    ml_y = np.array(pred["ml_pred"])
    naive_y = np.array(pred["naive_pred"])
    y_true = np.array(pred["y_true"])
    hist_y = np.array(cell["history"]["cpu_actual"])

    # default cursor: argmax |ml - naive| within inner 10-80% of window.
    # this lands the demo on the most informative moment without scrubbing.
    n_pred = len(ml_y)
    lo_idx = int(n_pred * 0.10)
    hi_idx = int(n_pred * 0.80)
    diffs = np.abs(ml_y - naive_y)
    default_cursor = lo_idx + int(np.argmax(diffs[lo_idx:hi_idx]))

    # threshold default: 1.2x max history but clamp to a useful range so
    # low-utilisation cells don't get a meaninglessly small threshold line
    default_threshold = float(min(
        hist_y.max() * 1.2,
        max(hist_y.max(), ml_y.max()) + 5,
    ))

    s1, s2 = st.columns([3, 2])
    cursor_idx = s1.slider(
        "Now (cursor scrubs through prediction window)",
        min_value=0, max_value=n_pred - 1, value=default_cursor,
        key=f"cursor_{active_cid}_{active_hz}",
    )
    threshold = s2.slider(
        "CPU threshold (%)",
        min_value=0.0,
        max_value=float(max(hist_y.max(), ml_y.max())) + 10,
        value=default_threshold, step=1.0,
        key=f"thr_{active_cid}_{active_hz}",
    )

    fig, ml_y, naive_y, y_true = build_hero(cell, cursor_idx, threshold)
    st.plotly_chart(fig, use_container_width=True,
                    key=f"hero_{active_cid}_{active_hz}")

    # cursor readout - small numerical strip showing the cursor moment
    ml_at = ml_y[cursor_idx]
    naive_at = naive_y[cursor_idx]
    true_at = y_true[cursor_idx]
    ml_err = abs(ml_at - true_at)
    naive_err = abs(naive_at - true_at)

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("ML forecast", f"{ml_at:.1f}%",
              delta=f"err {ml_err:.1f}%", delta_color="inverse")
    r2.metric("Naive forecast", f"{naive_at:.1f}%",
              delta=f"err {naive_err:.1f}%", delta_color="inverse")
    r3.metric("Actual outcome", f"{true_at:.1f}%")
    r4.metric("ML beats naive by", f"{naive_err - ml_err:+.1f}%")

    st.caption(f"BCF zone: **{meta['bcf_zone'].upper()}** - {meta['bcf_reason']}")

    st.markdown("---")
    render_info_cards(cell, cursor_idx)

    render_drilldown(cell)

else:
    st.subheader(f"All 4 demo containers at {active_hz}")
    render_compare_grid(active_hz)


# HPA always at the bottom regardless of view mode
st.markdown("---")
st.subheader(f"HPA simulation at {active_hz}: ML-Proactive vs reactive lags")
render_hpa_panel(active_hz)


# footer
st.markdown(
    "<div style='text-align:center;color:#9ca3af;font-size:11px;"
    "margin-top:32px;padding-top:16px;border-top:1px solid #f1f5f9;'>"
    "Hybrid Ensemble Learning for Proactive Resource Prediction in Kubernetes &middot; "
    "Bachelor's thesis, IU VNU-HCM 2026 &middot; "
    "Predictions precomputed, no live inference."
    "</div>",
    unsafe_allow_html=True,
)