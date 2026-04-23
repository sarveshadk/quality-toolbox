import io
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image

_favicon_path = Path(__file__).parent / "assets" / "favicon.png"
_favicon = Image.open(_favicon_path) if _favicon_path.exists() else None

st.set_page_config(
    page_title="OSIPI ASL QC Toolbox",
    page_icon=_favicon,
    layout="wide",
    initial_sidebar_state="expanded",
)

TEAL = "#2CA58D"
SALMON = "#E07A5F"
NAVY = "#1B4F72"
NAVY_LIGHT = "#2471A3"
PURPLE = "#6C3483"
AMBER_DASH = "#D4A017"
MUTED = "#6B7280"
GRID = "#E8E8E8"
BG = "#FAFAFA"
WHITE = "#FFFFFF"
TEXT = "#1A1A2E"
TEXT_LIGHT = "#6B7280"
PASS_CLR = TEAL
WARN_CLR = "#D4A017"
FAIL_CLR = SALMON
FLAG_MAP = {"PASS": PASS_CLR, "WARN": WARN_CLR, "FAIL": FAIL_CLR}

FONT_MAIN = "Source Sans Pro, Helvetica Neue, Arial, sans-serif"
FONT_MONO = "Consolas, JetBrains Mono, monospace"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Source Sans 3', {FONT_MAIN};
        color: {TEXT};
    }}

    section[data-testid="stSidebar"] {{
        background: {WHITE};
        border-right: 1px solid #E0E0E0;
    }}
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown span {{
        color: #374151;
    }}

    /* Push content below the fixed Streamlit header/navbar */
    .block-container {{ padding-top: 2.5rem; }}

    /* Ensure the top toolbar doesn't overlay content */
    header[data-testid="stHeader"] {{
        background: {WHITE};
        border-bottom: 1px solid #E8E8E8;
    }}

    /* Hide the default deploy button */
    .stDeployButton {{ display: none; }}

    .page-header {{
        border-bottom: 2px solid {NAVY};
        padding-bottom: 8px;
        margin-bottom: 20px;
    }}
    .page-header h2 {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {NAVY};
        margin: 0;
        letter-spacing: -0.2px;
    }}
    .page-header .header-sub {{
        font-size: 0.72rem;
        color: {TEXT_LIGHT};
        margin-top: 2px;
    }}

    .summary-row {{
        display: flex;
        gap: 10px;
        margin-bottom: 18px;
    }}
    .summary-card {{
        flex: 1;
        background: {WHITE};
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 12px 14px;
        text-align: center;
    }}
    .summary-card .sc-label {{
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {TEXT_LIGHT};
        font-weight: 600;
    }}
    .summary-card .sc-value {{
        font-size: 1.5rem;
        font-weight: 700;
        color: {TEXT};
        font-family: {FONT_MONO};
    }}
    .summary-card.pass {{ border-left: 3px solid {TEAL}; }}
    .summary-card.warn {{ border-left: 3px solid {AMBER_DASH}; }}
    .summary-card.fail {{ border-left: 3px solid {SALMON}; }}

    .sec-title {{
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: {TEXT_LIGHT};
        margin: 20px 0 6px 0;
        padding-bottom: 4px;
        border-bottom: 1px solid #E8E8E8;
    }}

    .flag-tag {{
        display: inline-block;
        padding: 1px 8px;
        border-radius: 3px;
        font-size: 0.7rem;
        font-weight: 700;
        font-family: {FONT_MONO};
        letter-spacing: 0.4px;
    }}
    .flag-tag.PASS {{ background: #D1FAE5; color: #065F46; }}
    .flag-tag.WARN {{ background: #FEF3C7; color: #92400E; }}
    .flag-tag.FAIL {{ background: #FEE2E2; color: #991B1B; }}

    div[data-testid="stMetric"] {{
        background: {WHITE};
        border: 1px solid #E0E0E0;
        border-radius: 4px;
        padding: 8px 10px;
    }}
    div[data-testid="stMetric"] label {{
        font-size: 0.65rem !important;
        text-transform: uppercase;
        letter-spacing: 0.4px;
        color: {TEXT_LIGHT} !important;
    }}
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
        font-family: {FONT_MONO} !important;
        font-size: 1rem !important;
        color: {TEXT} !important;
    }}

    .ref-line {{
        font-size: 0.68rem;
        color: #9CA3AF;
        font-style: italic;
        margin-top: 12px;
        padding-top: 6px;
        border-top: 1px solid #F0F0F0;
    }}
</style>
""", unsafe_allow_html=True)


CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=WHITE,
    font=dict(family=FONT_MAIN, size=11, color=TEXT),
    margin=dict(t=40, b=45, l=50, r=20),
    xaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#CCCCCC", linewidth=1),
    yaxis=dict(gridcolor=GRID, zeroline=False, linecolor="#CCCCCC", linewidth=1),
)


def header(title, sub=""):
    st.markdown(f"""
    <div class="page-header">
        <h2>{title}</h2>
        <div class="header-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)


def fmt(val, d=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "-"
    if isinstance(val, float):
        return f"{val:.{d}f}"
    return str(val)


def demo_data() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 30
    subs = [f"sub-{i:03d}" for i in range(1, n + 1)]
    flags = rng.choice(["PASS", "WARN", "FAIL"], n, p=[0.6, 0.25, 0.15])
    return pd.DataFrame({
        "subject_id": subs,
        "session_id": ["ses-01"] * n,
        "overall_flag": flags,
        "qei": np.clip(rng.normal(0.75, 0.15, n), 0, 1),
        "pss": np.clip(rng.normal(0.7, 0.12, n), 0, 1),
        "di": np.abs(rng.normal(50, 30, n)),
        "ngm_cbf": np.clip(rng.normal(0.05, 0.04, n), 0, 1),
        "mean_gm_cbf": rng.normal(45, 12, n),
        "median_gm_cbf": rng.normal(43, 11, n),
        "std_gm_cbf": rng.normal(15, 5, n),
        "spatial_cov": rng.normal(40, 15, n),
        "temporal_snr": rng.normal(15, 6, n),
        "spatial_snr": rng.normal(8, 3, n),
        "mean_fd": np.abs(rng.normal(0.3, 0.2, n)),
        "max_fd": np.abs(rng.normal(0.8, 0.5, n)),
        "n_spikes": rng.integers(0, 5, n),
        "label_efficiency": np.clip(rng.normal(0.015, 0.005, n), 0, 0.05),
        "pattern_valid": rng.choice([True, False], n, p=[0.9, 0.1]),
        "m0_snr": rng.normal(50, 15, n),
        "gm_coverage": np.clip(rng.normal(0.3, 0.08, n), 0, 1),
        "processing_time": rng.uniform(5, 30, n),
        "error": [""] * n,
    })


def make_distribution_fig(df, metric_key, title, xlabel, threshold=None, thresh_label=None):
    pass_vals = df.loc[df["overall_flag"] == "PASS", metric_key].dropna()
    fail_vals = df.loc[df["overall_flag"].isin(["FAIL", "WARN"]), metric_key].dropna()

    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=pass_vals, name="Pass scans",
        marker_color=TEAL, opacity=0.6,
        nbinsx=20,
    ))
    fig.add_trace(go.Histogram(
        x=fail_vals, name="Fail/Warn scans",
        marker_color=SALMON, opacity=0.6,
        nbinsx=20,
    ))

    if threshold is not None:
        label = thresh_label or f"Threshold ({threshold})"
        fig.add_vline(
            x=threshold, line_dash="dash", line_color=PURPLE, line_width=2,
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(size=10, color=PURPLE),
        )

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"<b>{title}</b>", font=dict(size=13, color=NAVY)),
        xaxis_title=xlabel,
        yaxis_title="Count",
        barmode="overlay",
        height=280,
        legend=dict(
            orientation="h", y=-0.22, x=0.5, xanchor="center",
            font=dict(size=10), bgcolor="rgba(0,0,0,0)",
        ),
        bargap=0.05,
    )
    return fig


def make_fd_dvars_fig(df):
    rng = np.random.default_rng(99)
    n_vols = 80
    fd = np.abs(rng.normal(0.15, 0.08, n_vols))
    dvars = np.abs(rng.normal(0.9, 0.3, n_vols))

    spikes = [17, 43, 66]
    for s in spikes:
        fd[s] = rng.uniform(1.8, 2.8)
        dvars[s] = rng.uniform(2.5, 4.5)

    fd_thresh = 0.5
    dvars_thresh = 1.5 * np.median(dvars)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=["", ""])

    fig.add_trace(go.Scatter(
        x=list(range(n_vols)), y=fd,
        mode="lines", name="FD (mm)",
        line=dict(color=NAVY_LIGHT, width=1.5),
        fill="tozeroy", fillcolor="rgba(36,113,163,0.15)",
    ), row=1, col=1)

    fig.add_hline(y=fd_thresh, line_dash="dash", line_color=SALMON, line_width=1.5,
                  annotation_text=f"Threshold {fd_thresh} mm (Power 2012)",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color=SALMON),
                  row=1, col=1)

    for s in spikes:
        fig.add_annotation(
            x=s, y=fd[s], text=f"vol {s}<br>FD={fd[s]:.1f}mm",
            showarrow=True, arrowhead=2, arrowcolor="#8B0000",
            font=dict(size=8, color="#8B0000"),
            bgcolor="rgba(255,255,255,0.85)", borderpad=2,
            row=1, col=1,
        )

    fig.add_trace(go.Scatter(
        x=list(range(n_vols)), y=dvars,
        mode="lines", name="DVARS",
        line=dict(color=TEAL, width=1.5),
        fill="tozeroy", fillcolor="rgba(44,165,141,0.12)",
    ), row=2, col=1)

    fig.add_hline(y=dvars_thresh, line_dash="dash", line_color=SALMON, line_width=1.5,
                  annotation_text=f"Threshold 1.5x median = {dvars_thresh:.2f}",
                  annotation_position="top right",
                  annotation_font=dict(size=9, color=SALMON),
                  row=2, col=1)

    for s in spikes:
        fig.add_vrect(x0=s-0.8, x1=s+0.8,
                      fillcolor="rgba(224,122,95,0.15)", line_width=0,
                      row=1, col=1)
        fig.add_vrect(x0=s-0.8, x1=s+0.8,
                      fillcolor="rgba(224,122,95,0.15)", line_width=0,
                      row=2, col=1)

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="<b>Motion Tracking Module</b> - FD and DVARS Traces with Spike Detection",
                   font=dict(size=13, color=NAVY)),
        height=420,
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center", font=dict(size=10)),
    )
    fig.update_yaxes(title_text="Framewise Disp. (mm)", row=1, col=1,
                     gridcolor=GRID, linecolor="#CCC")
    fig.update_yaxes(title_text="DVARS", row=2, col=1,
                     gridcolor=GRID, linecolor="#CCC")
    fig.update_xaxes(title_text="Volume index", row=2, col=1,
                     gridcolor=GRID, linecolor="#CCC")
    return fig


def page_overview(df):
    header("OSIPI ASL QC Toolbox", "Population Summary - Quality Evaluation of ASL CBF Maps")

    if df.empty:
        st.markdown("**No data loaded.**")
        st.markdown("Upload a QC results CSV from the sidebar, or load the built-in demo dataset to explore the interface.")
        if st.button("Load demo dataset (30 synthetic subjects)"):
            st.session_state.results_df = demo_data()
            st.rerun()
        st.markdown('<p class="ref-line">Dolui S et al. (2024) Automated QEI for ASL. JMRI. doi:10.1002/jmri.29308</p>', unsafe_allow_html=True)
        return

    n = len(df)
    n_p = int((df["overall_flag"] == "PASS").sum())
    n_w = int((df["overall_flag"] == "WARN").sum())
    n_f = int((df["overall_flag"] == "FAIL").sum())
    rate = n_p / n * 100 if n > 0 else 0

    st.markdown(f"""
    <div class="summary-row">
        <div class="summary-card"><div class="sc-label">Subjects</div><div class="sc-value">{n}</div></div>
        <div class="summary-card"><div class="sc-label">Pass Rate</div><div class="sc-value">{rate:.0f}%</div></div>
        <div class="summary-card pass"><div class="sc-label">Pass</div><div class="sc-value">{n_p}</div></div>
        <div class="summary-card warn"><div class="sc-label">Warn</div><div class="sc-value">{n_w}</div></div>
        <div class="summary-card fail"><div class="sc-label">Fail</div><div class="sc-value">{n_f}</div></div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<div class="sec-title">QC Flag Distribution</div>', unsafe_allow_html=True)
        counts = df["overall_flag"].value_counts()
        fig = go.Figure(data=[go.Pie(
            labels=counts.index.tolist(),
            values=counts.values.tolist(),
            hole=0.5,
            marker=dict(colors=[FLAG_MAP.get(f, MUTED) for f in counts.index],
                        line=dict(color=WHITE, width=2)),
            textinfo="label+value",
            textfont=dict(size=11, family=FONT_MAIN),
        )])
        fig.update_layout(
            showlegend=False, height=230,
            margin=dict(t=5, b=5, l=5, r=5),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family=FONT_MAIN, color=TEXT),
        )
        st.plotly_chart(fig, use_container_width=True, key="pie_overview")

    with c2:
        st.markdown('<div class="sec-title">Subject Results Table</div>', unsafe_allow_html=True)
        show = ["subject_id", "overall_flag", "qei", "spatial_cov", "mean_fd", "temporal_snr", "mean_gm_cbf"]
        avail = [c for c in show if c in df.columns]
        st.dataframe(df[avail], height=250, use_container_width=True)

    st.markdown('<div class="sec-title">Key Metric Distributions (Pass vs. Fail/Warn)</div>', unsafe_allow_html=True)

    dist_specs = [
        ("qei", "QEI Score Distribution", "QEI Score", 0.70, "GMM threshold (0.70)"),
        ("spatial_cov", "Spatial CoV Distribution", "Spatial CoV (%)", 80.0, "Threshold (80%)"),
        ("mean_gm_cbf", "Mean GM CBF Distribution", "Mean GM CBF (ml/100g/min)", None, None),
        ("mean_fd", "Mean Framewise Displacement", "Mean FD (mm)", 0.5, "Threshold 0.5 mm"),
    ]

    cols = st.columns(2)
    for i, (key, title, xlabel, thresh, tlabel) in enumerate(dist_specs):
        if key not in df.columns:
            continue
        fig = make_distribution_fig(df, key, title, xlabel, thresh, tlabel)
        cols[i % 2].plotly_chart(fig, use_container_width=True, key=f"dist_{key}")

    st.markdown('<p class="ref-line">Thresholds: Dolui et al. JMRI 2024 (QEI), Clement et al. NMR Biomed 2019 (CoV, CBF), Power et al. NeuroImage 2012 (FD)</p>', unsafe_allow_html=True)


def page_subject(df):
    header("Subject Report", "Per-subject QC metric detail view")

    if df.empty:
        st.info("No data loaded. Upload CSV or load demo data from the Overview page.")
        return

    subjects = df["subject_id"].unique().tolist()
    selected = st.selectbox("Subject ID", subjects)
    row = df[df["subject_id"] == selected].iloc[0]

    flag = str(row.get("overall_flag", "FAIL"))
    st.markdown(f"""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
        <span style="font-size:1.05rem; font-weight:700; font-family:{FONT_MONO}; color:{NAVY};">{selected}</span>
        <span class="flag-tag {flag}">{flag}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-title">QEI Components</div>', unsafe_allow_html=True)
    c = st.columns(4)
    for col, (l, k) in zip(c, [("QEI", "qei"), ("PSS", "pss"), ("DI", "di"), ("nGM CBF", "ngm_cbf")]):
        col.metric(l, fmt(row.get(k)))

    st.markdown('<div class="sec-title">CBF Histogram Metrics</div>', unsafe_allow_html=True)
    c = st.columns(4)
    for col, (l, k) in zip(c, [("Mean GM CBF", "mean_gm_cbf"), ("Median GM CBF", "median_gm_cbf"),
                                ("Std GM CBF", "std_gm_cbf"), ("Spatial CoV", "spatial_cov")]):
        col.metric(l, fmt(row.get(k), 1))

    st.markdown('<div class="sec-title">Motion Parameters</div>', unsafe_allow_html=True)
    c = st.columns(4)
    for col, (l, k) in zip(c, [("Mean FD (mm)", "mean_fd"), ("Max FD (mm)", "max_fd"),
                                ("Spikes", "n_spikes"), ("tSNR", "temporal_snr")]):
        col.metric(l, fmt(row.get(k)))

    st.markdown('<div class="sec-title">Additional Checks</div>', unsafe_allow_html=True)
    c = st.columns(4)
    for col, (l, k) in zip(c, [("sSNR", "spatial_snr"), ("Label Efficiency", "label_efficiency"),
                                ("M0 SNR", "m0_snr"), ("GM Coverage", "gm_coverage")]):
        col.metric(l, fmt(row.get(k)))

    pat = row.get("pattern_valid")
    if pat is not None:
        txt = "Valid" if pat else "INVERTED"
        clr = TEAL if pat else SALMON
        st.markdown(f'Control-label pattern: **:{txt}**', unsafe_allow_html=False)

    st.markdown("---")
    st.markdown('<div class="sec-title">Motion Trace (Simulated)</div>', unsafe_allow_html=True)
    fig = make_fd_dvars_fig(df)
    st.plotly_chart(fig, use_container_width=True, key="fd_dvars_subject")

    st.markdown('<p class="ref-line">QEI formula: Dolui et al. JMRI 2024. FD: Power et al. NeuroImage 2012. DVARS: RMS temporal derivative.</p>', unsafe_allow_html=True)


def page_population(df):
    header("Population Analysis", "Cross-subject metric comparison and outlier detection")

    if df.empty:
        st.info("No data loaded.")
        return

    st.markdown('<div class="sec-title">Metric Correlation Scatter</div>', unsafe_allow_html=True)

    numeric = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ("processing_time", "n_spikes")]

    if len(numeric) >= 2:
        c1, c2 = st.columns(2)
        x_ax = c1.selectbox("X axis", numeric, index=numeric.index("qei") if "qei" in numeric else 0)
        y_ax = c2.selectbox("Y axis", numeric, index=numeric.index("spatial_cov") if "spatial_cov" in numeric else 1)

        fig = px.scatter(
            df, x=x_ax, y=y_ax, color="overall_flag",
            color_discrete_map=FLAG_MAP,
            hover_data=["subject_id"], opacity=0.85,
        )
        fig.update_traces(marker=dict(size=9, line=dict(width=0.8, color="white")))
        fig.update_layout(
            **CHART_LAYOUT,
            height=380,
            legend=dict(title=None, orientation="h", y=-0.15, x=0.5, xanchor="center", font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True, key="scatter_pop")

    st.markdown('<div class="sec-title">Box Plots by QC Flag</div>', unsafe_allow_html=True)
    box_m = [m for m in ["qei", "mean_gm_cbf", "spatial_cov", "temporal_snr", "mean_fd"] if m in df.columns]

    for row_start in range(0, len(box_m), 3):
        row_metrics = box_m[row_start:row_start+3]
        cols = st.columns(len(row_metrics))
        for ci, m in enumerate(row_metrics):
            fig = px.box(
                df, x="overall_flag", y=m, color="overall_flag",
                color_discrete_map=FLAG_MAP,
                category_orders={"overall_flag": ["PASS", "WARN", "FAIL"]},
            )
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor=WHITE,
                font=dict(family=FONT_MAIN, size=10, color=TEXT),
                title=dict(text=f"<b>{m.replace('_',' ').title()}</b>", font=dict(size=11, color=NAVY)),
                height=240, showlegend=False,
                margin=dict(t=36, b=30, l=45, r=10),
                xaxis=dict(title=None, gridcolor=GRID, linecolor="#CCC"),
                yaxis=dict(title=None, gridcolor=GRID, linecolor="#CCC"),
            )
            cols[ci].plotly_chart(fig, use_container_width=True, key=f"boxpop_{m}")


def page_thresholds(df):
    header("Threshold Profiles", "Population-specific GMM-learned threshold management")

    profile_dir = Path(__file__).parent / "qc_toolbox" / "thresholds" / "profiles"
    available = sorted(p.stem for p in profile_dir.glob("*.json")) if profile_dir.exists() else []

    st.markdown('<div class="sec-title">Profile Selection</div>', unsafe_allow_html=True)
    sel = st.selectbox("Profile", available or ["default"], label_visibility="collapsed")

    ppath = profile_dir / f"{sel}.json"
    if ppath.exists():
        with open(ppath, "r") as fh:
            pdata = json.load(fh)
    else:
        pdata = {"name": sel, "thresholds": {}}

    st.markdown('<div class="sec-title">Threshold Parameters</div>', unsafe_allow_html=True)

    tlist = list(pdata.get("thresholds", {}).items())
    updated: Dict[str, Any] = {}

    for i in range(0, len(tlist), 2):
        cols = st.columns(2)
        for j in range(2):
            if i + j >= len(tlist):
                break
            metric, info = tlist[i + j]
            cur = info.get("threshold", 0.0)
            d = info.get("direction", "above")
            sym = ">=" if d == "above" else "<="
            nv = cols[j].slider(
                f"{metric}  {sym}  threshold",
                min_value=0.0, max_value=max(200.0, cur * 3),
                value=float(cur), step=0.01, key=f"thr_{metric}",
            )
            updated[metric] = {**info, "threshold": nv}

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Save custom profile"):
            sd = {"name": sel, "population": pdata.get("population", "custom"), "thresholds": updated}
            op = profile_dir / f"{sel}_custom.json"
            op.parent.mkdir(parents=True, exist_ok=True)
            with open(op, "w") as fh:
                json.dump(sd, fh, indent=2)
            st.success(f"Saved: {op.name}")
    with c2:
        if not df.empty and st.button("Re-flag with current thresholds"):
            dc = df.copy()
            for m, info in updated.items():
                if m in dc.columns:
                    t = info["threshold"]
                    d = info.get("direction", "above")
                    dc[f"{m}_pass"] = dc[m] >= t if d == "above" else dc[m] <= t
            st.dataframe(dc, use_container_width=True)

    st.markdown('<p class="ref-line">GMM-learned thresholds adapt to population distributions. Minimum n>=20. Youden optimizer for AURA expert labels.</p>', unsafe_allow_html=True)


def page_export(df):
    header("Export Results", "Download QC results in CSV, JSON, or HTML format")

    if df.empty:
        st.info("No data loaded.")
        return

    st.markdown('<div class="sec-title">Download</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                          "qc_results.csv", "text/csv", use_container_width=True)
    with c2:
        st.download_button("Download JSON", df.to_json(orient="records", indent=2).encode("utf-8"),
                          "qc_results.json", "application/json", use_container_width=True)
    with c3:
        n = len(df)
        np_ = int((df["overall_flag"] == "PASS").sum())
        nw = int((df["overall_flag"] == "WARN").sum())
        nf = int((df["overall_flag"] == "FAIL").sum())
        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"><title>OSIPI ASL QC Report</title>
<style>
body {{ font-family: Source Sans Pro, sans-serif; margin: 40px; color: #1A1A2E; background: #FAFAFA; }}
h1 {{ color: #1B4F72; font-size: 1.3rem; border-bottom: 2px solid #1B4F72; padding-bottom: 6px; }}
.stats {{ display:flex; gap:14px; margin:14px 0; }}
.s {{ background:white; border:1px solid #E0E0E0; border-radius:4px; padding:10px 18px; text-align:center; }}
.s .l {{ font-size:0.65rem; text-transform:uppercase; letter-spacing:0.8px; color:#6B7280; }}
.s .v {{ font-size:1.3rem; font-weight:700; }}
table {{ border-collapse:collapse; width:100%; margin-top:18px; font-size:0.82rem; }}
th {{ background:#1B4F72; color:white; padding:7px 10px; text-align:left; }}
td {{ padding:5px 10px; border-bottom:1px solid #E8E8E8; }}
tr:hover {{ background:#F5F5F5; }}
.footer {{ margin-top:24px; font-size:0.7rem; color:#9CA3AF; border-top:1px solid #E8E8E8; padding-top:8px; }}
</style></head><body>
<h1>OSIPI ASL QC Toolbox - Report</h1>
<div class="stats">
<div class="s"><div class="l">Subjects</div><div class="v">{n}</div></div>
<div class="s"><div class="l">Pass</div><div class="v" style="color:{TEAL}">{np_}</div></div>
<div class="s"><div class="l">Warn</div><div class="v" style="color:{AMBER_DASH}">{nw}</div></div>
<div class="s"><div class="l">Fail</div><div class="v" style="color:{SALMON}">{nf}</div></div>
</div>
{df.to_html(index=False)}
<div class="footer">Generated by OSIPI ASL QC Toolbox. References: Dolui et al. JMRI 2024, Clement et al. NMR Biomed 2019.</div>
</body></html>"""
        st.download_button("Download HTML Report", html.encode("utf-8"),
                          "qc_report.html", "text/html", use_container_width=True)

    st.markdown('<div class="sec-title">Data Preview</div>', unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True)


def main():
    if "results_df" not in st.session_state:
        st.session_state.results_df = pd.DataFrame()

    with st.sidebar:
        st.markdown("""
        <div style="padding:12px 0; border-bottom:1px solid #E0E0E0; margin-bottom:10px;">
            <div style="font-size:0.9rem; font-weight:700; color:#1B4F72;">OSIPI QC Toolbox</div>
            <div style="font-size:0.65rem; color:#9CA3AF; margin-top:1px;">ASL MRI Quality Control v1.0</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Pages",
            ["Overview", "Subject Report", "Population Analysis", "Threshold Profiles", "Export"],
            label_visibility="collapsed",
        )

        st.markdown("---")
        up = st.file_uploader("Load results CSV", type=["csv"], key="csv_up")
        if up:
            st.session_state.results_df = pd.read_csv(up, comment="#")
            st.rerun()

        if not st.session_state.results_df.empty:
            st.caption(f"{len(st.session_state.results_df)} subjects loaded")

        if st.button("Clear data"):
            st.session_state.results_df = pd.DataFrame()
            st.rerun()

        st.markdown("---")
        st.markdown('<p style="font-size:0.6rem; color:#9CA3AF;">GSoC 2026 | OSIPI<br>Dolui, Mora | MIT License</p>', unsafe_allow_html=True)

    df = st.session_state.results_df

    if page == "Overview":
        page_overview(df)
    elif page == "Subject Report":
        page_subject(df)
    elif page == "Population Analysis":
        page_population(df)
    elif page == "Threshold Profiles":
        page_thresholds(df)
    elif page == "Export":
        page_export(df)


main()
