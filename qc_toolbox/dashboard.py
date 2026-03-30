from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def run_dashboard(results_csv: Optional[str] = None) -> None:
    try:
        import streamlit as st
        import plotly.express as px
        import plotly.graph_objects as go
    except ImportError as exc:
        raise SystemExit(
            "Streamlit and plotly are required: pip install streamlit plotly"
        ) from exc

    st.set_page_config(page_title="QC Toolbox — ASL MRI", layout="wide")

    page = st.sidebar.radio(
        "Navigation",
        ["Run QC", "Subject Viewer", "Population Overview",
         "Threshold Editor", "Site Comparison", "Export"],
    )

    if "results_df" not in st.session_state:
        if results_csv and Path(results_csv).exists():
            st.session_state.results_df = pd.read_csv(
                results_csv, comment="#"
            )
        else:
            st.session_state.results_df = pd.DataFrame()

    df: pd.DataFrame = st.session_state.results_df

    if page == "Run QC":
        st.title("🧠 QC Toolbox — Run Pipeline")

        bids_dir = st.text_input("BIDS directory path")
        output_dir = st.text_input("Output directory", value="./qc_output")
        profile = st.selectbox("Threshold profile", ["default", "pediatric", "elderly"])
        n_workers = st.number_input("Workers", min_value=1, max_value=16, value=1)

        if st.button("▶ Run QC Pipeline"):
            if not bids_dir:
                st.error("Please enter a BIDS directory path.")
            else:
                with st.spinner("Running pipeline…"):
                    try:
                        from qc_toolbox.pipeline import QCPipeline

                        pipe = QCPipeline(
                            bids_dir=bids_dir,
                            output_dir=output_dir,
                            threshold_profile=profile,
                            n_workers=n_workers,
                            verbose=False,
                        )
                        result_df = pipe.run()
                        st.session_state.results_df = result_df
                        st.success(f"Completed — {len(result_df)} subjects processed.")
                    except Exception as exc:
                        st.error(f"Pipeline failed: {exc}")

        if not df.empty:
            st.subheader("Results")
            st.dataframe(df, use_container_width=True)

    elif page == "Subject Viewer":
        st.title("🔍 Subject Viewer")

        if df.empty:
            st.info("Load results first via 'Run QC' or provide a CSV.")
            return

        subjects = df["subject_id"].unique().tolist()
        selected = st.selectbox("Select subject", subjects)
        row = df[df["subject_id"] == selected].iloc[0]

        flag = row.get("overall_flag", "N/A")
        badge_color = {"PASS": "🟢", "WARN": "🟡", "FAIL": "🔴"}.get(flag, "⚪")
        st.markdown(f"### {badge_color} {selected} — **{flag}**")

        cols = st.columns(4)
        metric_pairs = [
            ("QEI", "qei"), ("Spatial CoV", "spatial_cov"),
            ("Mean FD", "mean_fd"), ("tSNR", "temporal_snr"),
        ]
        for col, (label, key) in zip(cols, metric_pairs):
            val = row.get(key, "N/A")
            col.metric(label, f"{val:.3f}" if isinstance(val, float) else str(val))

        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**All Metric Values**")
            for k, v in row.items():
                if k not in ("subject_id", "session_id", "error"):
                    st.text(f"  {k}: {v}")

        with col2:
            st.markdown("**Errors**")
            err = row.get("error", "")
            st.text(err if err else "None")

    elif page == "Population Overview":
        st.title("📊 Population Overview")

        if df.empty:
            st.info("No data loaded.")
            return

        counts = df["overall_flag"].value_counts()
        fig_pie = px.pie(
            names=counts.index, values=counts.values,
            title="QC Flag Distribution",
            color=counts.index,
            color_discrete_map={"PASS": "#2ECC71", "WARN": "#F1C40F", "FAIL": "#E74C3C"},
        )
        st.plotly_chart(fig_pie, use_container_width=True)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col_name in numeric_cols:
            if col_name in ("processing_time",):
                continue
            fig_hist = px.histogram(
                df, x=col_name, color="overall_flag",
                color_discrete_map={"PASS": "#2ECC71", "WARN": "#F1C40F", "FAIL": "#E74C3C"},
                title=col_name.replace("_", " ").title(),
                barmode="overlay", opacity=0.7,
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    elif page == "Threshold Editor":
        st.title("⚙️ Threshold Editor")

        profile_dir = Path(__file__).parent / "thresholds" / "profiles"
        available = sorted(p.stem for p in profile_dir.glob("*.json")) if profile_dir.exists() else []

        selected_profile = st.selectbox("Load profile", available or ["default"])

        profile_path = profile_dir / f"{selected_profile}.json"
        if profile_path.exists():
            with open(profile_path, "r") as fh:
                profile_data = json.load(fh)
        else:
            profile_data = {"name": selected_profile, "thresholds": {}}

        st.markdown("### Adjust Thresholds")
        updated_thresholds: Dict[str, Any] = {}
        for metric, info in profile_data.get("thresholds", {}).items():
            current = info.get("threshold", 0.0)
            direction = info.get("direction", "above")
            new_val = st.slider(
                f"{metric} ({direction})",
                min_value=0.0,
                max_value=max(200.0, current * 3),
                value=float(current),
                step=0.01,
                key=f"slider_{metric}",
            )
            updated_thresholds[metric] = {**info, "threshold": new_val}

        if st.button("💾 Save Profile"):
            save_data = {
                "name": selected_profile,
                "population": profile_data.get("population", "custom"),
                "thresholds": updated_thresholds,
            }
            out_path = profile_dir / f"{selected_profile}_custom.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w") as fh:
                json.dump(save_data, fh, indent=2)
            st.success(f"Saved to {out_path}")


        if not df.empty and st.button("🔄 Re-flag results"):
            df_copy = df.copy()
            for metric, info in updated_thresholds.items():
                if metric in df_copy.columns:
                    thresh = info["threshold"]
                    direction = info.get("direction", "above")
                    if direction == "above":
                        df_copy[f"{metric}_pass"] = df_copy[metric] >= thresh
                    else:
                        df_copy[f"{metric}_pass"] = df_copy[metric] <= thresh
            st.dataframe(df_copy, use_container_width=True)

    elif page == "Site Comparison":
        st.title("🏥 Site Comparison")

        if df.empty:
            st.info("No data loaded.")
            return

        site_col = st.text_input("Site column name", value="site")
        if site_col not in df.columns:
            st.warning(f"Column '{site_col}' not found. Available: {list(df.columns)}")
            return

        metrics = [c for c in df.select_dtypes(include=[np.number]).columns if c != "processing_time"]
        for m in metrics:
            fig = px.box(
                df, x=site_col, y=m, color=site_col,
                title=m.replace("_", " ").title(),
            )
            st.plotly_chart(fig, use_container_width=True)

    elif page == "Export":
        st.title("📥 Export")

        if df.empty:
            st.info("No data loaded.")
            return

        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button("📄 Download CSV", csv_bytes, "qc_results.csv", "text/csv")

        if st.button("🌐 Generate HTML Report"):
            with st.spinner("Generating…"):
                from qc_toolbox.report import QCReporter
                from qc_toolbox.pipeline import SubjectQCResult

                buf = io.StringIO()

                buf.write(f"<html><body><h1>QC Results</h1>{df.to_html()}</body></html>")
                html_bytes = buf.getvalue().encode("utf-8")
                st.download_button(
                    "Download HTML", html_bytes, "qc_report.html", "text/html"
                )

if __name__ == "__main__":
    run_dashboard()
