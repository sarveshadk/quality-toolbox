from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from qc_toolbox import ReportError, __version__
from qc_toolbox.pipeline import SubjectQCResult

logger = logging.getLogger(__name__)


class QCReporter:
    @staticmethod
    def generate_csv(
        results: List[SubjectQCResult],
        output_path: str | Path,
    ) -> None:
        from qc_toolbox.pipeline import QCPipeline

        df = QCPipeline._results_to_dataframe(results)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# qc_toolbox v{__version__}\n")
        df.to_csv(path, index=False, mode="a")
        logger.info("CSV report saved to %s", path)

    @staticmethod
    def generate_summary_png(
        results: List[SubjectQCResult],
        output_path: str | Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from qc_toolbox.pipeline import QCPipeline

        df = QCPipeline._results_to_dataframe(results)
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))

        def _hist(ax, col, title, color="#4A90D9"):
            if col in df.columns:
                vals = df[col].dropna().values
                if len(vals) > 0:
                    ax.hist(vals, bins=20, color=color, edgecolor="white", alpha=0.8)
            ax.set_title(title, fontsize=11)
            ax.set_ylabel("Count")

        _hist(axes[0, 0], "qei", "QEI Distribution")
        _hist(axes[0, 1], "spatial_cov", "Spatial CoV (%)", "#E67E22")
        _hist(axes[0, 2], "mean_gm_cbf", "Mean GM CBF", "#2ECC71")
        _hist(axes[1, 0], "mean_fd", "Mean FD (mm)", "#E74C3C")
        _hist(axes[1, 1], "label_efficiency", "Label Efficiency", "#9B59B6")

        ax_pie = axes[1, 2]
        counts = df["overall_flag"].value_counts()
        labels = counts.index.tolist()
        color_map = {"PASS": "#2ECC71", "WARN": "#F1C40F", "FAIL": "#E74C3C"}
        colors = [color_map.get(l, "#95A5A6") for l in labels]
        ax_pie.pie(counts.values, labels=labels, colors=colors, autopct="%1.0f%%")
        ax_pie.set_title("Overall QC Flags")

        fig.suptitle(f"QC Toolbox v{__version__} — Population Summary", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        logger.info("Summary PNG saved to %s", path)

    @staticmethod
    def generate_html_report(
        results: List[SubjectQCResult],
        output_path: str | Path,
    ) -> None:
        from qc_toolbox.pipeline import QCPipeline

        df = QCPipeline._results_to_dataframe(results)
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        buf = io.BytesIO()
        QCReporter.generate_summary_png(results, buf)
        buf.seek(0)

        try:
            img_data = buf.read()
            img_b64 = base64.b64encode(img_data).decode("ascii")
        except Exception:
            img_b64 = ""

        def _row_color(flag: str) -> str:
            if flag == "PASS":
                return "#d4edda"
            if flag == "WARN":
                return "#fff3cd"
            return "#f8d7da"

        table_rows = []
        for _, row in df.iterrows():
            cells = "".join(
                f"<td>{row.get(c, '')}</td>" for c in df.columns
            )
            bg = _row_color(str(row.get("overall_flag", "FAIL")))
            table_rows.append(f'<tr style="background:{bg}">{cells}</tr>')

        headers = "".join(f"<th>{c}</th>" for c in df.columns)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>QC Toolbox Report</title>
<style>
body {{ font-family: 'Segoe UI', Tahoma, sans-serif; margin: 2em; background: #f5f6fa; }}
h1 {{ color: #2c3e50; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1em; font-size: 0.85em; }}
th, td {{ border: 1px solid #bdc3c7; padding: 6px 10px; text-align: left; }}
th {{ background: #2c3e50; color: white; cursor: pointer; }}
img {{ max-width: 100%; margin: 1em 0; }}
</style>
</head>
<body>
<h1>QC Toolbox v{__version__} — Report</h1>
<p>Subjects: {len(df)} | PASS: {(df['overall_flag']=='PASS').sum()} |
WARN: {(df['overall_flag']=='WARN').sum()} |
FAIL: {(df['overall_flag']=='FAIL').sum()}</p>
<img src="data:image/png;base64,{img_b64}" alt="Summary">
<table id="qc-table">
<thead><tr>{headers}</tr></thead>
<tbody>{''.join(table_rows)}</tbody>
</table>
<script>
document.querySelectorAll('#qc-table th').forEach((th, i) => {{
  th.addEventListener('click', () => {{
    const table = th.closest('table');
    const tbody = table.querySelector('tbody');
    const rows = Array.from(tbody.querySelectorAll('tr'));
    const dir = th.dataset.dir === 'asc' ? 'desc' : 'asc';
    th.dataset.dir = dir;
    rows.sort((a, b) => {{
      const av = a.children[i].textContent;
      const bv = b.children[i].textContent;
      const an = parseFloat(av), bn = parseFloat(bv);
      if (!isNaN(an) && !isNaN(bn)) return dir === 'asc' ? an - bn : bn - an;
      return dir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av);
    }});
    rows.forEach(r => tbody.appendChild(r));
  }});
}});
</script>
</body>
</html>"""

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(html, encoding="utf-8")
        logger.info("HTML report saved to %s", path)

    @staticmethod
    def generate_per_subject_pdf_report(
        subject_result: SubjectQCResult,
        output_path: str | Path,
    ) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with PdfPages(str(path)) as pdf:
            fig, axes = plt.subplots(3, 2, figsize=(11.7, 16.5))
            fig.suptitle(
                f"QC Report — {subject_result.subject_id} "
                f"({subject_result.overall_flag})",
                fontsize=14, fontweight="bold",
            )

            metrics_text: List[str] = []
            if subject_result.qei_result:
                q = subject_result.qei_result
                metrics_text.append(
                    f"QEI: {q.qei_score:.3f}  PSS: {q.pss:.3f}  "
                    f"DI: {q.di:.1f}  nGMCBF: {q.ngm_cbf:.3f}"
                )

            if subject_result.motion_result:
                m = subject_result.motion_result
                metrics_text.append(
                    f"Mean FD: {m.mean_fd:.3f} mm  Max FD: {m.max_fd:.3f} mm  "
                    f"Spikes: {m.n_spikes}"
                )

            if subject_result.snr_result:
                s = subject_result.snr_result
                metrics_text.append(
                    f"tSNR: {s.temporal_snr:.1f}  sSNR: {s.spatial_snr:.1f}  "
                    f"ROI SNR: {s.roi_snr:.1f}"
                )

            if subject_result.histogram_result:
                h = subject_result.histogram_result
                metrics_text.append(
                    f"Mean GM CBF: {h.mean:.1f}  Std: {h.std:.1f}  "
                    f"Kurtosis: {h.kurtosis:.2f}"
                )

            if subject_result.spatial_cov_result:
                sc = subject_result.spatial_cov_result
                metrics_text.append(f"Spatial CoV: {sc.spatial_cov:.1f}%")

            if subject_result.control_label_result:
                cl = subject_result.control_label_result
                metrics_text.append(
                    f"Label Eff: {cl.label_efficiency:.4f}  "
                    f"Pattern Valid: {cl.pattern_valid}  "
                    f"Inverted: {cl.is_inverted}"
                )

            ax = axes[0, 0]
            ax.axis("off")
            ax.text(
                0.05, 0.95, "\n".join(metrics_text),
                transform=ax.transAxes, fontsize=8, verticalalignment="top",
                fontfamily="monospace",
            )

            ax = axes[0, 1]
            if subject_result.motion_result and subject_result.motion_result.fd_trace.size > 1:
                ax.plot(subject_result.motion_result.fd_trace, color="#E74C3C")
                ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8)
                ax.set_title("Framewise Displacement")
                ax.set_ylabel("FD (mm)")
            else:
                ax.axis("off")

            ax = axes[1, 0]
            if subject_result.motion_result and subject_result.motion_result.dvars_trace.size > 1:
                ax.plot(subject_result.motion_result.dvars_trace, color="#3498DB")
                ax.set_title("DVARS")
                ax.set_ylabel("DVARS")
            else:
                ax.axis("off")

            ax = axes[1, 1]
            if subject_result.qei_result is not None:
                cbf_smooth = subject_result.qei_result.smoothed_cbf
                brain = cbf_smooth > 0
                if np.any(brain):
                    ax.hist(
                        cbf_smooth[brain].ravel(), bins=60,
                        color="#2ECC71", edgecolor="white", alpha=0.8,
                    )
                    ax.set_title("Smoothed CBF Histogram")
            else:
                ax.axis("off")

            ax = axes[2, 0]
            if subject_result.qei_result:
                q = subject_result.qei_result
                labels_r = ["PSS", "1-nGMCBF", "1/(1+DI/100)"]
                values = [
                    q.pss,
                    1 - q.ngm_cbf,
                    1.0 / (1.0 + q.di / 100.0),
                ]
                angles = np.linspace(0, 2 * np.pi, len(labels_r), endpoint=False).tolist()
                values += values[:1]
                angles += angles[:1]
                ax_r = fig.add_axes(
                    ax.get_position(), polar=True, frameon=False
                )
                ax.axis("off")
                ax_r.plot(angles, values, "o-", color="#4A90D9", linewidth=1.5)
                ax_r.fill(angles, values, alpha=0.15, color="#4A90D9")
                ax_r.set_xticks(angles[:-1])
                ax_r.set_xticklabels(labels_r, fontsize=8)
                ax_r.set_title("QEI Components", fontsize=10)
            else:
                ax.axis("off")

            axes[2, 1].axis("off")

            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            plt.close(fig)

        logger.info("PDF report saved to %s", path)
