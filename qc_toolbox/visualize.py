from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)


class QCVisualizer:

    @staticmethod
    def plot_cbf_mosaic(
        cbf_map: np.ndarray,
        affine: np.ndarray,
        title: str = "CBF Map",
        n_slices: int = 9,
    ) -> Any:
        import matplotlib.pyplot as plt

        nz = cbf_map.shape[2]
        indices = np.linspace(int(nz * 0.15), int(nz * 0.85), n_slices, dtype=int)

        ncols = 3
        nrows = int(np.ceil(n_slices / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
        axes_flat = axes.ravel() if n_slices > 1 else [axes]

        vmin = float(np.percentile(cbf_map[cbf_map > 0], 2)) if np.any(cbf_map > 0) else 0
        vmax = float(np.percentile(cbf_map[cbf_map > 0], 98)) if np.any(cbf_map > 0) else 1

        for ax, idx in zip(axes_flat, indices):
            im = ax.imshow(
                np.rot90(cbf_map[:, :, idx]),
                cmap="jet", vmin=vmin, vmax=vmax,
            )
            ax.set_title(f"z={idx}", fontsize=9)
            ax.axis("off")

        for ax in axes_flat[n_slices:]:
            ax.axis("off")

        fig.colorbar(im, ax=axes_flat[:n_slices], shrink=0.6, label="CBF (ml/100g/min)")
        fig.suptitle(title, fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig

    @staticmethod
    def plot_motion_trace(
        fd_trace: np.ndarray,
        dvars_trace: np.ndarray,
        spike_indices: Optional[List[int]] = None,
    ) -> Any:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 5), sharex=True)

        t = np.arange(len(fd_trace))
        ax1.plot(t, fd_trace, color="#E74C3C", linewidth=1)
        ax1.axhline(0.5, color="gray", linestyle="--", linewidth=0.7)
        if spike_indices:
            ax1.scatter(
                spike_indices, fd_trace[spike_indices],
                color="black", marker="x", s=40, zorder=5,
            )
        ax1.set_ylabel("FD (mm)")
        ax1.set_title("Framewise Displacement")

        t2 = np.arange(len(dvars_trace))
        ax2.plot(t2, dvars_trace, color="#3498DB", linewidth=1)
        ax2.set_ylabel("DVARS")
        ax2.set_xlabel("Volume")
        ax2.set_title("DVARS")

        fig.tight_layout()
        return fig

    @staticmethod
    def plot_histogram(
        cbf_values: np.ndarray,
        title: str = "GM CBF Distribution",
        percentiles: Optional[Dict[int, float]] = None,
    ) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(
            cbf_values, bins=80, density=True,
            color="#4A90D9", alpha=0.7, edgecolor="white",
        )
        if percentiles:
            for p, v in sorted(percentiles.items()):
                ax.axvline(v, linestyle="--", linewidth=0.8, label=f"P{p}")
        ax.set_xlabel("CBF (ml/100g/min)")
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(fontsize=7)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_control_label_pattern(
        asl_timeseries: np.ndarray,
        aslcontext_df: pd.DataFrame,
        brain_mask: Optional[np.ndarray] = None,
    ) -> Any:

        import matplotlib.pyplot as plt

        if brain_mask is None:
            brain_mask = np.ones(asl_timeseries.shape[:3], dtype=bool)

        n_vols = asl_timeseries.shape[-1]
        means = [
            float(np.mean(asl_timeseries[..., t][brain_mask]))
            for t in range(n_vols)
        ]

        color_map = {"control": "#2ECC71", "label": "#E74C3C", "m0scan": "#3498DB"}
        colors = []
        for i in range(n_vols):
            if i < len(aslcontext_df):
                vtype = aslcontext_df.iloc[i]["volume_type"]
                colors.append(color_map.get(vtype, "gray"))
            else:
                colors.append("gray")

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(range(n_vols), means, color=colors, width=0.8)
        ax.set_xlabel("Volume index")
        ax.set_ylabel("Mean signal")
        ax.set_title("Control / Label Pattern")
        from matplotlib.patches import Patch

        handles = [
            Patch(color=c, label=l) for l, c in color_map.items()
        ]
        ax.legend(handles=handles, fontsize=8)
        fig.tight_layout()
        return fig


    @staticmethod
    def plot_qei_components(qei_result: Any) -> Any:
        import matplotlib.pyplot as plt

        labels = ["PSS", "1 − nGMCBF", "1 / (1 + DI/100)"]
        values = [
            qei_result.pss,
            1.0 - qei_result.ngm_cbf,
            1.0 / (1.0 + qei_result.di / 100.0),
        ]

        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
        values += values[:1]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"polar": True})
        ax.plot(angles, values, "o-", color="#4A90D9", linewidth=1.5)
        ax.fill(angles, values, alpha=0.15, color="#4A90D9")
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(f"QEI = {qei_result.qei_score:.3f}", fontsize=12, pad=20)
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_population_scatter(
        results_df: pd.DataFrame,
        x_metric: str,
        y_metric: str,
    ) -> Any:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 6))
        color_map = {"PASS": "#2ECC71", "WARN": "#F1C40F", "FAIL": "#E74C3C"}
        for flag, grp in results_df.groupby("overall_flag"):
            ax.scatter(
                grp[x_metric], grp[y_metric],
                c=color_map.get(str(flag), "gray"),
                label=flag, alpha=0.7, edgecolors="white", s=50,
            )
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(f"{y_metric} vs {x_metric}")
        ax.legend()
        fig.tight_layout()
        return fig

    @staticmethod
    def plot_site_comparison(
        results_df: pd.DataFrame,
        site_col: str = "site",
    ) -> Any:
        import matplotlib.pyplot as plt

        metrics = ["qei", "spatial_cov", "mean_gm_cbf", "temporal_snr", "mean_fd"]
        available = [m for m in metrics if m in results_df.columns]

        n = len(available)
        if n == 0:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No metrics available", ha="center")
            return fig

        fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))
        if n == 1:
            axes = [axes]

        sites = sorted(results_df[site_col].unique()) if site_col in results_df.columns else ["all"]

        for ax, metric in zip(axes, available):
            if site_col in results_df.columns:
                data = [
                    results_df.loc[results_df[site_col] == s, metric].dropna().values
                    for s in sites
                ]
                ax.boxplot(data, labels=sites)
            else:
                ax.boxplot(results_df[metric].dropna().values)
            ax.set_title(metric)
            ax.set_ylabel(metric)

        fig.suptitle("Site Comparison")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        return fig
