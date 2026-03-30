from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)

_PHYS_RANGE = (0.0, 120.0)
_MEAN_GM_RANGE = (10.0, 120.0)
_OUTSIDE_FRACTION_MAX = 0.05
_DEFAULT_PERCENTILES = (5, 10, 25, 50, 75, 90, 95)


@dataclass
class HistogramResult:

    mean: float
    median: float
    std: float
    vmin: float
    vmax: float
    percentiles: Dict[int, float] = field(default_factory=dict)
    kurtosis: float = 0.0
    skewness: float = 0.0
    fraction_outside: float = 0.0
    rms_diff: float = 0.0
    pass_flag: bool = True
    figure: Optional[object] = None


class HistogramMetric:

    def __init__(
        self,
        phys_range: tuple[float, float] = _PHYS_RANGE,
        mean_gm_range: tuple[float, float] = _MEAN_GM_RANGE,
        outside_fraction_max: float = _OUTSIDE_FRACTION_MAX,
    ) -> None:
        self.phys_range = phys_range
        self.mean_gm_range = mean_gm_range
        self.outside_fraction_max = outside_fraction_max

    def analyze(
        self,
        cbf_map: np.ndarray,
        gm_mask: np.ndarray,
        ps_cbf: Optional[np.ndarray] = None,
        generate_figure: bool = False,
    ) -> HistogramResult:
        try:
            return self._analyze(cbf_map, gm_mask, ps_cbf, generate_figure)
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"Histogram analysis failed: {exc}") from exc

    def _analyze(
        self,
        cbf_map: np.ndarray,
        gm_mask: np.ndarray,
        ps_cbf: Optional[np.ndarray],
        generate_figure: bool,
    ) -> HistogramResult:
        gm_vals = cbf_map[gm_mask].astype(np.float64)
        if gm_vals.size == 0:
            return HistogramResult(
                mean=0.0, median=0.0, std=0.0, vmin=0.0, vmax=0.0,
                pass_flag=False,
            )

        mean_v = float(np.mean(gm_vals))
        median_v = float(np.median(gm_vals))
        std_v = float(np.std(gm_vals, ddof=1))
        vmin = float(np.min(gm_vals))
        vmax = float(np.max(gm_vals))

        pcts = {
            int(p): float(np.percentile(gm_vals, p))
            for p in _DEFAULT_PERCENTILES
        }

        kurt = float(sp_stats.kurtosis(gm_vals, fisher=True))
        skew = float(sp_stats.skew(gm_vals))

        n_outside = int(np.sum(
            (gm_vals < self.phys_range[0]) | (gm_vals > self.phys_range[1])
        ))
        fraction_outside = n_outside / gm_vals.size

        rms_diff = 0.0
        if ps_cbf is not None:
            ps_vals = ps_cbf[gm_mask].astype(np.float64)
            if ps_vals.size == gm_vals.size:
                rms_diff = float(np.sqrt(np.mean((gm_vals - ps_vals) ** 2)))

        mean_ok = self.mean_gm_range[0] <= mean_v <= self.mean_gm_range[1]
        frac_ok = fraction_outside <= self.outside_fraction_max
        pass_flag = mean_ok and frac_ok

        fig = None
        if generate_figure:
            fig = self._make_figure(gm_vals, pcts)

        return HistogramResult(
            mean=mean_v,
            median=median_v,
            std=std_v,
            vmin=vmin,
            vmax=vmax,
            percentiles=pcts,
            kurtosis=kurt,
            skewness=skew,
            fraction_outside=fraction_outside,
            rms_diff=rms_diff,
            pass_flag=pass_flag,
            figure=fig,
        )

    @staticmethod
    def _make_figure(
        gm_vals: np.ndarray,
        pcts: Dict[int, float],
    ) -> object:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(gm_vals, bins=80, density=True, color="#4A90D9", alpha=0.7,
                edgecolor="white", linewidth=0.5)

        ax.axvspan(0, 120, alpha=0.08, color="green", label="Physiological range")

        colors = ["#E74C3C", "#E67E22", "#F1C40F", "#2ECC71",
                  "#F1C40F", "#E67E22", "#E74C3C"]
        for (p, v), c in zip(sorted(pcts.items()), colors):
            ax.axvline(v, color=c, linestyle="--", linewidth=0.8,
                       label=f"P{p}={v:.1f}")

        ax.set_xlabel("GM CBF (ml/100g/min)")
        ax.set_ylabel("Density")
        ax.set_title("Grey-Matter CBF Distribution")
        ax.legend(fontsize=7, loc="upper right")
        fig.tight_layout()
        return fig
