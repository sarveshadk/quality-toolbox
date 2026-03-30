from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)


@dataclass
class ControlLabelResult:

    pattern_valid: bool
    is_inverted: bool
    label_efficiency: float
    n_outlier_pairs: int
    outlier_pair_indices: List[int] = field(default_factory=list)
    mean_diff_signal: float = 0.0
    temporal_snr: float = 0.0
    pass_flag: bool = True


class ControlLabelMetric:

    def __init__(self, outlier_sd: float = 2.0) -> None:
        self.outlier_sd = outlier_sd

    def analyze(
        self,
        asl_timeseries: np.ndarray,
        aslcontext_df: pd.DataFrame,
        brain_mask: Optional[np.ndarray] = None,
    ) -> ControlLabelResult:
        try:
            return self._analyze(asl_timeseries, aslcontext_df, brain_mask)
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(
                f"Control-label analysis failed: {exc}"
            ) from exc

    def _analyze(
        self,
        ts: np.ndarray,
        ctx: pd.DataFrame,
        brain_mask: Optional[np.ndarray],
    ) -> ControlLabelResult:
        if ts.ndim != 4:
            raise QCComputationError(f"Expected 4-D, got {ts.ndim}-D.")

        n_vols = ts.shape[-1]
        if brain_mask is None:
            brain_mask = np.ones(ts.shape[:3], dtype=bool)

        control_idx = ctx.index[ctx["volume_type"] == "control"].tolist()
        label_idx = ctx.index[ctx["volume_type"] == "label"].tolist()

        control_idx = [i for i in control_idx if i < n_vols]
        label_idx = [i for i in label_idx if i < n_vols]

        if not control_idx or not label_idx:
            warnings.warn("No control or label volumes found.", stacklevel=2)
            return ControlLabelResult(
                pattern_valid=False, is_inverted=False, label_efficiency=0.0,
                n_outlier_pairs=0, pass_flag=False,
            )

        typed: List[tuple[int, str]] = []
        for i in control_idx:
            typed.append((i, "control"))
        for i in label_idx:
            typed.append((i, "label"))
        typed.sort(key=lambda x: x[0])

        pattern_valid = True
        for k in range(1, len(typed)):
            if typed[k][1] == typed[k - 1][1]:
                pattern_valid = False
                break

        mean_control = float(
            np.mean([np.mean(ts[..., i][brain_mask]) for i in control_idx])
        )
        mean_label = float(
            np.mean([np.mean(ts[..., i][brain_mask]) for i in label_idx])
        )
        is_inverted = mean_label > mean_control

        if abs(mean_control) > 1e-10:
            label_efficiency = abs(mean_control - mean_label) / abs(mean_control)
        else:
            label_efficiency = 0.0

        n_pairs = min(len(control_idx), len(label_idx))
        diff_maps: List[np.ndarray] = []
        for p in range(n_pairs):
            c = ts[..., control_idx[p]].astype(np.float64)
            l = ts[..., label_idx[p]].astype(np.float64)
            diff_maps.append(c - l)

        if n_pairs == 0:
            return ControlLabelResult(
                pattern_valid=pattern_valid, is_inverted=is_inverted,
                label_efficiency=label_efficiency, n_outlier_pairs=0,
                pass_flag=False,
            )

        diff_stack = np.stack(diff_maps, axis=-1)
        median_map = np.median(diff_stack, axis=-1)

        deviations = np.array([
            float(np.sqrt(np.mean((diff_maps[p][brain_mask] - median_map[brain_mask]) ** 2)))
            for p in range(n_pairs)
        ])

        if deviations.size > 1:
            dev_median = float(np.median(deviations))
            dev_std = float(np.std(deviations, ddof=1)) if deviations.size > 2 else 0.0
            threshold = dev_median + self.outlier_sd * dev_std if dev_std > 0 else np.inf
            outlier_pair_indices = [
                int(p) for p in range(n_pairs) if deviations[p] > threshold
            ]
        else:
            outlier_pair_indices = []

        mean_diff_per_pair = np.array([
            float(np.mean(dm[brain_mask])) for dm in diff_maps
        ])
        mean_diff_signal = float(np.mean(mean_diff_per_pair))
        std_diff = float(np.std(mean_diff_per_pair, ddof=1)) if n_pairs > 1 else 0.0
        temporal_snr = (
            abs(mean_diff_signal) / std_diff if std_diff > 1e-10 else 0.0
        )

        pass_flag = (
            pattern_valid
            and (not is_inverted)
            and len(outlier_pair_indices) <= max(1, n_pairs // 5)
        )

        return ControlLabelResult(
            pattern_valid=pattern_valid,
            is_inverted=is_inverted,
            label_efficiency=label_efficiency,
            n_outlier_pairs=len(outlier_pair_indices),
            outlier_pair_indices=outlier_pair_indices,
            mean_diff_signal=mean_diff_signal,
            temporal_snr=temporal_snr,
            pass_flag=pass_flag,
        )
