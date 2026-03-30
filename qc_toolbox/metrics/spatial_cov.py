from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)

_SPATIAL_COV_PASS: float = 80.0


@dataclass
class SpatialCovResult:
    spatial_cov: float
    gm_wm_ratio: float
    vascular_artifact_index: float
    pass_flag: bool


class SpatialCovMetric:
    def __init__(self, cov_threshold: float = _SPATIAL_COV_PASS) -> None:
        self.cov_threshold = cov_threshold

    def compute(
        self,
        cbf_map: np.ndarray,
        gm_mask: np.ndarray,
        wm_mask: Optional[np.ndarray] = None,
    ) -> SpatialCovResult:
        try:
            return self._compute(cbf_map, gm_mask, wm_mask)
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"Spatial CoV failed: {exc}") from exc

    def _compute(
        self,
        cbf_map: np.ndarray,
        gm_mask: np.ndarray,
        wm_mask: Optional[np.ndarray],
    ) -> SpatialCovResult:
        gm_vals = cbf_map[gm_mask].astype(np.float64)

        if gm_vals.size == 0:
            return SpatialCovResult(
                spatial_cov=0.0, gm_wm_ratio=0.0,
                vascular_artifact_index=0.0, pass_flag=False,
            )

        mean_gm = float(np.mean(gm_vals))
        std_gm = float(np.std(gm_vals, ddof=1))

        if abs(mean_gm) < 1e-10:
            spatial_cov = float("inf")
        else:
            spatial_cov = 100.0 * std_gm / abs(mean_gm)

        gm_wm_ratio = 0.0
        if wm_mask is not None:
            wm_vals = cbf_map[wm_mask].astype(np.float64)
            if wm_vals.size > 0:
                mean_wm = float(np.mean(wm_vals))
                if abs(mean_wm) > 1e-10:
                    gm_wm_ratio = mean_gm / mean_wm

        if abs(mean_gm) > 1e-10:
            threshold = 3.0 * abs(mean_gm)
            n_hyper = int(np.sum(gm_vals > threshold))
            vascular_artifact_index = n_hyper / gm_vals.size
        else:
            vascular_artifact_index = 0.0

        return SpatialCovResult(
            spatial_cov=spatial_cov,
            gm_wm_ratio=gm_wm_ratio,
            vascular_artifact_index=vascular_artifact_index,
            pass_flag=spatial_cov <= self.cov_threshold,
        )
