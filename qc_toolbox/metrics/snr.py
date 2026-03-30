from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)

_TEMPORAL_SNR_PASS: float = 10.0


@dataclass
class SNRResult:
    temporal_snr: float
    spatial_snr: float
    roi_snr: float
    temporal_sd_map: np.ndarray
    pass_flag: bool


class SNRMetric:
    def __init__(self, temporal_snr_threshold: float = _TEMPORAL_SNR_PASS) -> None:
        self.temporal_snr_threshold = temporal_snr_threshold

    def compute(
        self,
        asl_timeseries: np.ndarray,
        mean_cbf: np.ndarray,
        brain_mask: np.ndarray,
        gm_mask: Optional[np.ndarray] = None,
    ) -> SNRResult:
        try:
            tsnr = self.compute_temporal_snr(asl_timeseries, brain_mask)
            ssnr = self.compute_spatial_snr(mean_cbf, brain_mask)
            rsnr = self.compute_roi_snr(mean_cbf, gm_mask) if gm_mask is not None else 0.0
            sd_map = self.compute_temporal_sd_map(asl_timeseries)
            return SNRResult(
                temporal_snr=tsnr,
                spatial_snr=ssnr,
                roi_snr=rsnr,
                temporal_sd_map=sd_map,
                pass_flag=tsnr >= self.temporal_snr_threshold,
            )
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"SNR computation failed: {exc}") from exc

    @staticmethod
    def compute_temporal_snr(
        asl_timeseries: np.ndarray,
        brain_mask: np.ndarray,
    ) -> float:
        if asl_timeseries.ndim != 4 or asl_timeseries.shape[-1] < 2:
            return 0.0

        brain_ts = asl_timeseries[brain_mask].astype(np.float64)
        mean_sig = float(np.mean(brain_ts))
        temporal_std = float(np.mean(np.std(brain_ts, axis=1, ddof=1)))

        if temporal_std < 1e-10:
            return 0.0
        return mean_sig / temporal_std

    @staticmethod
    def compute_spatial_snr(
        mean_cbf: np.ndarray,
        brain_mask: np.ndarray,
    ) -> float:
        brain_vals = mean_cbf[brain_mask].astype(np.float64)
        bg_vals = mean_cbf[~brain_mask].astype(np.float64)

        if bg_vals.size == 0 or np.std(bg_vals) < 1e-10:
            return 0.0
        return float(np.mean(brain_vals)) / float(np.std(bg_vals))

    @staticmethod
    def compute_roi_snr(
        mean_cbf: np.ndarray,
        gm_mask: np.ndarray,
    ) -> float:
        vals = mean_cbf[gm_mask].astype(np.float64)
        if vals.size == 0 or np.std(vals) < 1e-10:
            return 0.0
        return float(np.mean(vals)) / float(np.std(vals, ddof=1))

    @staticmethod
    def compute_temporal_sd_map(asl_timeseries: np.ndarray) -> np.ndarray:
        if asl_timeseries.ndim != 4 or asl_timeseries.shape[-1] < 2:
            return np.zeros(asl_timeseries.shape[:3], dtype=np.float64)
        return np.std(asl_timeseries.astype(np.float64), axis=-1, ddof=1)
