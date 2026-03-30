from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)

_FD_PASS_THRESHOLD: float = 0.5
_DVARS_SPIKE_FACTOR: float = 1.5
_HEAD_RADIUS_MM: float = 50.0


@dataclass
class MotionResult:

    mean_fd: float
    max_fd: float
    n_spikes: int
    spike_indices: List[int] = field(default_factory=list)
    fd_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    dvars_trace: np.ndarray = field(default_factory=lambda: np.array([]))
    pass_flag: bool = True


class MotionMetric:

    def __init__(
        self,
        fd_threshold: float = _FD_PASS_THRESHOLD,
        dvars_spike_factor: float = _DVARS_SPIKE_FACTOR,
        reference_vol_idx: int = 0,
    ) -> None:
        self.fd_threshold = fd_threshold
        self.dvars_spike_factor = dvars_spike_factor
        self.reference_vol_idx = reference_vol_idx

    def compute_framewise_displacement(
        self,
        asl_timeseries: np.ndarray,
        brain_mask: Optional[np.ndarray] = None,
    ) -> MotionResult:
        try:
            return self._compute(asl_timeseries, brain_mask)
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"Motion metric failed: {exc}") from exc

    def _compute(
        self,
        ts: np.ndarray,
        brain_mask: Optional[np.ndarray],
    ) -> MotionResult:
        if ts.ndim != 4:
            raise QCComputationError(
                f"Expected 4-D time-series, got {ts.ndim}-D."
            )

        n_vols = ts.shape[-1]
        if n_vols < 2:
            return MotionResult(
                mean_fd=0.0, max_fd=0.0, n_spikes=0,
                spike_indices=[], fd_trace=np.zeros(1),
                dvars_trace=np.zeros(1), pass_flag=True,
            )

        if brain_mask is None:
            brain_mask = np.ones(ts.shape[:3], dtype=bool)

        fd_trace = self._try_rigid_body_fd(ts, brain_mask)
        dvars_trace = self._compute_dvars(ts, brain_mask)

        median_dvars = float(np.median(dvars_trace)) if dvars_trace.size > 0 else 0.0
        dvars_thresh = self.dvars_spike_factor * median_dvars if median_dvars > 0 else np.inf

        spike_indices: List[int] = []
        for t in range(len(fd_trace)):
            fd_spike = fd_trace[t] > self.fd_threshold
            dv_spike = (
                dvars_trace[t] > dvars_thresh
                if t < len(dvars_trace)
                else False
            )
            if fd_spike or dv_spike:
                spike_indices.append(t)

        mean_fd = float(np.mean(fd_trace))
        max_fd = float(np.max(fd_trace))

        return MotionResult(
            mean_fd=mean_fd,
            max_fd=max_fd,
            n_spikes=len(spike_indices),
            spike_indices=spike_indices,
            fd_trace=fd_trace,
            dvars_trace=dvars_trace,
            pass_flag=mean_fd <= self.fd_threshold,
        )

    def _try_rigid_body_fd(
        self, ts: np.ndarray, brain_mask: np.ndarray
    ) -> np.ndarray:
        try:
            return self._rigid_body_dipy(ts, brain_mask)
        except Exception:
            logger.info(
                "Rigid-body registration unavailable — using mean-signal proxy."
            )
            return self._proxy_fd(ts, brain_mask)

    def _rigid_body_dipy(
        self, ts: np.ndarray, brain_mask: np.ndarray
    ) -> np.ndarray:
        from dipy.align.imaffine import (
            AffineRegistration,
            MutualInformationMetric,
            transform_centers_of_mass,
        )
        from dipy.align.transforms import RigidTransform3D

        ref_vol = ts[..., self.reference_vol_idx].astype(np.float64)
        n_vols = ts.shape[-1]

        metric = MutualInformationMetric(nbins=32, sampling_proportion=0.3)
        affreg = AffineRegistration(metric=metric, level_iters=[100, 50])
        rigid = RigidTransform3D()

        identity = np.eye(4)
        prev_params = np.zeros(6)
        fd = np.zeros(n_vols)

        for t in range(n_vols):
            if t == self.reference_vol_idx:
                continue
            moving = ts[..., t].astype(np.float64)
            try:
                com = transform_centers_of_mass(ref_vol, identity, moving, identity)
                result = affreg.optimize(
                    ref_vol, moving, rigid, None,
                    identity, identity,
                    starting_affine=com.affine,
                )
                mat = result.affine
                tx, ty, tz = mat[0, 3], mat[1, 3], mat[2, 3]
                rx = np.arctan2(mat[2, 1], mat[2, 2])
                ry = np.arctan2(-mat[2, 0], np.sqrt(mat[2, 1] ** 2 + mat[2, 2] ** 2))
                rz = np.arctan2(mat[1, 0], mat[0, 0])
                params = np.array([tx, ty, tz, rx, ry, rz])
            except Exception:
                params = prev_params.copy()

            delta = params - prev_params
            fd[t] = (
                np.sum(np.abs(delta[:3]))
                + _HEAD_RADIUS_MM * np.sum(np.abs(delta[3:]))
            )
            prev_params = params

        return fd

    def _proxy_fd(self, ts: np.ndarray, brain_mask: np.ndarray) -> np.ndarray:
        n_vols = ts.shape[-1]
        means = np.array(
            [float(np.mean(ts[..., t][brain_mask])) for t in range(n_vols)]
        )
        fd = np.zeros(n_vols)
        fd[1:] = np.abs(np.diff(means))
        scale = np.median(means) if np.median(means) > 0 else 1.0
        fd = fd / scale * 5.0
        return fd

    @staticmethod
    def _compute_dvars(
        ts: np.ndarray, brain_mask: np.ndarray
    ) -> np.ndarray:
        n_vols = ts.shape[-1]
        dvars = np.zeros(n_vols)
        for t in range(1, n_vols):
            diff = ts[..., t].astype(np.float64) - ts[..., t - 1].astype(np.float64)
            dvars[t] = float(np.sqrt(np.mean(diff[brain_mask] ** 2)))
        return dvars
