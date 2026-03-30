from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.stats import pearsonr

from qc_toolbox import QCComputationError
from qc_toolbox.core.smoother import smooth_to_fwhm

logger = logging.getLogger(__name__)

_ALPHA: float = -3.0126
_BETA: float = 2.4419
_GAMMA: float = 0.054
_DELTA: float = 0.9272
_EPSILON: float = 2.8478
_ZETA: float = 0.5196

_QEI_PASS_THRESHOLD: float = 0.70


@dataclass
class QEIResult:

    qei_score: float
    pss: float
    di: float
    ngm_cbf: float
    smoothed_cbf: np.ndarray
    pseudo_structural_cbf: np.ndarray
    pass_flag: bool
    component_flags: Dict[str, bool] = field(default_factory=dict)
    ci_lower: Optional[float] = None
    ci_upper: Optional[float] = None


def _safe_cube_root(x: float) -> float:
    if x >= 0:
        return float(np.cbrt(x))
    return -float(np.cbrt(-x))


def _compute_qei_from_components(pss: float, di: float, ngm_cbf: float) -> float:
    pss = float(np.clip(pss, 0.0, 1.0))
    di = max(di, 0.0)
    ngm_cbf = float(np.clip(ngm_cbf, 0.0, 1.0))

    term1 = 1.0 - np.exp(_ALPHA * (pss ** _BETA))
    term2 = np.exp(-(_GAMMA * (di ** _DELTA) + _EPSILON * (ngm_cbf ** _ZETA)))

    product = term1 * term2
    return _safe_cube_root(product)


class QEIMetric:

    ALPHA = _ALPHA
    BETA = _BETA
    GAMMA = _GAMMA
    DELTA = _DELTA
    EPSILON = _EPSILON
    ZETA = _ZETA

    def __init__(self, pass_threshold: float = _QEI_PASS_THRESHOLD) -> None:
        self.pass_threshold = pass_threshold

    def compute(
        self,
        cbf_map: np.ndarray,
        gm_prob: np.ndarray,
        wm_prob: np.ndarray,
        affine: np.ndarray,
        fwhm_mm: float = 5.0,
        brain_mask: Optional[np.ndarray] = None,
        n_bootstrap: int = 100,
        seed: int = 42,
    ) -> QEIResult:
        try:
            return self._compute_impl(
                cbf_map, gm_prob, wm_prob, affine,
                fwhm_mm, brain_mask, n_bootstrap, seed,
            )
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"QEI computation failed: {exc}") from exc

    def _compute_impl(
        self,
        cbf_map: np.ndarray,
        gm_prob: np.ndarray,
        wm_prob: np.ndarray,
        affine: np.ndarray,
        fwhm_mm: float,
        brain_mask: Optional[np.ndarray],
        n_bootstrap: int,
        seed: int,
    ) -> QEIResult:
        smoothed_cbf = smooth_to_fwhm(cbf_map, affine, fwhm_mm)

        ps_cbf = 2.5 * gm_prob + 1.0 * wm_prob
        ps_norm = np.max(ps_cbf)
        if ps_norm > 0:
            ps_cbf = ps_cbf / ps_norm

        if brain_mask is None:
            brain_mask = (gm_prob + wm_prob) > 0.1

        brain_idx = brain_mask.ravel().astype(bool)
        if np.sum(brain_idx) < 10:
            raise QCComputationError("Brain mask has fewer than 10 voxels.")

        gm_mask = gm_prob > 0.5
        wm_mask = wm_prob > 0.5

        flat_cbf = smoothed_cbf.ravel()[brain_idx]
        flat_ps = ps_cbf.ravel()[brain_idx]

        if np.std(flat_cbf) < 1e-12 or np.std(flat_ps) < 1e-12:
            pss = 0.0
        else:
            pss, _ = pearsonr(flat_cbf, flat_ps)
            pss = float(np.clip(pss, 0.0, 1.0))

        gm_vals = smoothed_cbf[gm_mask]
        wm_vals = smoothed_cbf[wm_mask]

        mean_gm_cbf = float(np.mean(gm_vals)) if gm_vals.size > 0 else 0.0

        gm_var = float(np.var(gm_vals, ddof=1)) if gm_vals.size > 1 else 0.0
        wm_var = float(np.var(wm_vals, ddof=1)) if wm_vals.size > 1 else 0.0

        n_gm = max(gm_vals.size, 1)
        n_wm = max(wm_vals.size, 1)
        pooled_var = (
            (n_gm - 1) * gm_var + (n_wm - 1) * wm_var
        ) / max(n_gm + n_wm - 2, 1)

        denom = abs(mean_gm_cbf)
        if denom < 1e-6:
            di = 1e6
        else:
            di = pooled_var / denom

        if gm_vals.size > 0:
            ngm_cbf = float(np.sum(gm_vals < 0)) / float(gm_vals.size)
        else:
            ngm_cbf = 0.0

        qei_score = _compute_qei_from_components(pss, di, ngm_cbf)
        qei_score = float(np.clip(qei_score, 0.0, 1.0))

        component_flags = {
            "pss_ok": pss >= 0.3,
            "di_ok": di <= 500.0,
            "ngm_cbf_ok": ngm_cbf <= 0.5,
        }

        ci_lower, ci_upper = self._bootstrap_ci(
            smoothed_cbf, gm_mask, wm_mask, ps_cbf, brain_mask,
            n_bootstrap, seed,
        )

        return QEIResult(
            qei_score=qei_score,
            pss=pss,
            di=di,
            ngm_cbf=ngm_cbf,
            smoothed_cbf=smoothed_cbf,
            pseudo_structural_cbf=ps_cbf,
            pass_flag=qei_score >= self.pass_threshold,
            component_flags=component_flags,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
        )

    def _bootstrap_ci(
        self,
        smoothed_cbf: np.ndarray,
        gm_mask: np.ndarray,
        wm_mask: np.ndarray,
        ps_cbf: np.ndarray,
        brain_mask: np.ndarray,
        n_bootstrap: int,
        seed: int,
    ) -> tuple[Optional[float], Optional[float]]:
        if n_bootstrap <= 0:
            return None, None

        rng = np.random.default_rng(seed)
        gm_indices = np.argwhere(gm_mask)
        wm_indices = np.argwhere(wm_mask)
        brain_indices = np.argwhere(brain_mask)

        if gm_indices.shape[0] < 5:
            return None, None

        qei_samples = []
        for _ in range(n_bootstrap):
            boot_gm = gm_indices[
                rng.integers(0, gm_indices.shape[0], size=gm_indices.shape[0])
            ]
            boot_gm_vals = smoothed_cbf[
                boot_gm[:, 0], boot_gm[:, 1], boot_gm[:, 2]
            ]

            flat_cbf = smoothed_cbf[brain_mask]
            flat_ps = ps_cbf[brain_mask]
            if np.std(flat_cbf) < 1e-12 or np.std(flat_ps) < 1e-12:
                pss_b = 0.0
            else:
                pss_b, _ = pearsonr(flat_cbf, flat_ps)
                pss_b = float(np.clip(pss_b, 0.0, 1.0))

            mean_gm = float(np.mean(boot_gm_vals))
            gm_var = float(np.var(boot_gm_vals, ddof=1)) if boot_gm_vals.size > 1 else 0.0

            wm_vals = smoothed_cbf[wm_mask]
            wm_var = float(np.var(wm_vals, ddof=1)) if wm_vals.size > 1 else 0.0
            n_g = boot_gm_vals.size
            n_w = max(wm_vals.size, 1)
            pooled = ((n_g - 1) * gm_var + (n_w - 1) * wm_var) / max(n_g + n_w - 2, 1)
            denom = abs(mean_gm) if abs(mean_gm) > 1e-6 else 1e-6
            di_b = pooled / denom

            ngm_b = float(np.sum(boot_gm_vals < 0)) / max(n_g, 1)

            q = _compute_qei_from_components(pss_b, di_b, ngm_b)
            qei_samples.append(float(np.clip(q, 0.0, 1.0)))

        arr = np.array(qei_samples)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))
