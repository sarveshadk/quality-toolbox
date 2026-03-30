from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from scipy import ndimage as ndi

try:
    from skimage.filters import threshold_multiotsu
except ImportError:
    threshold_multiotsu = None

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)


@dataclass
class TissueMasks:

    gm_mask: np.ndarray
    wm_mask: np.ndarray
    brain_mask: np.ndarray
    gm_prob: np.ndarray
    wm_prob: np.ndarray
    gm_volume: int
    wm_volume: int
    brain_volume: int


class TissueMaskDeriver:
    def __init__(
        self,
        gm_template_path: Optional[str | Path] = None,
        wm_template_path: Optional[str | Path] = None,
    ) -> None:
        self.gm_template_path = (
            Path(gm_template_path) if gm_template_path else None
        )
        self.wm_template_path = (
            Path(wm_template_path) if wm_template_path else None
        )

    def derive(
        self,
        cbf_map: np.ndarray,
        affine: np.ndarray,
        brain_threshold: float = 0.0,
    ) -> TissueMasks:
        if self.gm_template_path and self.wm_template_path:
            return self._from_templates(cbf_map)
        return self._from_otsu(cbf_map, brain_threshold)

    def _from_templates(self, cbf_map: np.ndarray) -> TissueMasks:
        try:
            gm_prob = np.asarray(
                nib.load(str(self.gm_template_path)).dataobj, dtype=np.float64
            )
            wm_prob = np.asarray(
                nib.load(str(self.wm_template_path)).dataobj, dtype=np.float64
            )
        except Exception as exc:
            raise QCComputationError(
                f"Cannot load tissue templates: {exc}"
            ) from exc

        gm_prob = np.clip(gm_prob, 0.0, 1.0)
        wm_prob = np.clip(wm_prob, 0.0, 1.0)

        gm_mask = gm_prob > 0.5
        wm_mask = wm_prob > 0.5
        brain_mask = gm_mask | wm_mask

        return TissueMasks(
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            brain_mask=brain_mask,
            gm_prob=gm_prob,
            wm_prob=wm_prob,
            gm_volume=int(np.sum(gm_mask)),
            wm_volume=int(np.sum(wm_mask)),
            brain_volume=int(np.sum(brain_mask)),
        )

    def _from_otsu(
        self, cbf_map: np.ndarray, brain_threshold: float
    ) -> TissueMasks:
        brain_mask = cbf_map > brain_threshold

        brain_mask = ndi.binary_fill_holes(brain_mask)
        struct = ndi.generate_binary_structure(3, 1)
        brain_mask = ndi.binary_opening(brain_mask, structure=struct, iterations=1)
        brain_mask = ndi.binary_closing(brain_mask, structure=struct, iterations=1)

        brain_vals = cbf_map[brain_mask]
        if brain_vals.size < 10:
            warnings.warn(
                "Too few brain voxels for Otsu segmentation — returning empty masks.",
                stacklevel=2,
            )
            empty = np.zeros_like(cbf_map, dtype=bool)
            prob_zero = np.zeros_like(cbf_map, dtype=np.float64)
            return TissueMasks(
                gm_mask=empty,
                wm_mask=empty,
                brain_mask=brain_mask,
                gm_prob=prob_zero,
                wm_prob=prob_zero,
                gm_volume=0,
                wm_volume=0,
                brain_volume=int(np.sum(brain_mask)),
            )

        if threshold_multiotsu is not None:
            try:
                thresholds = threshold_multiotsu(brain_vals, classes=3)
            except Exception:
                thresholds = np.percentile(brain_vals, [33, 66])
        else:
            thresholds = np.percentile(brain_vals, [33, 66])

        t_low, t_high = float(thresholds[0]), float(thresholds[1])

        wm_mask = brain_mask & (cbf_map >= t_low) & (cbf_map < t_high)
        gm_mask = brain_mask & (cbf_map >= t_high)
        scale = max(t_high - t_low, 1e-6)
        gm_prob = np.zeros_like(cbf_map, dtype=np.float64)
        wm_prob = np.zeros_like(cbf_map, dtype=np.float64)

        gm_prob[brain_mask] = 1.0 / (
            1.0 + np.exp(-(cbf_map[brain_mask] - t_high) / (0.25 * scale))
        )
        mid = (t_low + t_high) / 2.0
        wm_prob[brain_mask] = np.exp(
            -0.5 * ((cbf_map[brain_mask] - mid) / (0.3 * scale)) ** 2
        )

        total = gm_prob + wm_prob
        mask_nonzero = total > 0
        gm_prob[mask_nonzero] /= total[mask_nonzero]
        wm_prob[mask_nonzero] /= total[mask_nonzero]

        return TissueMasks(
            gm_mask=gm_mask,
            wm_mask=wm_mask,
            brain_mask=brain_mask,
            gm_prob=gm_prob,
            wm_prob=wm_prob,
            gm_volume=int(np.sum(gm_mask)),
            wm_volume=int(np.sum(wm_mask)),
            brain_volume=int(np.sum(brain_mask)),
        )
