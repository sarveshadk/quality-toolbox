from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from qc_toolbox import QCComputationError
from qc_toolbox.core.smoother import voxel_size_from_affine
from qc_toolbox.core.tissue_masks import TissueMasks

logger = logging.getLogger(__name__)


@dataclass
class TissueMaskQAResult:

    coverage_ratio: float
    csf_leakage_index: float
    n_components: int
    dice: Optional[float]
    jaccard: Optional[float]
    symmetry_ratio: float
    pass_flag: bool


class TissueMaskQA:
    def __init__(
        self,
        expected_gm_volume_cm3: tuple[float, float] = (400.0, 700.0),
        max_components: int = 5,
        symmetry_threshold: float = 0.20,
    ) -> None:
        self.expected_gm_volume_cm3 = expected_gm_volume_cm3
        self.max_components = max_components
        self.symmetry_threshold = symmetry_threshold

    def assess(
        self,
        masks: TissueMasks,
        cbf_map: np.ndarray,
        affine: np.ndarray,
        reference_gm_mask: Optional[np.ndarray] = None,
    ) -> TissueMaskQAResult:
        try:
            return self._assess(masks, cbf_map, affine, reference_gm_mask)
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"Tissue mask QA failed: {exc}") from exc

    def _assess(
        self,
        masks: TissueMasks,
        cbf_map: np.ndarray,
        affine: np.ndarray,
        reference_gm_mask: Optional[np.ndarray],
    ) -> TissueMaskQAResult:
        vs = voxel_size_from_affine(affine)
        voxel_vol_mm3 = vs[0] * vs[1] * vs[2]
        voxel_vol_cm3 = voxel_vol_mm3 / 1000.0

        gm_vol_cm3 = masks.gm_volume * voxel_vol_cm3
        expected_mid = (
            self.expected_gm_volume_cm3[0] + self.expected_gm_volume_cm3[1]
        ) / 2.0
        coverage_ratio = gm_vol_cm3 / expected_mid if expected_mid > 0 else 0.0

        neg_mask = cbf_map < 0
        overlap = neg_mask & masks.gm_mask
        if np.any(overlap):
            csf_leakage_index = float(np.mean(cbf_map[overlap]))
        else:
            csf_leakage_index = 0.0

        labelled, n_components = ndi.label(masks.gm_mask)

        dice: Optional[float] = None
        jaccard: Optional[float] = None
        if reference_gm_mask is not None:
            ref = reference_gm_mask.astype(bool)
            subj = masks.gm_mask.astype(bool)
            intersection = float(np.sum(ref & subj))
            union = float(np.sum(ref | subj))
            sum_both = float(np.sum(ref)) + float(np.sum(subj))
            dice = 2.0 * intersection / sum_both if sum_both > 0 else 0.0
            jaccard = intersection / union if union > 0 else 0.0

        mid_x = masks.gm_mask.shape[0] // 2
        left_vol = int(np.sum(masks.gm_mask[:mid_x, ...]))
        right_vol = int(np.sum(masks.gm_mask[mid_x:, ...]))
        if right_vol > 0:
            symmetry_ratio = left_vol / right_vol
        elif left_vol > 0:
            symmetry_ratio = float("inf")
        else:
            symmetry_ratio = 1.0

        coverage_ok = (
            self.expected_gm_volume_cm3[0] <= gm_vol_cm3
            <= self.expected_gm_volume_cm3[1]
        )
        components_ok = n_components <= self.max_components
        symmetry_ok = abs(1.0 - symmetry_ratio) <= self.symmetry_threshold
        pass_flag = coverage_ok and components_ok and symmetry_ok

        return TissueMaskQAResult(
            coverage_ratio=coverage_ratio,
            csf_leakage_index=csf_leakage_index,
            n_components=n_components,
            dice=dice,
            jaccard=jaccard,
            symmetry_ratio=symmetry_ratio,
            pass_flag=pass_flag,
        )
