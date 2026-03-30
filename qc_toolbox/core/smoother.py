from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter
_FWHM_TO_SIGMA: float = 2.0 * np.sqrt(2.0 * np.log(2.0))


def apply_gaussian_smooth(
    volume: np.ndarray,
    fwhm_mm: float,
    voxel_size: tuple[float, ...],
) -> np.ndarray:
    
    sigma_mm = fwhm_mm / _FWHM_TO_SIGMA
    sigma_vox = tuple(sigma_mm / vs for vs in voxel_size[:3])
    return gaussian_filter(volume.astype(np.float64), sigma=sigma_vox, mode="constant")


def voxel_size_from_affine(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.sqrt(np.sum(affine[:3, i] ** 2))) for i in range(3))


def smooth_to_fwhm(
    volume: np.ndarray,
    affine: np.ndarray,
    fwhm_mm: float,
) -> np.ndarray:
    vs = voxel_size_from_affine(affine)
    return apply_gaussian_smooth(volume, fwhm_mm, vs)
