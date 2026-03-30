from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from qc_toolbox.core.bids_loader import SubjectData

logger = logging.getLogger(__name__)


def _make_affine(voxel_size: Tuple[float, float, float] = (3.0, 3.0, 3.0)) -> np.ndarray:
    aff = np.eye(4)
    aff[0, 0], aff[1, 1], aff[2, 2] = voxel_size
    return aff


def _make_brain_mask(shape: Tuple[int, int, int]) -> np.ndarray:
    cx, cy, cz = shape[0] // 2, shape[1] // 2, shape[2] // 2
    rx, ry, rz = shape[0] * 0.38, shape[1] * 0.38, shape[2] * 0.38
    z, y, x = np.mgrid[:shape[0], :shape[1], :shape[2]]
    mask = (
        ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 + ((z - cz) / rz) ** 2
    ) <= 1.0
    return mask


def _make_spatial_cbf(
    shape: Tuple[int, int, int],
    mean_gm: float = 50.0,
    mean_wm: float = 25.0,
    sigma: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    brain = _make_brain_mask(shape)

    inner = _make_brain_mask(
        (int(shape[0] * 0.6), int(shape[1] * 0.6), int(shape[2] * 0.6))
    )
    wm = np.zeros(shape, dtype=bool)
    s0 = (shape[0] - inner.shape[0]) // 2
    s1 = (shape[1] - inner.shape[1]) // 2
    s2 = (shape[2] - inner.shape[2]) // 2
    wm[s0:s0 + inner.shape[0], s1:s1 + inner.shape[1], s2:s2 + inner.shape[2]] = inner

    gm = brain & ~wm

    rng = np.random.default_rng(42)
    noise = rng.normal(0, 5, shape)
    noise = gaussian_filter(noise, sigma=sigma)

    cbf = np.zeros(shape, dtype=np.float64)
    cbf[gm] = mean_gm + noise[gm]
    cbf[wm] = mean_wm + noise[wm]

    return cbf, gm, wm


def _make_asl_timeseries(
    cbf: np.ndarray,
    brain_mask: np.ndarray,
    n_pairs: int = 10,
    baseline: float = 1000.0,
    noise_level: float = 5.0,
    seed: int = 42,
) -> Tuple[np.ndarray, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    shape = cbf.shape
    n_vols = n_pairs * 2
    ts = np.zeros((*shape, n_vols), dtype=np.float64)

    types: List[str] = []
    for i in range(n_vols):
        noise = rng.normal(0, noise_level, shape)
        if i % 2 == 0:
            ts[..., i] = baseline + noise
            ts[..., i][brain_mask] += cbf[brain_mask] * 0.5
            types.append("control")
        else:
            ts[..., i] = baseline + noise
            ts[..., i][brain_mask] -= cbf[brain_mask] * 0.5
            types.append("label")

    ctx = pd.DataFrame({"volume_type": types})
    return ts, ctx


def make_clean_cbf(
    shape: Tuple[int, int, int] = (64, 64, 40),
    voxel_size: Tuple[float, float, float] = (3.0, 3.0, 3.0),
    mean_gm: float = 50.0,
    mean_wm: float = 25.0,
    n_pairs: int = 10,
) -> SubjectData:
    cbf, gm, wm = _make_spatial_cbf(shape, mean_gm, mean_wm)
    brain = gm | wm
    ts, ctx = _make_asl_timeseries(cbf, brain, n_pairs=n_pairs)

    affine = _make_affine(voxel_size)
    ctrl_idx = ctx.index[ctx["volume_type"] == "control"].tolist()
    lbl_idx = ctx.index[ctx["volume_type"] == "label"].tolist()
    cbf_map = np.mean(ts[..., ctrl_idx], axis=-1) - np.mean(ts[..., lbl_idx], axis=-1)

    return SubjectData(
        subject_id="sub-clean01",
        session_id=None,
        cbf_map=cbf_map,
        m0_map=np.full(shape, 1000.0, dtype=np.float64),
        asl_timeseries=ts,
        metadata={
            "ArterialSpinLabelingType": "pCASL",
            "PostLabelingDelay": 1800,
            "LabelingDuration": 1800,
            "M0Type": "Separate",
            "MagneticFieldStrength": 3.0,
        },
        affine=affine,
        aslcontext=ctx,
    )


def make_motion_corrupted_cbf(
    n_spikes: int = 5,
    fd_magnitude: float = 2.0,
    shape: Tuple[int, int, int] = (64, 64, 40),
) -> SubjectData:
    sd = make_clean_cbf(shape=shape)
    rng = np.random.default_rng(99)

    ts = sd.asl_timeseries.copy()
    n_vols = ts.shape[-1]
    spike_vols = rng.choice(n_vols, size=min(n_spikes, n_vols), replace=False)

    for v in spike_vols:
        shift = int(fd_magnitude)
        ts[:, :, :, v] = np.roll(ts[:, :, :, v], shift, axis=0)
        ts[:, :, :, v] += rng.normal(0, 50, ts.shape[:3])

    ctrl = sd.aslcontext.index[sd.aslcontext["volume_type"] == "control"].tolist()
    lbl = sd.aslcontext.index[sd.aslcontext["volume_type"] == "label"].tolist()
    cbf_map = np.mean(ts[..., ctrl], axis=-1) - np.mean(ts[..., lbl], axis=-1)

    return SubjectData(
        subject_id="sub-motion01",
        session_id=None,
        cbf_map=cbf_map,
        m0_map=sd.m0_map,
        asl_timeseries=ts,
        metadata=sd.metadata,
        affine=sd.affine,
        aslcontext=sd.aslcontext,
    )


def make_low_snr_cbf(
    snr: float = 3.0,
    shape: Tuple[int, int, int] = (64, 64, 40),
) -> SubjectData:
    cbf, gm, wm = _make_spatial_cbf(shape, mean_gm=50.0, mean_wm=25.0)
    brain = gm | wm
    noise_level = 50.0 / max(snr, 0.1)
    ts, ctx = _make_asl_timeseries(cbf, brain, noise_level=noise_level, seed=77)

    ctrl = ctx.index[ctx["volume_type"] == "control"].tolist()
    lbl = ctx.index[ctx["volume_type"] == "label"].tolist()
    cbf_map = np.mean(ts[..., ctrl], axis=-1) - np.mean(ts[..., lbl], axis=-1)

    return SubjectData(
        subject_id="sub-lowsnr01",
        session_id=None,
        cbf_map=cbf_map,
        m0_map=np.full(shape, 1000.0),
        asl_timeseries=ts,
        metadata={
            "ArterialSpinLabelingType": "pCASL",
            "PostLabelingDelay": 1800,
            "LabelingDuration": 1800,
            "M0Type": "Separate",
            "MagneticFieldStrength": 3.0,
        },
        affine=_make_affine(),
        aslcontext=ctx,
    )


def make_inverted_pattern_cbf(
    shape: Tuple[int, int, int] = (64, 64, 40),
) -> SubjectData:
    cbf, gm, wm = _make_spatial_cbf(shape)
    brain = gm | wm

    rng = np.random.default_rng(55)
    n_pairs = 10
    n_vols = n_pairs * 2
    ts = np.zeros((*shape, n_vols), dtype=np.float64)
    types: List[str] = []

    for i in range(n_vols):
        noise = rng.normal(0, 5, shape)
        if i % 2 == 0:
            ts[..., i] = 1000 + noise
            ts[..., i][brain] -= cbf[brain] * 0.5
            types.append("control")
        else:
            ts[..., i] = 1000 + noise
            ts[..., i][brain] += cbf[brain] * 0.5
            types.append("label")

    ctx = pd.DataFrame({"volume_type": types})
    ctrl = [i for i in range(n_vols) if i % 2 == 0]
    lbl = [i for i in range(n_vols) if i % 2 == 1]
    cbf_map = np.mean(ts[..., ctrl], axis=-1) - np.mean(ts[..., lbl], axis=-1)

    return SubjectData(
        subject_id="sub-inverted01",
        session_id=None,
        cbf_map=cbf_map,
        m0_map=np.full(shape, 1000.0),
        asl_timeseries=ts,
        metadata={
            "ArterialSpinLabelingType": "pCASL",
            "PostLabelingDelay": 1800,
            "LabelingDuration": 1800,
            "M0Type": "Separate",
            "MagneticFieldStrength": 3.0,
        },
        affine=_make_affine(),
        aslcontext=ctx,
    )


def make_negative_cbf_cbf(
    neg_fraction: float = 0.3,
    shape: Tuple[int, int, int] = (64, 64, 40),
) -> SubjectData:
    sd = make_clean_cbf(shape=shape)
    cbf = sd.cbf_map.copy()
    brain = _make_brain_mask(shape)
    gm_idx = np.argwhere(brain & (cbf > np.median(cbf[brain])))
    rng = np.random.default_rng(66)
    n_neg = int(neg_fraction * len(gm_idx))
    neg_sel = rng.choice(len(gm_idx), size=n_neg, replace=False)
    for idx in neg_sel:
        cbf[gm_idx[idx][0], gm_idx[idx][1], gm_idx[idx][2]] = rng.uniform(-30, -5)

    return SubjectData(
        subject_id="sub-negcbf01",
        session_id=None,
        cbf_map=cbf,
        m0_map=sd.m0_map,
        asl_timeseries=sd.asl_timeseries,
        metadata=sd.metadata,
        affine=sd.affine,
        aslcontext=sd.aslcontext,
    )


def make_bad_m0_cbf(
    m0_type: str = "saturated",
    shape: Tuple[int, int, int] = (64, 64, 40),
) -> SubjectData:
    sd = make_clean_cbf(shape=shape)
    m0 = sd.m0_map.copy() if sd.m0_map is not None else np.full(shape, 1000.0)
    brain = _make_brain_mask(shape)
    rng = np.random.default_rng(88)

    if m0_type == "saturated":
        brain_idx = np.argwhere(brain)
        n_sat = int(0.10 * len(brain_idx))
        sel = rng.choice(len(brain_idx), size=n_sat, replace=False)
        for idx in sel:
            m0[brain_idx[idx][0], brain_idx[idx][1], brain_idx[idx][2]] = 4999.0
    elif m0_type == "dropout":
        brain_idx = np.argwhere(brain)
        n_drop = int(0.15 * len(brain_idx))
        sel = rng.choice(len(brain_idx), size=n_drop, replace=False)
        for idx in sel:
            m0[brain_idx[idx][0], brain_idx[idx][1], brain_idx[idx][2]] = 10.0

    return SubjectData(
        subject_id="sub-badm001",
        session_id=None,
        cbf_map=sd.cbf_map,
        m0_map=m0,
        asl_timeseries=sd.asl_timeseries,
        metadata=sd.metadata,
        affine=sd.affine,
        aslcontext=sd.aslcontext,
    )


def make_multicenter_dataset(
    n_sites: int = 3,
    n_subjects_per_site: int = 10,
    shape: Tuple[int, int, int] = (32, 32, 20),
) -> List[SubjectData]:
    subjects: List[SubjectData] = []
    rng = np.random.default_rng(123)

    for site in range(n_sites):
        site_mean_gm = 45.0 + site * 10
        site_mean_wm = 22.0 + site * 5
        for subj in range(n_subjects_per_site):
            gm = site_mean_gm + rng.normal(0, 5)
            wm = site_mean_wm + rng.normal(0, 3)
            sd = make_clean_cbf(shape=shape, mean_gm=gm, mean_wm=wm, n_pairs=5)
            sd = SubjectData(
                subject_id=f"sub-site{site:02d}_{subj:02d}",
                session_id=None,
                cbf_map=sd.cbf_map,
                m0_map=sd.m0_map,
                asl_timeseries=sd.asl_timeseries,
                metadata={**sd.metadata, "site": f"site-{site:02d}"},
                affine=sd.affine,
                aslcontext=sd.aslcontext,
            )
            subjects.append(sd)

    return subjects
