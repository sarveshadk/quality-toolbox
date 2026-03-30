from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd

from qc_toolbox import BIDSLoadError

logger = logging.getLogger(__name__)

@dataclass
class SubjectData:

    subject_id: str
    session_id: Optional[str]
    cbf_map: np.ndarray
    m0_map: Optional[np.ndarray]
    asl_timeseries: np.ndarray
    metadata: Dict[str, Any]
    affine: np.ndarray
    aslcontext: pd.DataFrame = field(default_factory=pd.DataFrame)

_SIDECAR_KEYS = [
    "ArterialSpinLabelingType",
    "PostLabelingDelay",
    "LabelingDuration",
    "M0Type",
    "MagneticFieldStrength",
    "RepetitionTime",
    "EchoTime",
    "BackgroundSuppression",
]


def _find_asl_files(bids_root: Path) -> List[Tuple[Path, str, Optional[str]]]:

    results: List[Tuple[Path, str, Optional[str]]] = []
    for nifti in sorted(bids_root.rglob("*_asl.nii.gz")):
        parts = nifti.parts
        subject_id: Optional[str] = None
        session_id: Optional[str] = None
        for p in parts:
            if p.startswith("sub-"):
                subject_id = p
            if p.startswith("ses-"):
                session_id = p
        if subject_id is None:
            logger.warning("Cannot determine subject for %s — skipping.", nifti)
            continue
        results.append((nifti, subject_id, session_id))
    return results


def _derive_sidecar_path(nifti_path: Path, suffix: str, ext: str) -> Path:

    stem = nifti_path.name.replace("_asl.nii.gz", "")
    return nifti_path.parent / f"{stem}_{suffix}{ext}"


def _load_nifti(path: Path) -> Tuple[np.ndarray, np.ndarray]:

    try:
        img = nib.load(str(path))
        return np.asarray(img.dataobj, dtype=np.float64), img.affine
    except Exception as exc:
        raise BIDSLoadError(f"Failed to load NIfTI {path}: {exc}") from exc


def _parse_aslcontext(path: Path) -> pd.DataFrame:
    
    if not path.exists():
        raise BIDSLoadError(f"aslcontext.tsv not found: {path}")
    try:
        df = pd.read_csv(path, sep="\t")
    except Exception as exc:
        raise BIDSLoadError(f"Cannot parse aslcontext: {exc}") from exc

    if df.columns[0] != "volume_type":
        df = pd.read_csv(path, sep="\t", header=None, names=["volume_type"])
    return df


def _parse_sidecar(path: Path) -> Dict[str, Any]:

    if not path.exists():
        warnings.warn(f"JSON sidecar not found: {path}", stacklevel=2)
        return {}
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception as exc:
        warnings.warn(f"Cannot parse JSON sidecar {path}: {exc}", stacklevel=2)
        return {}
    return {k: data[k] for k in _SIDECAR_KEYS if k in data}


class BIDSLoader:

    def __init__(self, bids_root: str | Path) -> None:
        self.bids_root = Path(bids_root)
        if not self.bids_root.is_dir():
            raise BIDSLoadError(f"BIDS root is not a directory: {self.bids_root}")

    def discover_subjects(self) -> List[Tuple[Path, str, Optional[str]]]:
       
        found = _find_asl_files(self.bids_root)
        if not found:
            warnings.warn(
                f"No *_asl.nii.gz files found under {self.bids_root}", stacklevel=2
            )
        return found

    def load_subject(
        self,
        entry: Tuple[Path, str, Optional[str]],
    ) -> SubjectData:

        nifti_path, subject_id, session_id = entry

        data, affine = _load_nifti(nifti_path)
        if data.ndim == 3:
            data = data[..., np.newaxis]
        asl_timeseries = data 

        ctx_path = _derive_sidecar_path(nifti_path, "aslcontext", ".tsv")
        aslcontext = _parse_aslcontext(ctx_path)

        json_path = _derive_sidecar_path(nifti_path, "asl", ".json")
        metadata = _parse_sidecar(json_path)

        control_idx = aslcontext.index[
            aslcontext["volume_type"] == "control"
        ].tolist()
        label_idx = aslcontext.index[
            aslcontext["volume_type"] == "label"
        ].tolist()
        m0_idx = aslcontext.index[
            aslcontext["volume_type"] == "m0scan"
        ].tolist()

        n_vols = asl_timeseries.shape[-1]

        control_idx = [i for i in control_idx if i < n_vols]
        label_idx = [i for i in label_idx if i < n_vols]
        m0_idx = [i for i in m0_idx if i < n_vols]
        if control_idx and label_idx:
            mean_control = np.mean(
                asl_timeseries[..., control_idx], axis=-1
            )
            mean_label = np.mean(
                asl_timeseries[..., label_idx], axis=-1
            )
            cbf_map = mean_control - mean_label
        else:
            warnings.warn(
                f"{subject_id}: no control/label volumes found; CBF set to zeros.",
                stacklevel=2,
            )
            cbf_map = np.zeros(asl_timeseries.shape[:3], dtype=np.float64)

        m0_type = metadata.get("M0Type", "")
        m0_map: Optional[np.ndarray] = None

        if m0_type == "Separate":
            m0_nifti = nifti_path.parent / nifti_path.name.replace(
                "_asl.nii.gz", "_m0scan.nii.gz"
            )
            if m0_nifti.exists():
                m0_data, _ = _load_nifti(m0_nifti)
                m0_map = (
                    np.mean(m0_data, axis=-1)
                    if m0_data.ndim == 4
                    else m0_data
                )
            else:
                warnings.warn(
                    f"{subject_id}: M0Type=Separate but m0scan not found.",
                    stacklevel=2,
                )

        elif m0_type == "Included":
            if m0_idx:
                m0_vols = asl_timeseries[..., m0_idx]
                m0_map = np.mean(m0_vols, axis=-1) if m0_vols.ndim == 4 else m0_vols
            else:
                warnings.warn(
                    f"{subject_id}: M0Type=Included but no m0scan in aslcontext.",
                    stacklevel=2,
                )

        elif m0_type == "UseControlAsM0":
            if control_idx:
                m0_map = np.mean(
                    asl_timeseries[..., control_idx], axis=-1
                )
            else:
                warnings.warn(
                    f"{subject_id}: M0Type=UseControlAsM0 but no control volumes.",
                    stacklevel=2,
                )

        elif m0_type == "Estimate":
            m0_map = None

        return SubjectData(
            subject_id=subject_id,
            session_id=session_id,
            cbf_map=cbf_map,
            m0_map=m0_map,
            asl_timeseries=asl_timeseries,
            metadata=metadata,
            affine=affine,
            aslcontext=aslcontext,
        )

    def load_all(self) -> List[SubjectData]:
        entries = self.discover_subjects()
        results: List[SubjectData] = []
        for entry in entries:
            try:
                results.append(self.load_subject(entry))
            except BIDSLoadError as exc:
                logger.error("Failed to load %s: %s", entry[1], exc)
        return results
