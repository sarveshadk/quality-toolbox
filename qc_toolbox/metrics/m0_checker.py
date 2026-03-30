from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from qc_toolbox import QCComputationError

logger = logging.getLogger(__name__)


@dataclass
class M0Result:

    snr: float
    saturation_fraction: float
    dropout_fraction: float
    within_range: bool
    metadata_valid: bool
    metadata_warnings: list = None
    pass_flag: bool = True

    def __post_init__(self) -> None:
        if self.metadata_warnings is None:
            self.metadata_warnings = []


class M0Checker:

    def __init__(
        self,
        m0_min: float = 0.0,
        m0_max: float = 5000.0,
        saturation_threshold: float = 0.05,
        dropout_threshold: float = 0.10,
        dropout_ratio: float = 0.20,
    ) -> None:
        self.m0_min = m0_min
        self.m0_max = m0_max
        self.saturation_threshold = saturation_threshold
        self.dropout_threshold = dropout_threshold
        self.dropout_ratio = dropout_ratio

    def check(
        self,
        m0_map: Optional[np.ndarray],
        brain_mask: np.ndarray,
        metadata: Dict[str, Any],
    ) -> M0Result:
        try:
            return self._check(m0_map, brain_mask, metadata)
        except QCComputationError:
            raise
        except Exception as exc:
            raise QCComputationError(f"M0 check failed: {exc}") from exc

    def _check(
        self,
        m0_map: Optional[np.ndarray],
        brain_mask: np.ndarray,
        metadata: Dict[str, Any],
    ) -> M0Result:
        meta_valid, meta_warns = self._validate_metadata(metadata)

        if m0_map is None:
            warnings.warn("No M0 map provided — skipping signal checks.", stacklevel=2)
            return M0Result(
                snr=0.0,
                saturation_fraction=0.0,
                dropout_fraction=0.0,
                within_range=False,
                metadata_valid=meta_valid,
                metadata_warnings=meta_warns,
                pass_flag=False,
            )

        brain_vals = m0_map[brain_mask].astype(np.float64)
        n_brain = brain_vals.size
        if n_brain == 0:
            return M0Result(
                snr=0.0, saturation_fraction=0.0, dropout_fraction=0.0,
                within_range=False, metadata_valid=meta_valid,
                metadata_warnings=meta_warns, pass_flag=False,
            )

        field_strength = metadata.get("MagneticFieldStrength", 3.0)
        scale = field_strength / 3.0
        effective_max = self.m0_max * scale

        within_range = bool(
            np.min(brain_vals) >= self.m0_min
            and np.max(brain_vals) <= effective_max * 2
        )

        hist_max = float(np.percentile(brain_vals, 99.9))
        hist_min = float(np.percentile(brain_vals, 0.1))
        if abs(hist_max - hist_min) < 1e-6 * abs(hist_max):
            saturation_fraction = 0.0
        else:
            n_saturated = int(np.sum(brain_vals >= hist_max * 0.98))
            saturation_fraction = n_saturated / n_brain

        median_m0 = float(np.median(brain_vals))
        dropout_level = self.dropout_ratio * median_m0
        n_dropout = int(np.sum(brain_vals < dropout_level))
        dropout_fraction = n_dropout / n_brain

        background_mask = ~brain_mask
        bg_vals = m0_map[background_mask].astype(np.float64)
        if bg_vals.size > 0 and np.std(bg_vals) > 1e-10:
            snr = float(np.mean(brain_vals)) / float(np.std(bg_vals))
        else:
            snr = float(np.mean(brain_vals)) / max(float(np.std(brain_vals)), 1e-10)

        pass_flag = (
            within_range
            and saturation_fraction <= self.saturation_threshold
            and dropout_fraction <= self.dropout_threshold
            and meta_valid
        )

        return M0Result(
            snr=snr,
            saturation_fraction=saturation_fraction,
            dropout_fraction=dropout_fraction,
            within_range=within_range,
            metadata_valid=meta_valid,
            metadata_warnings=meta_warns,
            pass_flag=pass_flag,
        )

    @staticmethod
    def _validate_metadata(
        metadata: Dict[str, Any],
    ) -> tuple[bool, list[str]]:
        warns: list[str] = []
        valid = True

        asl_type = metadata.get("ArterialSpinLabelingType", "")
        pld = metadata.get("PostLabelingDelay")
        ld = metadata.get("LabelingDuration")

        if asl_type.lower() in ("pcasl", "casl"):
            if pld is not None:
                pld_val = float(pld) if not isinstance(pld, (list, tuple)) else float(pld[0])
                if not (500 <= pld_val <= 4000):
                    warns.append(
                        f"PostLabelingDelay={pld_val} ms outside typical range [500, 4000]."
                    )
                    valid = False
            if ld is not None:
                ld_val = float(ld)
                if not (500 <= ld_val <= 3000):
                    warns.append(
                        f"LabelingDuration={ld_val} ms outside typical range [500, 3000]."
                    )
                    valid = False

        m0_type = metadata.get("M0Type", "")
        if m0_type and m0_type not in (
            "Separate", "Included", "Estimate", "UseControlAsM0"
        ):
            warns.append(f"Unknown M0Type: {m0_type}")
            valid = False

        return valid, warns
