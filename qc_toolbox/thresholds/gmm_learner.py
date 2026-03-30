from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from qc_toolbox import ThresholdError

logger = logging.getLogger(__name__)

_METRIC_COLUMNS = [
    "qei", "spatial_cov", "mean_gm_cbf", "temporal_snr", "mean_fd",
]


@dataclass
class MetricThreshold:

    metric: str
    threshold: float
    direction: str = "above"
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    n_components_optimal: int = 2


@dataclass
class ThresholdProfile:

    name: str
    population: str = "default"
    thresholds: Dict[str, MetricThreshold] = field(default_factory=dict)


class GMMThresholdLearner:

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def fit(
        self,
        metrics_df: pd.DataFrame,
        population: str = "default",
    ) -> ThresholdProfile:
        from sklearn.mixture import GaussianMixture

        profile = ThresholdProfile(name=population, population=population)

        direction_map: Dict[str, str] = {
            "qei": "above",
            "spatial_cov": "below",
            "mean_gm_cbf": "above",
            "temporal_snr": "above",
            "mean_fd": "below",
        }

        for col in _METRIC_COLUMNS:
            if col not in metrics_df.columns:
                continue

            vals = metrics_df[col].dropna().values.astype(np.float64)
            if vals.size < 10:
                logger.warning("Skipping %s — too few samples (%d).", col, vals.size)
                continue

            direction = direction_map.get(col, "above")

            try:
                bics: List[float] = []
                for n in (1, 2, 3):
                    gmm_tmp = GaussianMixture(
                        n_components=n, random_state=self.seed, max_iter=200
                    )
                    gmm_tmp.fit(vals.reshape(-1, 1))
                    bics.append(gmm_tmp.bic(vals.reshape(-1, 1)))

                optimal_n = int(np.argmin(bics)) + 1

                gmm = GaussianMixture(
                    n_components=2, random_state=self.seed, max_iter=200
                )
                gmm.fit(vals.reshape(-1, 1))

                means = gmm.means_.flatten()
                stds = np.sqrt(gmm.covariances_.flatten())

                threshold = self._find_intersection(
                    means[0], stds[0], gmm.weights_[0],
                    means[1], stds[1], gmm.weights_[1],
                )

                ci_lower, ci_upper = self._bootstrap_threshold(
                    vals, n_boot=100
                )

                profile.thresholds[col] = MetricThreshold(
                    metric=col,
                    threshold=threshold,
                    direction=direction,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    n_components_optimal=optimal_n,
                )

            except Exception as exc:
                logger.warning("GMM fit failed for %s: %s", col, exc)

        if not profile.thresholds:
            raise ThresholdError("No thresholds could be learned.")

        return profile

    @staticmethod
    def _find_intersection(
        mu1: float, s1: float, w1: float,
        mu2: float, s2: float, w2: float,
    ) -> float:
        lo = min(mu1, mu2) - 2 * max(s1, s2)
        hi = max(mu1, mu2) + 2 * max(s1, s2)
        xs = np.linspace(lo, hi, 5000)

        s1 = max(s1, 1e-10)
        s2 = max(s2, 1e-10)

        pdf1 = w1 * np.exp(-0.5 * ((xs - mu1) / s1) ** 2) / (s1 * np.sqrt(2 * np.pi))
        pdf2 = w2 * np.exp(-0.5 * ((xs - mu2) / s2) ** 2) / (s2 * np.sqrt(2 * np.pi))

        diff = pdf1 - pdf2
        sign_changes = np.where(np.diff(np.sign(diff)))[0]

        if len(sign_changes) == 0:
            return float((mu1 + mu2) / 2)

        mid = (mu1 + mu2) / 2
        best = sign_changes[np.argmin(np.abs(xs[sign_changes] - mid))]
        return float(xs[best])

    def _bootstrap_threshold(
        self, vals: np.ndarray, n_boot: int = 100
    ) -> tuple[float, float]:
        from sklearn.mixture import GaussianMixture

        rng = np.random.default_rng(self.seed)
        thresholds: List[float] = []

        for _ in range(n_boot):
            sample = rng.choice(vals, size=vals.size, replace=True)
            try:
                gmm = GaussianMixture(
                    n_components=2, random_state=self.seed, max_iter=100
                )
                gmm.fit(sample.reshape(-1, 1))
                means = gmm.means_.flatten()
                stds = np.sqrt(gmm.covariances_.flatten())
                t = self._find_intersection(
                    means[0], stds[0], gmm.weights_[0],
                    means[1], stds[1], gmm.weights_[1],
                )
                thresholds.append(t)
            except Exception:
                continue

        if not thresholds:
            median_val = float(np.median(vals))
            return median_val, median_val

        arr = np.array(thresholds)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    @staticmethod
    def save_profile(profile: ThresholdProfile, path: str | Path) -> None:
        data: Dict[str, Any] = {
            "name": profile.name,
            "population": profile.population,
            "thresholds": {
                k: asdict(v) for k, v in profile.thresholds.items()
            },
        }
        with open(str(path), "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)

    @staticmethod
    def load_profile(path: str | Path) -> ThresholdProfile:
        with open(str(path), "r", encoding="utf-8") as fh:
            data = json.load(fh)

        profile = ThresholdProfile(
            name=data.get("name", "unknown"),
            population=data.get("population", "unknown"),
        )
        for k, v in data.get("thresholds", {}).items():
            profile.thresholds[k] = MetricThreshold(**v)

        return profile
