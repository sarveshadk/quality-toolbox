from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from qc_toolbox import ThresholdError

logger = logging.getLogger(__name__)


@dataclass
class YoudenResult:

    optimal_threshold: float
    auc: float
    sensitivity: float
    specificity: float
    youden_j: float


class YoudenOptimizer:

    def optimize(
        self,
        metric_scores: np.ndarray,
        ground_truth_labels: np.ndarray,
    ) -> YoudenResult:
        try:
            return self._optimize(
                np.asarray(metric_scores, dtype=np.float64),
                np.asarray(ground_truth_labels, dtype=np.int32),
            )
        except ThresholdError:
            raise
        except Exception as exc:
            raise ThresholdError(f"Youden optimization failed: {exc}") from exc

    def _optimize(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> YoudenResult:
        from sklearn.metrics import roc_curve, roc_auc_score

        if scores.size < 5:
            raise ThresholdError("Need at least 5 samples.")

        unique = np.unique(labels)
        if len(unique) < 2:
            raise ThresholdError("Need both positive and negative labels.")

        auc = float(roc_auc_score(labels, scores))
        fpr, tpr, thresholds = roc_curve(labels, scores)

        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))

        optimal_threshold = float(thresholds[best_idx])
        sensitivity = float(tpr[best_idx])
        specificity = float(1.0 - fpr[best_idx])
        youden_j = float(j_scores[best_idx])

        return YoudenResult(
            optimal_threshold=optimal_threshold,
            auc=auc,
            sensitivity=sensitivity,
            specificity=specificity,
            youden_j=youden_j,
        )
