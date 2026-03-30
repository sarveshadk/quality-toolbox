from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.qei import QEIMetric, _compute_qei_from_components
from qc_toolbox.core.tissue_masks import TissueMaskDeriver


class TestQEIFormula:

    def test_perfect_scan(self):
        qei = _compute_qei_from_components(pss=0.95, di=5.0, ngm_cbf=0.0)
        assert 0.8 <= qei <= 1.0

    def test_terrible_scan(self):
        qei = _compute_qei_from_components(pss=0.1, di=500.0, ngm_cbf=0.5)
        assert 0.0 <= qei <= 0.3

    def test_boundary_pss_zero(self):
        qei = _compute_qei_from_components(pss=0.0, di=10.0, ngm_cbf=0.05)
        assert qei < 0.2

    def test_negative_clipping(self):
        qei = _compute_qei_from_components(pss=0.0, di=1e6, ngm_cbf=1.0)
        assert qei >= 0.0


class TestQEIMetric:

    def test_clean_subject_passes(self, clean_subject):
        deriver = TissueMaskDeriver()
        masks = deriver.derive(clean_subject.cbf_map, clean_subject.affine)

        metric = QEIMetric()
        result = metric.compute(
            clean_subject.cbf_map,
            masks.gm_prob, masks.wm_prob,
            clean_subject.affine,
            n_bootstrap=10,
        )

        assert result.qei_score > 0.0
        assert 0 <= result.pss <= 1
        assert result.di >= 0
        assert 0 <= result.ngm_cbf <= 1
        assert result.smoothed_cbf.shape == clean_subject.cbf_map.shape

    def test_negative_cbf_raises_ngm(self, negative_cbf_subject):
        deriver = TissueMaskDeriver()
        masks = deriver.derive(negative_cbf_subject.cbf_map, negative_cbf_subject.affine)

        metric = QEIMetric()
        result = metric.compute(
            negative_cbf_subject.cbf_map,
            masks.gm_prob, masks.wm_prob,
            negative_cbf_subject.affine,
            n_bootstrap=0,
        )

        assert result.ngm_cbf > 0.0

    def test_bootstrap_ci(self, clean_subject):
        deriver = TissueMaskDeriver()
        masks = deriver.derive(clean_subject.cbf_map, clean_subject.affine)

        metric = QEIMetric()
        result = metric.compute(
            clean_subject.cbf_map,
            masks.gm_prob, masks.wm_prob,
            clean_subject.affine,
            n_bootstrap=20,
        )

        assert result.ci_lower is not None
        assert result.ci_upper is not None
        assert result.ci_lower <= result.qei_score <= result.ci_upper or \
               abs(result.ci_lower - result.qei_score) < 0.1

    def test_empty_mask_raises(self):
        metric = QEIMetric()
        cbf = np.zeros((10, 10, 10))
        gm_prob = np.zeros((10, 10, 10))
        wm_prob = np.zeros((10, 10, 10))
        affine = np.eye(4)

        with pytest.raises(Exception):
            metric.compute(cbf, gm_prob, wm_prob, affine, n_bootstrap=0)
