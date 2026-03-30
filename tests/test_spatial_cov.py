

from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.spatial_cov import SpatialCovMetric


class TestSpatialCovMetric:

    def test_clean_data(self, clean_subject, clean_masks):
        metric = SpatialCovMetric()
        result = metric.compute(
            clean_subject.cbf_map, clean_masks.gm_mask, clean_masks.wm_mask,
        )
        assert result.spatial_cov > 0
        assert result.gm_wm_ratio > 0

    def test_gm_wm_ratio(self, clean_subject, clean_masks):
        metric = SpatialCovMetric()
        result = metric.compute(
            clean_subject.cbf_map, clean_masks.gm_mask, clean_masks.wm_mask,
        )

        assert 0.5 < result.gm_wm_ratio < 5.0

    def test_vascular_artifact_index(self, clean_subject, clean_masks):
        metric = SpatialCovMetric()
        result = metric.compute(
            clean_subject.cbf_map, clean_masks.gm_mask,
        )
        assert 0.0 <= result.vascular_artifact_index <= 1.0

    def test_empty_mask(self):
        metric = SpatialCovMetric()
        cbf = np.zeros((10, 10, 10))
        mask = np.zeros((10, 10, 10), dtype=bool)
        result = metric.compute(cbf, mask)
        assert result.pass_flag is False

    def test_no_wm_mask(self, clean_subject, clean_masks):
        metric = SpatialCovMetric()
        result = metric.compute(clean_subject.cbf_map, clean_masks.gm_mask)
        assert result.gm_wm_ratio == 0.0
