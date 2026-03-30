

from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.histogram import HistogramMetric


class TestHistogramMetric:

    def test_clean_data_passes(self, clean_subject, clean_masks):
        metric = HistogramMetric()
        result = metric.analyze(clean_subject.cbf_map, clean_masks.gm_mask)

        assert result.mean > 0
        assert result.median > 0
        assert result.std > 0
        assert len(result.percentiles) == 7
        assert 50 in result.percentiles

    def test_kurtosis_and_skewness_computed(self, clean_subject, clean_masks):
        metric = HistogramMetric()
        result = metric.analyze(clean_subject.cbf_map, clean_masks.gm_mask)
        assert isinstance(result.kurtosis, float)
        assert isinstance(result.skewness, float)

    def test_empty_mask(self):
        metric = HistogramMetric()
        cbf = np.zeros((10, 10, 10))
        mask = np.zeros((10, 10, 10), dtype=bool)
        result = metric.analyze(cbf, mask)
        assert result.pass_flag is False

    def test_physiological_range(self, clean_subject, clean_masks):
        metric = HistogramMetric()
        result = metric.analyze(clean_subject.cbf_map, clean_masks.gm_mask)
        assert 0.0 <= result.fraction_outside <= 1.0

    def test_figure_generation(self, clean_subject, clean_masks):
        metric = HistogramMetric()
        result = metric.analyze(
            clean_subject.cbf_map, clean_masks.gm_mask,
            generate_figure=True,
        )
        assert result.figure is not None

    def test_rms_diff_with_ps(self, clean_subject, clean_masks):
        metric = HistogramMetric()
        ps_cbf = 2.5 * clean_masks.gm_prob + 1.0 * clean_masks.wm_prob
        result = metric.analyze(
            clean_subject.cbf_map, clean_masks.gm_mask, ps_cbf=ps_cbf,
        )
        assert result.rms_diff > 0.0
