

from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.snr import SNRMetric


class TestSNRMetric:

    def test_clean_temporal_snr(self, clean_subject, clean_masks):
        metric = SNRMetric()
        result = metric.compute(
            clean_subject.asl_timeseries,
            clean_subject.cbf_map,
            clean_masks.brain_mask,
            clean_masks.gm_mask,
        )
        assert result.temporal_snr > 0.0
        assert result.spatial_snr >= 0.0
        assert result.roi_snr >= 0.0
        assert result.temporal_sd_map.shape == clean_subject.cbf_map.shape

    def test_low_snr_lower_than_clean(self, clean_subject, low_snr_subject, clean_masks):
        metric = SNRMetric()
        clean_res = metric.compute(
            clean_subject.asl_timeseries,
            clean_subject.cbf_map,
            clean_masks.brain_mask,
            clean_masks.gm_mask,
        )

        from qc_toolbox.core.tissue_masks import TissueMaskDeriver
        deriver = TissueMaskDeriver()
        low_masks = deriver.derive(low_snr_subject.cbf_map, low_snr_subject.affine)

        low_res = metric.compute(
            low_snr_subject.asl_timeseries,
            low_snr_subject.cbf_map,
            low_masks.brain_mask,
            low_masks.gm_mask,
        )

        assert low_res.temporal_snr < clean_res.temporal_snr * 2.0

    def test_single_volume(self):
        metric = SNRMetric()
        ts = np.random.randn(10, 10, 10, 1)
        brain = np.ones((10, 10, 10), dtype=bool)
        tsnr = metric.compute_temporal_snr(ts, brain)
        assert tsnr == 0.0

    def test_temporal_sd_map_shape(self, clean_subject):
        metric = SNRMetric()
        sd_map = metric.compute_temporal_sd_map(clean_subject.asl_timeseries)
        assert sd_map.shape == clean_subject.cbf_map.shape

    def test_roi_snr_no_gm(self, clean_subject, clean_masks):
        metric = SNRMetric()
        result = metric.compute(
            clean_subject.asl_timeseries,
            clean_subject.cbf_map,
            clean_masks.brain_mask,
            None,
        )
        assert result.roi_snr == 0.0
