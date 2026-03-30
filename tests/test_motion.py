from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.motion import MotionMetric


class TestMotionMetric:

    def test_clean_data_low_fd(self, clean_subject):
        metric = MotionMetric()
        result = metric.compute_framewise_displacement(
            clean_subject.asl_timeseries
        )
        assert result.mean_fd >= 0.0
        assert result.fd_trace.shape[0] == clean_subject.asl_timeseries.shape[-1]
        assert result.dvars_trace.shape[0] == clean_subject.asl_timeseries.shape[-1]

    def test_motion_corrupted_higher_fd(self, clean_subject, motion_subject):
        metric = MotionMetric()

        clean_res = metric.compute_framewise_displacement(clean_subject.asl_timeseries)
        motion_res = metric.compute_framewise_displacement(motion_subject.asl_timeseries)

        assert np.max(motion_res.dvars_trace) >= np.max(clean_res.dvars_trace) * 0.5

    def test_single_volume(self):
        ts = np.random.randn(10, 10, 10, 1)
        metric = MotionMetric()
        result = metric.compute_framewise_displacement(ts)
        assert result.mean_fd == 0.0
        assert result.n_spikes == 0

    def test_non_4d_raises(self):
        metric = MotionMetric()
        with pytest.raises(Exception):
            metric.compute_framewise_displacement(np.zeros((10, 10, 10)))

    def test_dvars_first_volume_zero(self, clean_subject):
        metric = MotionMetric()
        result = metric.compute_framewise_displacement(clean_subject.asl_timeseries)
        assert result.dvars_trace[0] == 0.0

    def test_spike_detection(self, motion_subject):
        metric = MotionMetric()
        brain = np.ones(motion_subject.asl_timeseries.shape[:3], dtype=bool)
        result = metric.compute_framewise_displacement(
            motion_subject.asl_timeseries, brain
        )

        assert isinstance(result.spike_indices, list)
