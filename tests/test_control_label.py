from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from qc_toolbox.metrics.control_label import ControlLabelMetric


class TestControlLabelMetric:

    def test_clean_pattern_valid(self, clean_subject):
        metric = ControlLabelMetric()
        result = metric.analyze(
            clean_subject.asl_timeseries,
            clean_subject.aslcontext,
        )
        assert result.pattern_valid is True
        assert result.is_inverted is False
        assert result.pass_flag is True
        assert result.label_efficiency > 0.0

    def test_inverted_detected(self, inverted_subject):
        metric = ControlLabelMetric()
        result = metric.analyze(
            inverted_subject.asl_timeseries,
            inverted_subject.aslcontext,
        )
        assert result.is_inverted is True
        assert result.pass_flag is False

    def test_empty_context(self, clean_subject):
        metric = ControlLabelMetric()
        empty_ctx = pd.DataFrame({"volume_type": []})
        result = metric.analyze(clean_subject.asl_timeseries, empty_ctx)
        assert result.pass_flag is False
        assert result.pattern_valid is False

    def test_non_alternating_pattern(self, clean_subject):
        metric = ControlLabelMetric()
        bad_ctx = pd.DataFrame({"volume_type": ["control", "control", "label", "label"] * 5})
        result = metric.analyze(clean_subject.asl_timeseries, bad_ctx)
        assert result.pattern_valid is False

    def test_temporal_snr_positive(self, clean_subject):
        metric = ControlLabelMetric()
        result = metric.analyze(
            clean_subject.asl_timeseries,
            clean_subject.aslcontext,
        )
        assert result.temporal_snr >= 0.0
