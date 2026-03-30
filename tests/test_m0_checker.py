

from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.m0_checker import M0Checker


class TestM0Checker:

    def test_clean_m0_passes(self, clean_subject, clean_masks):
        checker = M0Checker()
        result = checker.check(
            clean_subject.m0_map,
            clean_masks.brain_mask,
            clean_subject.metadata,
        )
        assert result.within_range is True
        assert result.snr > 0.0
        assert result.pass_flag is True

    def test_saturated_m0(self, bad_m0_subject, clean_masks):
        checker = M0Checker()
        result = checker.check(
            bad_m0_subject.m0_map,
            clean_masks.brain_mask,
            bad_m0_subject.metadata,
        )

        assert result.snr > 0.0

    def test_none_m0(self, clean_masks):
        checker = M0Checker()
        result = checker.check(None, clean_masks.brain_mask, {})
        assert result.pass_flag is False

    def test_metadata_validation_bad_pld(self, clean_masks):
        checker = M0Checker()
        bad_meta = {
            "ArterialSpinLabelingType": "pCASL",
            "PostLabelingDelay": 100,
        }
        m0 = np.full((64, 64, 40), 1000.0)
        result = checker.check(m0, clean_masks.brain_mask, bad_meta)
        assert result.metadata_valid is False
        assert len(result.metadata_warnings) > 0

    def test_metadata_validation_good(self, clean_masks):
        checker = M0Checker()
        good_meta = {
            "ArterialSpinLabelingType": "pCASL",
            "PostLabelingDelay": 1800,
            "LabelingDuration": 1800,
            "M0Type": "Separate",
        }
        m0 = np.full((64, 64, 40), 1000.0)
        result = checker.check(m0, clean_masks.brain_mask, good_meta)
        assert result.metadata_valid is True
