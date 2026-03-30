from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qc_toolbox.thresholds.gmm_learner import GMMThresholdLearner, ThresholdProfile
from qc_toolbox.thresholds.youden_optimizer import YoudenOptimizer


class TestGMMThresholdLearner:

    @pytest.fixture
    def sample_metrics_df(self):
        rng = np.random.default_rng(42)
        n = 60
        good = rng.normal(0.8, 0.05, n // 2)
        bad = rng.normal(0.4, 0.1, n // 2)
        qei = np.concatenate([good, bad])

        return pd.DataFrame({
            "qei": qei,
            "spatial_cov": rng.normal(50, 15, n),
            "mean_gm_cbf": rng.normal(45, 10, n),
            "temporal_snr": rng.normal(15, 5, n),
            "mean_fd": rng.normal(0.3, 0.15, n),
        })

    def test_fit(self, sample_metrics_df):
        learner = GMMThresholdLearner()
        profile = learner.fit(sample_metrics_df)

        assert profile.name == "default"
        assert len(profile.thresholds) > 0
        assert "qei" in profile.thresholds
        assert profile.thresholds["qei"].threshold > 0

    def test_save_load(self, sample_metrics_df):
        learner = GMMThresholdLearner()
        profile = learner.fit(sample_metrics_df)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name

        learner.save_profile(profile, path)
        loaded = learner.load_profile(path)

        assert loaded.name == profile.name
        assert set(loaded.thresholds.keys()) == set(profile.thresholds.keys())

    def test_too_few_samples(self):
        learner = GMMThresholdLearner()
        tiny_df = pd.DataFrame({"qei": [0.5, 0.6, 0.7]})
        with pytest.raises(Exception):
            learner.fit(tiny_df)

    def test_profile_json_files_valid(self):
        profiles_dir = Path(__file__).parent.parent / "qc_toolbox" / "thresholds" / "profiles"
        for json_file in profiles_dir.glob("*.json"):
            profile = GMMThresholdLearner.load_profile(json_file)
            assert isinstance(profile, ThresholdProfile)
            assert len(profile.thresholds) > 0


class TestYoudenOptimizer:

    def test_perfect_separation(self):
        scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        opt = YoudenOptimizer()
        result = opt.optimize(scores, labels)

        assert result.auc > 0.9
        assert 0.4 < result.optimal_threshold < 0.7
        assert result.sensitivity > 0.8
        assert result.specificity > 0.8
        assert result.youden_j > 0.6

    def test_too_few_samples(self):
        opt = YoudenOptimizer()
        with pytest.raises(Exception):
            opt.optimize(np.array([0.5, 0.6]), np.array([0, 1]))

    def test_single_class_raises(self):
        opt = YoudenOptimizer()
        with pytest.raises(Exception):
            opt.optimize(
                np.array([0.5, 0.6, 0.7, 0.8, 0.9]),
                np.array([1, 1, 1, 1, 1]),
            )
