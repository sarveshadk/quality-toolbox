

from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.pipeline import QCPipeline, SubjectQCResult, _determine_flag
from tests.synthetic.generators import make_clean_cbf, make_motion_corrupted_cbf


class TestDetermineFlag:

    def test_all_pass(self):
        r = SubjectQCResult(subject_id="test")


        assert _determine_flag(r) == "FAIL"

    def test_pass_with_qei(self, clean_subject, clean_masks):
        from qc_toolbox.metrics.qei import QEIMetric, QEIResult

        r = SubjectQCResult(subject_id="test")
        r.qei_result = QEIResult(
            qei_score=0.85, pss=0.7, di=10, ngm_cbf=0.02,
            smoothed_cbf=np.zeros((10, 10, 10)),
            pseudo_structural_cbf=np.zeros((10, 10, 10)),
            pass_flag=True,
        )
        assert _determine_flag(r) == "PASS"


class TestRunSubject:

    def test_clean_subject_runs(self, clean_subject):
        pipe = QCPipeline.__new__(QCPipeline)
        pipe.run_motion = True
        pipe.run_control_label = True
        pipe.run_m0 = True
        pipe.run_snr = True
        pipe.run_histogram = True
        pipe.run_tissue_qa = True
        pipe.run_spatial_cov = True
        pipe.n_bootstrap = 5

        result = pipe.run_subject(clean_subject)

        assert result.subject_id == clean_subject.subject_id
        assert result.processing_time > 0
        assert result.overall_flag in ("PASS", "WARN", "FAIL")
        assert result.error is None

    def test_motion_subject_flags(self, motion_subject):
        pipe = QCPipeline.__new__(QCPipeline)
        pipe.run_motion = True
        pipe.run_control_label = True
        pipe.run_m0 = True
        pipe.run_snr = True
        pipe.run_histogram = True
        pipe.run_tissue_qa = True
        pipe.run_spatial_cov = True
        pipe.n_bootstrap = 5

        result = pipe.run_subject(motion_subject)
        assert result.overall_flag in ("PASS", "WARN", "FAIL")

    def test_disabled_metrics(self, clean_subject):
        pipe = QCPipeline.__new__(QCPipeline)
        pipe.run_motion = False
        pipe.run_control_label = False
        pipe.run_m0 = False
        pipe.run_snr = False
        pipe.run_histogram = False
        pipe.run_tissue_qa = False
        pipe.run_spatial_cov = False
        pipe.n_bootstrap = 0

        result = pipe.run_subject(clean_subject)
        assert result.motion_result is None
        assert result.control_label_result is None
