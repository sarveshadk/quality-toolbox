from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from tests.synthetic.generators import (
    make_clean_cbf,
    make_motion_corrupted_cbf,
    make_low_snr_cbf,
    make_inverted_pattern_cbf,
    make_negative_cbf_cbf,
    make_bad_m0_cbf,
)
from qc_toolbox.core.tissue_masks import TissueMaskDeriver


@pytest.fixture
def clean_subject():
    return make_clean_cbf()


@pytest.fixture
def motion_subject():
    return make_motion_corrupted_cbf()


@pytest.fixture
def low_snr_subject():
    return make_low_snr_cbf(snr=3.0)


@pytest.fixture
def inverted_subject():
    return make_inverted_pattern_cbf()


@pytest.fixture
def negative_cbf_subject():
    return make_negative_cbf_cbf(neg_fraction=0.3)


@pytest.fixture
def bad_m0_subject():
    return make_bad_m0_cbf(m0_type="saturated")


@pytest.fixture
def clean_masks(clean_subject):
    deriver = TissueMaskDeriver()
    return deriver.derive(clean_subject.cbf_map, clean_subject.affine)


@pytest.fixture
def sample_aslcontext():
    types = ["control", "label"] * 10
    return pd.DataFrame({"volume_type": types})
