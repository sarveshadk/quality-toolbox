

from __future__ import annotations

import numpy as np
import pytest

from qc_toolbox.metrics.tissue_mask_qa import TissueMaskQA


class TestTissueMaskQA:

    def test_clean_masks(self, clean_subject, clean_masks):
        qa = TissueMaskQA()
        result = qa.assess(clean_masks, clean_subject.cbf_map, clean_subject.affine)

        assert result.coverage_ratio > 0.0
        assert result.n_components >= 1
        assert result.symmetry_ratio > 0.0
        assert result.dice is None
        assert result.jaccard is None

    def test_with_reference(self, clean_subject, clean_masks):
        qa = TissueMaskQA()

        result = qa.assess(
            clean_masks, clean_subject.cbf_map, clean_subject.affine,
            reference_gm_mask=clean_masks.gm_mask,
        )
        assert result.dice is not None
        assert result.dice == pytest.approx(1.0)
        assert result.jaccard is not None
        assert result.jaccard == pytest.approx(1.0)

    def test_csf_leakage_index(self, negative_cbf_subject):
        from qc_toolbox.core.tissue_masks import TissueMaskDeriver
        deriver = TissueMaskDeriver()
        masks = deriver.derive(negative_cbf_subject.cbf_map, negative_cbf_subject.affine)

        qa = TissueMaskQA()
        result = qa.assess(
            masks, negative_cbf_subject.cbf_map, negative_cbf_subject.affine,
        )

        assert isinstance(result.csf_leakage_index, float)

    def test_symmetry(self, clean_subject, clean_masks):
        qa = TissueMaskQA()
        result = qa.assess(clean_masks, clean_subject.cbf_map, clean_subject.affine)

        assert result.symmetry_ratio > 0.0
