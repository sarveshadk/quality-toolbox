from __future__ import annotations

import logging
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from qc_toolbox import QCComputationError, __version__
from qc_toolbox.core.bids_loader import BIDSLoader, SubjectData
from qc_toolbox.core.tissue_masks import TissueMaskDeriver, TissueMasks
from qc_toolbox.metrics.qei import QEIMetric, QEIResult
from qc_toolbox.metrics.motion import MotionMetric, MotionResult
from qc_toolbox.metrics.control_label import ControlLabelMetric, ControlLabelResult
from qc_toolbox.metrics.m0_checker import M0Checker, M0Result
from qc_toolbox.metrics.snr import SNRMetric, SNRResult
from qc_toolbox.metrics.histogram import HistogramMetric, HistogramResult
from qc_toolbox.metrics.tissue_mask_qa import TissueMaskQA, TissueMaskQAResult
from qc_toolbox.metrics.spatial_cov import SpatialCovMetric, SpatialCovResult
from qc_toolbox.thresholds.gmm_learner import (
    GMMThresholdLearner,
    ThresholdProfile,
)

logger = logging.getLogger(__name__)


@dataclass
class SubjectQCResult:

    subject_id: str
    session_id: Optional[str] = None
    qei_result: Optional[QEIResult] = None
    motion_result: Optional[MotionResult] = None
    control_label_result: Optional[ControlLabelResult] = None
    m0_result: Optional[M0Result] = None
    snr_result: Optional[SNRResult] = None
    histogram_result: Optional[HistogramResult] = None
    tissue_mask_qa_result: Optional[TissueMaskQAResult] = None
    spatial_cov_result: Optional[SpatialCovResult] = None
    overall_flag: str = "FAIL"
    processing_time: float = 0.0
    error: Optional[str] = None


def _determine_flag(result: SubjectQCResult) -> str:
    flags: List[bool] = []

    if result.qei_result is not None:
        flags.append(result.qei_result.pass_flag)
    if result.motion_result is not None:
        flags.append(result.motion_result.pass_flag)
    if result.control_label_result is not None:
        flags.append(result.control_label_result.pass_flag)
    if result.m0_result is not None:
        flags.append(result.m0_result.pass_flag)
    if result.snr_result is not None:
        flags.append(result.snr_result.pass_flag)
    if result.histogram_result is not None:
        flags.append(result.histogram_result.pass_flag)
    if result.tissue_mask_qa_result is not None:
        flags.append(result.tissue_mask_qa_result.pass_flag)
    if result.spatial_cov_result is not None:
        flags.append(result.spatial_cov_result.pass_flag)

    if not flags:
        return "FAIL"
    n_fail = sum(1 for f in flags if not f)
    if n_fail == 0:
        return "PASS"
    if n_fail <= 2:
        return "WARN"
    return "FAIL"


class QCPipeline:

    def __init__(
        self,
        bids_dir: str | Path,
        output_dir: str | Path,
        threshold_profile: str = "default",
        run_motion: bool = True,
        run_control_label: bool = True,
        run_m0: bool = True,
        run_snr: bool = True,
        run_histogram: bool = True,
        run_tissue_qa: bool = True,
        run_spatial_cov: bool = True,
        n_bootstrap: int = 100,
        verbose: bool = True,
        n_workers: int = 1,
    ) -> None:
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.threshold_profile_name = threshold_profile
        self.run_motion = run_motion
        self.run_control_label = run_control_label
        self.run_m0 = run_m0
        self.run_snr = run_snr
        self.run_histogram = run_histogram
        self.run_tissue_qa = run_tissue_qa
        self.run_spatial_cov = run_spatial_cov
        self.n_bootstrap = n_bootstrap
        self.verbose = verbose
        self.n_workers = max(1, n_workers)


        self.threshold_profile = self._load_profile(threshold_profile)

    @staticmethod
    def _load_profile(name: str) -> ThresholdProfile:
        profiles_dir = Path(__file__).parent / "thresholds" / "profiles"
        path = profiles_dir / f"{name}.json"
        if path.exists():
            return GMMThresholdLearner.load_profile(path)
        default = profiles_dir / "default.json"
        if default.exists():
            warnings.warn(
                f"Profile '{name}' not found — using default.", stacklevel=2
            )
            return GMMThresholdLearner.load_profile(default)
        return ThresholdProfile(name="fallback")

    def run(self) -> pd.DataFrame:
        loader = BIDSLoader(self.bids_dir)
        entries = loader.discover_subjects()

        if not entries:
            logger.warning("No subjects found in %s", self.bids_dir)
            return pd.DataFrame()

        results: List[SubjectQCResult] = []

        if self.n_workers > 1 and len(entries) > 1:
            results = self._run_parallel(loader, entries)
        else:
            results = self._run_sequential(loader, entries)


        if len(results) > 0:
            self.run_population_analysis(results)

        return self._results_to_dataframe(results)

    def _run_sequential(
        self,
        loader: BIDSLoader,
        entries: list,
    ) -> List[SubjectQCResult]:

        results: List[SubjectQCResult] = []
        iterator = tqdm(entries, desc="QC Pipeline", disable=not self.verbose)
        for entry in iterator:
            try:
                sd = loader.load_subject(entry)
                res = self.run_subject(sd)
                results.append(res)
                iterator.set_postfix(
                    subject=sd.subject_id, flag=res.overall_flag
                )
            except Exception as exc:
                logger.error("Subject %s failed: %s", entry[1], exc)
                results.append(
                    SubjectQCResult(
                        subject_id=entry[1],
                        session_id=entry[2],
                        error=str(exc),
                    )
                )
        return results

    def _run_parallel(
        self,
        loader: BIDSLoader,
        entries: list,
    ) -> List[SubjectQCResult]:
        results: List[SubjectQCResult] = []
        with ProcessPoolExecutor(max_workers=self.n_workers) as pool:
            futures = {}
            for entry in entries:
                try:
                    sd = loader.load_subject(entry)
                    future = pool.submit(self.run_subject, sd)
                    futures[future] = sd.subject_id
                except Exception as exc:
                    results.append(
                        SubjectQCResult(
                            subject_id=entry[1],
                            session_id=entry[2],
                            error=str(exc),
                        )
                    )
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="QC Pipeline",
                disable=not self.verbose,
            ):
                try:
                    results.append(future.result())
                except Exception as exc:
                    sid = futures[future]
                    logger.error("Subject %s failed in worker: %s", sid, exc)
                    results.append(
                        SubjectQCResult(subject_id=sid, error=str(exc))
                    )
        return results

    def run_subject(self, subject_data: SubjectData) -> SubjectQCResult:
        t0 = time.time()
        result = SubjectQCResult(
            subject_id=subject_data.subject_id,
            session_id=subject_data.session_id,
        )

        try:
            deriver = TissueMaskDeriver()
            masks = deriver.derive(subject_data.cbf_map, subject_data.affine)

            try:
                qei = QEIMetric()
                result.qei_result = qei.compute(
                    subject_data.cbf_map,
                    masks.gm_prob,
                    masks.wm_prob,
                    subject_data.affine,
                    n_bootstrap=self.n_bootstrap,
                )
            except Exception as exc:
                logger.warning("%s: QEI failed: %s", subject_data.subject_id, exc)

            if self.run_motion:
                try:
                    motion = MotionMetric()
                    result.motion_result = motion.compute_framewise_displacement(
                        subject_data.asl_timeseries, masks.brain_mask
                    )
                except Exception as exc:
                    logger.warning("%s: Motion failed: %s", subject_data.subject_id, exc)

            if self.run_control_label:
                try:
                    cl = ControlLabelMetric()
                    result.control_label_result = cl.analyze(
                        subject_data.asl_timeseries,
                        subject_data.aslcontext,
                        masks.brain_mask,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s: Control-label failed: %s",
                        subject_data.subject_id, exc,
                    )

            if self.run_m0:
                try:
                    m0c = M0Checker()
                    result.m0_result = m0c.check(
                        subject_data.m0_map,
                        masks.brain_mask,
                        subject_data.metadata,
                    )
                except Exception as exc:
                    logger.warning("%s: M0 failed: %s", subject_data.subject_id, exc)

            if self.run_snr:
                try:
                    snr = SNRMetric()
                    result.snr_result = snr.compute(
                        subject_data.asl_timeseries,
                        subject_data.cbf_map,
                        masks.brain_mask,
                        masks.gm_mask,
                    )
                except Exception as exc:
                    logger.warning("%s: SNR failed: %s", subject_data.subject_id, exc)

            if self.run_histogram:
                try:
                    hm = HistogramMetric()
                    ps_cbf = (
                        result.qei_result.pseudo_structural_cbf
                        if result.qei_result is not None
                        else None
                    )
                    result.histogram_result = hm.analyze(
                        subject_data.cbf_map, masks.gm_mask, ps_cbf
                    )
                except Exception as exc:
                    logger.warning(
                        "%s: Histogram failed: %s", subject_data.subject_id, exc
                    )

            if self.run_tissue_qa:
                try:
                    tmqa = TissueMaskQA()
                    result.tissue_mask_qa_result = tmqa.assess(
                        masks, subject_data.cbf_map, subject_data.affine
                    )
                except Exception as exc:
                    logger.warning(
                        "%s: Tissue mask QA failed: %s",
                        subject_data.subject_id, exc,
                    )

            if self.run_spatial_cov:
                try:
                    sc = SpatialCovMetric()
                    result.spatial_cov_result = sc.compute(
                        subject_data.cbf_map,
                        masks.gm_mask,
                        masks.wm_mask,
                    )
                except Exception as exc:
                    logger.warning(
                        "%s: Spatial CoV failed: %s",
                        subject_data.subject_id, exc,
                    )

        except Exception as exc:
            result.error = str(exc)
            logger.error("%s: Pipeline failed: %s", subject_data.subject_id, exc)

        result.processing_time = time.time() - t0
        result.overall_flag = _determine_flag(result)
        return result

    def run_population_analysis(
        self, results: List[SubjectQCResult]
    ) -> None:
        df = self._results_to_dataframe(results)

        csv_path = self.output_dir / "qc_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Saved results to %s", csv_path)

        if len(df) > 20:
            try:
                learner = GMMThresholdLearner()
                profile = learner.fit(df, population="learned")
                out_path = self.output_dir / "learned_thresholds.json"
                learner.save_profile(profile, out_path)
                logger.info("Saved learned thresholds to %s", out_path)
            except Exception as exc:
                logger.warning("GMM threshold learning failed: %s", exc)

    @staticmethod
    def _results_to_dataframe(
        results: List[SubjectQCResult],
    ) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for r in results:
            row: Dict[str, Any] = {
                "subject_id": r.subject_id,
                "session_id": r.session_id or "",
                "overall_flag": r.overall_flag,
                "processing_time": r.processing_time,
                "error": r.error or "",
            }

            if r.qei_result:
                row.update({
                    "qei": r.qei_result.qei_score,
                    "pss": r.qei_result.pss,
                    "di": r.qei_result.di,
                    "ngm_cbf": r.qei_result.ngm_cbf,
                })

            if r.histogram_result:
                row.update({
                    "mean_gm_cbf": r.histogram_result.mean,
                    "median_gm_cbf": r.histogram_result.median,
                    "std_gm_cbf": r.histogram_result.std,
                })

            if r.spatial_cov_result:
                row["spatial_cov"] = r.spatial_cov_result.spatial_cov

            if r.snr_result:
                row["temporal_snr"] = r.snr_result.temporal_snr
                row["spatial_snr"] = r.snr_result.spatial_snr

            if r.motion_result:
                row.update({
                    "mean_fd": r.motion_result.mean_fd,
                    "max_fd": r.motion_result.max_fd,
                    "n_spikes": r.motion_result.n_spikes,
                })

            if r.control_label_result:
                row.update({
                    "label_efficiency": r.control_label_result.label_efficiency,
                    "pattern_valid": r.control_label_result.pattern_valid,
                })

            if r.m0_result:
                row["m0_snr"] = r.m0_result.snr

            if r.tissue_mask_qa_result:
                row.update({
                    "gm_coverage": r.tissue_mask_qa_result.coverage_ratio,
                    "n_mask_components": r.tissue_mask_qa_result.n_components,
                })

            rows.append(row)

        return pd.DataFrame(rows)
