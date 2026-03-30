from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from qc_toolbox import __version__

logger = logging.getLogger("qc_toolbox")


@click.group()
@click.version_option(version=__version__, prog_name="qc-toolbox")
def cli() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    )


@cli.command()
@click.option("--bids", required=True, type=click.Path(exists=True), help="BIDS root directory.")
@click.option("--output", required=True, type=click.Path(), help="Output directory.")
@click.option("--profile", default="default", help="Threshold profile name.")
@click.option("--qei-min", default=None, type=float, help="Override QEI minimum threshold.")
@click.option("--spatial-cov-max", default=None, type=float, help="Override spatial CoV max.")
@click.option("--mean-gm-min", default=None, type=float, help="Override mean GM CBF minimum.")
@click.option("--mean-gm-max", default=None, type=float, help="Override mean GM CBF maximum.")
@click.option("--fd-max", default=None, type=float, help="Override max FD threshold (mm).")
@click.option("--no-motion", is_flag=True, help="Skip motion analysis.")
@click.option("--no-m0", is_flag=True, help="Skip M0 checks.")
@click.option("--n-workers", default=1, type=int, help="Number of parallel workers.")
@click.option("--verbose/--quiet", default=True, help="Verbose output.")
def run(
    bids: str,
    output: str,
    profile: str,
    qei_min: float | None,
    spatial_cov_max: float | None,
    mean_gm_min: float | None,
    mean_gm_max: float | None,
    fd_max: float | None,
    no_motion: bool,
    no_m0: bool,
    n_workers: int,
    verbose: bool,
) -> None:
    from qc_toolbox.pipeline import QCPipeline

    click.echo(f"QC Toolbox v{__version__}")
    click.echo(f"BIDS directory : {bids}")
    click.echo(f"Output         : {output}")
    click.echo(f"Profile        : {profile}")

    pipe = QCPipeline(
        bids_dir=bids,
        output_dir=output,
        threshold_profile=profile,
        run_motion=not no_motion,
        run_m0=not no_m0,
        n_workers=n_workers,
        verbose=verbose,
    )

    df = pipe.run()

    if df.empty:
        click.echo("No subjects were processed.", err=True)
        sys.exit(1)

    click.echo(f"Processed {len(df)} subjects.")
    click.echo(f"PASS: {(df['overall_flag'] == 'PASS').sum()}  "
               f"WARN: {(df['overall_flag'] == 'WARN').sum()}  "
               f"FAIL: {(df['overall_flag'] == 'FAIL').sum()}")

    csv_path = Path(output) / "qc_results.csv"
    click.echo(f"Results saved to {csv_path}")


@cli.command()
@click.option("--results", required=True, type=click.Path(exists=True), help="QC results CSV.")
@click.option("--output", required=True, type=click.Path(), help="Output file/directory.")
@click.option(
    "--format", "fmt", default="csv",
    type=click.Choice(["csv", "html", "pdf", "all"]),
    help="Report format.",
)
def report(results: str, output: str, fmt: str) -> None:
    import pandas as pd

    from qc_toolbox.report import QCReporter
    from qc_toolbox.pipeline import SubjectQCResult

    df = pd.read_csv(results, comment="#")
    click.echo(f"Loaded {len(df)} rows from {results}")

    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)

    dummy_results: list[SubjectQCResult] = [
        SubjectQCResult(subject_id=row.get("subject_id", ""), session_id=row.get("session_id"))
        for _, row in df.iterrows()
    ]

    if fmt in ("csv", "all"):
        QCReporter.generate_csv(dummy_results, out_dir / "qc_results.csv")
    if fmt in ("html", "all"):
        QCReporter.generate_html_report(dummy_results, out_dir / "qc_report.html")
    if fmt in ("pdf", "all"):
        click.echo("Per-subject PDFs require full pipeline results (not CSV).")

    click.echo(f"Reports saved to {out_dir}")


@cli.command("learn-thresholds")
@click.option("--results", required=True, type=click.Path(exists=True), help="QC results CSV.")
@click.option("--output", required=True, type=click.Path(), help="Output JSON path.")
@click.option("--population", default="learned", help="Population label.")
def learn_thresholds(results: str, output: str, population: str) -> None:
    import pandas as pd

    from qc_toolbox.thresholds.gmm_learner import GMMThresholdLearner

    df = pd.read_csv(results, comment="#")
    click.echo(f"Loaded {len(df)} rows.")

    learner = GMMThresholdLearner()
    profile = learner.fit(df, population=population)
    learner.save_profile(profile, output)

    click.echo(f"Thresholds saved to {output}")
    for k, v in profile.thresholds.items():
        click.echo(f"  {k}: {v.threshold:.4f} ({v.direction})")


@cli.command()
@click.option("--results", default=None, type=click.Path(), help="Pre-computed results CSV.")
@click.option("--port", default=8501, type=int, help="Streamlit port.")
def dashboard(results: str | None, port: int) -> None:
    import subprocess

    dashboard_path = Path(__file__).parent / "dashboard.py"
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(dashboard_path),
        "--server.port", str(port),
        "--",
    ]
    if results:
        cmd.append(results)

    click.echo(f"Launching dashboard on port {port}…")
    subprocess.run(cmd, check=False)


@cli.command("validate-bids")
@click.option("--bids", required=True, type=click.Path(exists=True), help="BIDS root directory.")
def validate_bids(bids: str) -> None:
    from qc_toolbox.core.bids_loader import BIDSLoader

    loader = BIDSLoader(bids)
    entries = loader.discover_subjects()

    if not entries:
        click.echo("❌ No ASL files found.", err=True)
        sys.exit(1)

    click.echo(f"✅ Found {len(entries)} ASL acquisitions:")
    for nifti, sub, ses in entries:
        ses_str = f" / {ses}" if ses else ""
        click.echo(f"  {sub}{ses_str}")

        parent = nifti.parent
        stem = nifti.name.replace("_asl.nii.gz", "")
        ctx = parent / f"{stem}_aslcontext.tsv"
        js = parent / f"{stem}_asl.json"

        issues = []
        if not ctx.exists():
            issues.append("missing aslcontext.tsv")
        if not js.exists():
            issues.append("missing asl.json")
        if issues:
            click.echo(f"    ⚠ {', '.join(issues)}")
        else:
            click.echo("    ✓ sidecar files present")
