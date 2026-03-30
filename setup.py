from setuptools import setup, find_packages
from pathlib import Path

long_description = ""
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

setup(
    name="qc-toolbox",
    version="1.0.0",
    description="Quality Check Toolbox for Arterial Spin Labeling (ASL) MRI data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="OSIPI — Open Science Initiative for Perfusion Imaging",
    author_email="osipi@ismrm.org",
    url="https://github.com/OSIPI/qc-toolbox",
    license="Apache-2.0",
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={
        "qc_toolbox": ["thresholds/profiles/*.json"],
    },
    python_requires=">=3.9",
    install_requires=[
        "nibabel>=4.0.0",
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.3.0",
        "scikit-image>=0.21.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "plotly>=5.15.0",
        "streamlit>=1.28.0",
        "click>=8.1.0",
        "tqdm>=4.65.0",
        "dipy>=1.7.0",
        "nilearn>=0.10.0",
        "seaborn>=0.12.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "qc-toolbox=qc_toolbox.cli:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
)
