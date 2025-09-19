from pathlib import Path

from setuptools import find_packages, setup

README = Path(__file__).with_name("README.md")

setup(
    name="dedx-analysis",
    version="0.1.0",
    description="Extract mean and sigma of TPC dE/dx vs momentum using Gaussian Process Regression",
    long_description=README.read_text(encoding="utf-8") if README.exists() else "",
    long_description_content_type="text/markdown",
    author="Codex Assistant",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy",
        "pandas",
        "awkward>=2.0",
        "uproot",
        "matplotlib",
        "scikit-learn",
    ],
    entry_points={
        "console_scripts": [
            "dedx-analysis=dedx_analysis.cli:main",
        ],
    },
)
