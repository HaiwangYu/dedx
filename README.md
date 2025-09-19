# dedx-analysis

This package loads TPC dE/dx samples from ROOT files and extracts the mean and spread of dE/dx versus momentum for a configurable particle species. The workflow mimics the ROOT `TTree::Draw` command used in the lab but is implemented entirely in Python using `uproot`, `numpy`, and scikit-learn's Gaussian Process Regression tools.

## Features

- Selection of tracks with configurable absolute PID and dE/dx cuts
- 2D histogram of signed momentum (`p * sign(pid)`) versus dE/dx
- Gaussian Process Regression to estimate the mean and uncertainty of dE/dx in each momentum bin
- CSV summary output and publication-ready visualization overlaying the GP band on the histogram

## Installation

```bash
python -m pip install .
```

## Usage

```bash
python -m dedx_analysis \
  --input-file combined.root \
  --pid 211 \
  --tree-name T \
  --output-csv pion_dedx.csv \
  --output-plot pion_dedx.png
```

Additional options allow control of histogram binning, ranges, and the subsampling used for the Gaussian Process fit. Run `python -m dedx_analysis --help` for the full set of arguments.

## Outputs

- `dedx_summary.csv`: columns `momentum`, `dedx_mean`, `dedx_sigma`
- `dedx_summary.png`: 2D histogram with GP mean and ±1σ band overlayed

## Development

The CLI entry point is defined in `dedx_analysis/cli.py`. The end-to-end workflow resides in `dedx_analysis/analysis.py`.
