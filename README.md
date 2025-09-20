# dedx-analysis

This package loads TPC dE/dx samples from ROOT files and extracts the mean and spread of dE/dx versus momentum for configurable particle species. The workflow mimics the ROOT `TTree::Draw` command used in the lab but is implemented entirely in Python using `uproot`, `numpy`, and scikit-learn's Gaussian Process Regression tools.

## Features

- Selection of tracks with configurable absolute PID and dE/dx cuts
- 2D histogram of signed momentum (`p * sign(pid)`) versus dE/dx
- Gaussian Process Regression to estimate the mean and uncertainty of dE/dx in each momentum bin
- Batch processing for multiple species with per-PID and combined visualizations
- CSV summaries, individual GP band plots, and evaluation metrics (efficiency/purity)

## Installation

Create a virtual environment and install the package in editable mode while iterating:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Usage

Single-species band extraction:

```bash
python -m dedx_analysis \
  --input-file combined.root \
  --pid 211 \
  --tree-name T \
  --output-csv pion_dedx.csv \
  --output-plot pion_dedx.png
```

Batch processing for π/K/p plus evaluation and a combined plot:

```bash
dedx-analysis \
  --input-file combined.root \
  --pid-list 211 321 2212 \
  --band-output-dir bands \
  --combined-band-plot bands/dedx_bands_all.png \
  --evaluation-output-dir evaluation
```

This command writes individual band CSV/PNG pairs to `bands/`, overlays all species in `dedx_bands_all.png`, and generates per-species efficiency/purity curves in `evaluation/`. Add `--skip-evaluation` to omit the final comparison step or `--evaluation-file` to point at a different ROOT file.

Control how aggressively the models avoid the charge-sign crossover by providing `--momentum-gap MIN MAX` (default `-0.2 0.2`).

Run `dedx-analysis --help` for the full set of arguments.

## Outputs

- `dedx_summary.csv`: columns `momentum`, `dedx_mean`, `dedx_sigma`
- `dedx_summary.png`: 2D histogram with GP mean and ±1σ band overlayed
- `bands/dedx_band_<pid>.csv`: CSV summaries for each species when using `--pid-list`
- `bands/dedx_band_<pid>.png`: Individual GP band plots per species
- `bands/dedx_bands_all.png`: Optional multi-species overlay plot
- `evaluation/pid_performance_<pid>_metrics.(csv|png)`: Efficiency/purity curves per species

## Development

The CLI entry point is defined in `dedx_analysis/cli.py`. Core analysis utilities live in `dedx_analysis/analysis.py`, while multi-species orchestration and evaluation helpers are in `dedx_analysis/pipeline.py`.
