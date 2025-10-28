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

The CLI always works with a PID list. To analyse a single species, pass a single PID value; for multiple species, list them all. Both entry points (`python -m dedx_analysis` and the installed `dedx-analysis`) accept the same arguments.

Single-species example (protons only):

```bash
python -m dedx_analysis \
  --input-file combined.root \
  --pid-list 2212 \
  --analysis-momentum-range 0.5 3.0 \
  --band-output-dir bands \
  --combined-band-plot bands/dedx_band_2212.png
```

π/K/p batch run with evaluation and a shared overlay:

```bash
dedx-analysis \
  --input-file calotrkana-1M.root \
  --pid-list 211 321 2212 \
  --momentum-range 0.1 3.0 \
  --analysis-momentum-range 0.1 3.0 \
  --band-output-dir bands-0.4-3.0-1M \
  --combined-band-plot bands/dedx_bands_all.png \
  --evaluation-output-dir evaluation
```

Every run reports which PID bands are generated vs. reused, writes the individual CSV/PNG outputs to `--band-output-dir`, optionally produces a combined plot, and (unless `--skip-evaluation` is supplied) creates per-species efficiency/purity curves under `--evaluation-output-dir`. Provide `--evaluation-file` to compare against a different ROOT file.

Control how aggressively the models avoid the charge-sign crossover by providing `--momentum-gap MIN MAX` (default `-0.2 0.2`). Restrict the signed momentum used for the fits via `--analysis-momentum-range MIN MAX` (default `-0.3 0.3`) and cap the number of selected events with `--max-events N` (default `0`, meaning all events). Use `--no-progress` to disable the CLI progress bar.

If band CSV/PNG outputs already exist in `--band-output-dir`, reruns will reuse them; pass `--force-regenerate` to rebuild the bands from scratch.

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

## cheatsheet


10/27
```bash
dedx-analysis \
  --input-file calotrkana-1M.root \
  --pid-list 211 321 2212 \
  --momentum-range 0.4 3.0 \
  --analysis-momentum-range 0.4 3.0 \
  --band-output-dir bands-0.4-3.0-1M \
  --combined-band-plot bands-0.4-3.0-1M/dedx_bands_all.png \
  --evaluation-output-dir evaluation-0.4-3.0-1M \
  --prior-distribution-dir priors-0.4-3.0-1M
```