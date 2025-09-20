"""Core analysis utilities for dedx vs momentum extraction."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import uproot
import awkward as ak
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler


@dataclass(slots=True)
class DedxAnalysisConfig:
    """Configuration for the dedx analysis workflow."""

    input_file: str
    pid: int
    tree_name: str = "T"
    dedx_branch: str = "tpc_seeds_dedx"
    momentum_branch: str = "tpc_seeds_maxparticle_p"
    pid_branch: str = "tpc_seeds_maxparticle_pid"
    dedx_max: float = 1000.0
    momentum_bins: int = 200
    dedx_bins: int = 400
    momentum_range: Tuple[float, float] = (-3.0, 3.0)
    dedx_range: Tuple[float, float] = (0.0, 1000.0)
    momentum_gap: Tuple[float, float] = (-0.2, 0.2)
    output_csv: str = "dedx_summary.csv"
    output_plot: str = "dedx_summary.png"
    sample_size: Optional[int] = 5000
    random_seed: Optional[int] = 42


class DedxAnalysisError(RuntimeError):
    """Raised when the analysis cannot be completed."""


def run_analysis(config: DedxAnalysisConfig) -> dict:
    """Execute the dedx analysis pipeline.

    Returns
    -------
    dict
        Collected results including histogram data and model outputs.
    """

    data = _load_filtered_data(config)
    if data["p_signed"].size < 2:
        raise DedxAnalysisError("Not enough events after filtering to run GPR")

    hist, momentum_edges, dedx_edges = _build_histogram(
        data["p_signed"],
        data["dedx"],
        bins=(config.momentum_bins, config.dedx_bins),
        ranges=(config.momentum_range, config.dedx_range),
    )

    momentum_centers = 0.5 * (momentum_edges[:-1] + momentum_edges[1:])

    means, sigmas = _fit_gaussian_process(
        data["p_signed"],
        data["dedx"],
        momentum_centers,
        sample_size=config.sample_size,
        random_seed=config.random_seed,
        momentum_gap=config.momentum_gap,
    )

    csv_path = _write_csv(config.output_csv, momentum_centers, means, sigmas)
    plot_path = _plot_results(
        hist,
        momentum_edges,
        dedx_edges,
        momentum_centers,
        means,
        sigmas,
        output_path=config.output_plot,
    )

    return {
        "histogram": hist,
        "momentum_edges": momentum_edges,
        "dedx_edges": dedx_edges,
        "momentum_centers": momentum_centers,
        "mean": means,
        "sigma": sigmas,
        "csv_path": csv_path,
        "plot_path": plot_path,
    }


def _load_filtered_data(config: DedxAnalysisConfig) -> dict:
    """Load ROOT data and apply selection cuts."""

    arrays = load_root_branches(
        input_file=config.input_file,
        tree_name=config.tree_name,
        dedx_branch=config.dedx_branch,
        momentum_branch=config.momentum_branch,
        pid_branch=config.pid_branch,
    )

    dedx = arrays["dedx"]
    momentum = arrays["momentum"]
    pid = arrays["pid"]

    pid_abs = np.abs(pid)
    pid_match = np.isclose(pid_abs, float(config.pid), atol=0.5)
    finite_mask = np.isfinite(dedx) & np.isfinite(momentum) & np.isfinite(pid)
    dedx_mask = dedx < config.dedx_max
    selection = pid_match & finite_mask & dedx_mask

    if not np.any(selection):
        raise DedxAnalysisError(
            "No entries survive the selection. Check PID or selection thresholds."
        )

    dedx_sel = dedx[selection]
    momentum_sel = momentum[selection]
    pid_sel = pid[selection]

    p_signed = momentum_sel * np.sign(pid_sel)

    return {"dedx": dedx_sel, "momentum": momentum_sel, "pid": pid_sel, "p_signed": p_signed}


def _ak_to_numpy(array: ak.Array) -> np.ndarray:
    """Convert an Awkward array to a plain NumPy array, preserving NaNs."""

    np_array = ak.to_numpy(array, allow_missing=True)
    if np.ma.isMaskedArray(np_array):
        np_array = np_array.filled(np.nan)
    return np.asarray(np_array)


def load_root_branches(
    *,
    input_file: str,
    tree_name: str,
    dedx_branch: str,
    momentum_branch: str,
    pid_branch: str,
) -> dict:
    """Load dedx, momentum, and pid branches from a ROOT file as NumPy arrays."""

    root_path = Path(input_file)
    if not root_path.exists():
        raise DedxAnalysisError(f"Input file not found: {root_path}")

    with uproot.open(root_path) as file:
        if tree_name not in file:
            raise DedxAnalysisError(
                f"Tree '{tree_name}' not found in file '{root_path.name}'"
            )
        tree = file[tree_name]
        arrays = tree.arrays(
            [dedx_branch, momentum_branch, pid_branch],
            library="ak",
        )

    dedx = arrays[dedx_branch]
    momentum = arrays[momentum_branch]
    pid = arrays[pid_branch]

    if getattr(dedx, "ndim", 1) > 1:
        dedx = ak.flatten(dedx, axis=None)
        momentum = ak.flatten(momentum, axis=None)
        pid = ak.flatten(pid, axis=None)

    return {
        "dedx": _ak_to_numpy(dedx),
        "momentum": _ak_to_numpy(momentum),
        "pid": _ak_to_numpy(pid),
    }


def _build_histogram(
    momentum: np.ndarray,
    dedx: np.ndarray,
    bins: Tuple[int, int],
    ranges: Tuple[Tuple[float, float], Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a 2D histogram of dedx vs signed momentum."""

    hist, momentum_edges, dedx_edges = np.histogram2d(
        momentum,
        dedx,
        bins=bins,
        range=ranges,
    )
    return hist, momentum_edges, dedx_edges


def _fit_gaussian_process(
    momentum: np.ndarray,
    dedx: np.ndarray,
    evaluation_points: np.ndarray,
    *,
    sample_size: Optional[int],
    random_seed: Optional[int],
    momentum_gap: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a Gaussian Process to dedx vs momentum data."""

    rng = np.random.default_rng(random_seed)
    x = momentum.reshape(-1, 1)
    y = dedx

    gap_min, gap_max = sorted(momentum_gap)
    mean = np.full_like(evaluation_points, np.nan, dtype=float)
    std = np.full_like(evaluation_points, np.nan, dtype=float)

    region_specs = (
        ("negative", lambda values: values <= gap_min),
        ("positive", lambda values: values >= gap_max),
    )

    for region_name, selector in region_specs:
        region_mask = selector(x[:, 0])
        sample_count = int(np.count_nonzero(region_mask))
        if sample_count < 2:
            raise DedxAnalysisError(
                "Not enough samples in the {region} momentum region after applying the gap "
                "({gap_min}, {gap_max}).".format(
                    region=region_name, gap_min=gap_min, gap_max=gap_max
                )
            )

        x_region = x[region_mask]
        y_region = y[region_mask]

        if sample_size is not None and sample_size < sample_count:
            indices = rng.choice(sample_count, size=sample_size, replace=False)
            x_region = x_region[indices]
            y_region = y_region[indices]

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x_region)
        kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
        gpr.fit(x_scaled, y_region)

        eval_mask = selector(evaluation_points)
        if not np.any(eval_mask):
            continue

        x_eval = evaluation_points[eval_mask].reshape(-1, 1)
        x_eval_scaled = scaler.transform(x_eval)
        region_mean, region_std = gpr.predict(x_eval_scaled, return_std=True)
        mean[eval_mask] = region_mean
        std[eval_mask] = region_std

    return mean, std


def _write_csv(
    output_path: str,
    momentum_centers: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
) -> Path:
    """Persist the momentum vs dedx summary as CSV."""

    df = pd.DataFrame(
        {
            "momentum": momentum_centers,
            "dedx_mean": means,
            "dedx_sigma": sigmas,
        }
    )
    csv_path = Path(output_path)
    df.to_csv(csv_path, index=False)
    return csv_path


def _plot_results(
    hist: np.ndarray,
    momentum_edges: np.ndarray,
    dedx_edges: np.ndarray,
    momentum_centers: np.ndarray,
    means: np.ndarray,
    sigmas: np.ndarray,
    *,
    output_path: str,
) -> Path:
    """Create a visualization combining the histogram and GP summary."""

    fig, ax = plt.subplots(figsize=(9, 6))
    mesh = ax.pcolormesh(
        momentum_edges,
        dedx_edges,
        hist.T,
        shading="auto",
        cmap="viridis",
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Counts")

    ax.plot(momentum_centers, means, color="tab:red", label="GP mean")
    ax.fill_between(
        momentum_centers,
        means - sigmas,
        means + sigmas,
        color="tab:red",
        alpha=0.25,
        label="GP ±1σ",
    )

    ax.set_xlabel("Signed momentum (GeV/c)")
    ax.set_ylabel("TPC dE/dx (arb. units)")
    ax.set_title("TPC dE/dx vs momentum")
    ax.legend(loc="best")

    fig.tight_layout()
    plot_path = Path(output_path)
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path
