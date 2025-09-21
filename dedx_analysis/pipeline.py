"""Pipeline helpers for running multi-PID dE/dx analysis and evaluations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from .analysis import (
    DedxAnalysisConfig,
    DedxAnalysisError,
    load_root_branches,
    run_analysis,
)


@dataclass(slots=True)
class PIDBandResult:
    """Summary of files generated for a single PID band."""

    pid: int
    csv_path: Path
    plot_path: Path


@dataclass(slots=True)
class SpeciesEvaluationResult:
    """Paths to evaluation artefacts for a specific PID species."""

    pid: int
    csv_path: Path
    plot_path: Path


class _BandModel:
    """Interpolation helper around a momentum-dependent mean/sigma band."""

    def __init__(self, momentum: np.ndarray, mean: np.ndarray, sigma: np.ndarray) -> None:
        order = np.argsort(momentum)
        self._momentum = momentum[order]
        self._mean = mean[order]
        self._sigma = np.where(sigma[order] > 0, sigma[order], np.nan)

    def evaluate(self, momentum_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        mean = np.interp(
            momentum_values,
            self._momentum,
            self._mean,
            left=np.nan,
            right=np.nan,
        )
        sigma = np.interp(
            momentum_values,
            self._momentum,
            self._sigma,
            left=np.nan,
            right=np.nan,
        )
        sigma = np.where(sigma > 0, sigma, np.nan)
        return mean, sigma


def generate_pid_bands(
    *,
    input_file: str,
    pids: Sequence[int],
    tree_name: str,
    dedx_branch: str,
    momentum_branch: str,
    pid_branch: str,
    dedx_max: float,
    momentum_bins: int,
    dedx_bins: int,
    momentum_range: tuple[float, float],
    dedx_range: tuple[float, float],
    momentum_gap: tuple[float, float] = (-0.1, 0.1),
    analysis_momentum_range: Optional[tuple[float, float]] = (-0.3, 0.3),
    max_events: int = 0,
    show_progress: bool = True,
    output_dir: str | Path,
    band_prefix: str,
    sample_size: int | None,
    random_seed: int | None,
) -> list[PIDBandResult]:
    """Run the single-PID analysis for multiple PIDs in sequence."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[PIDBandResult] = []
    for pid in pids:
        csv_path = output_path / f"{band_prefix}_{abs(pid)}.csv"
        plot_path = output_path / f"{band_prefix}_{abs(pid)}.png"
        config = DedxAnalysisConfig(
            input_file=input_file,
            pid=pid,
            tree_name=tree_name,
            dedx_branch=dedx_branch,
            momentum_branch=momentum_branch,
            pid_branch=pid_branch,
            dedx_max=dedx_max,
            momentum_bins=momentum_bins,
            dedx_bins=dedx_bins,
            momentum_range=momentum_range,
            dedx_range=dedx_range,
            momentum_gap=momentum_gap,
            analysis_momentum_range=analysis_momentum_range,
            max_events=max_events,
            show_progress=show_progress,
            output_csv=str(csv_path),
            output_plot=str(plot_path),
            sample_size=sample_size,
            random_seed=random_seed,
        )
        run_analysis(config)
        results.append(
            PIDBandResult(
                pid=abs(pid),
                csv_path=csv_path.resolve(),
                plot_path=plot_path.resolve(),
            )
        )

    return results


def evaluate_pid_bands(
    band_results: Sequence[PIDBandResult],
    *,
    evaluation_file: str,
    tree_name: str,
    dedx_branch: str,
    momentum_branch: str,
    pid_branch: str,
    dedx_max: float,
    momentum_range: tuple[float, float],
    momentum_bins: int,
    output_dir: str | Path,
    figure_prefix: str,
) -> list[SpeciesEvaluationResult]:
    """Compare band predictions against truth and write efficiency/purity curves."""

    if not band_results:
        raise DedxAnalysisError("No band results provided for evaluation")

    species = sorted({result.pid for result in band_results})
    band_models = _load_band_models(band_results)

    arrays = load_root_branches(
        input_file=evaluation_file,
        tree_name=tree_name,
        dedx_branch=dedx_branch,
        momentum_branch=momentum_branch,
        pid_branch=pid_branch,
    )

    dedx = arrays["dedx"]
    momentum = arrays["momentum"]
    pid = arrays["pid"]

    finite_mask = np.isfinite(dedx) & np.isfinite(momentum) & np.isfinite(pid)
    dedx_mask = dedx < dedx_max
    species_mask = np.isin(np.abs(pid), species)
    selection = finite_mask & dedx_mask & species_mask

    if not np.any(selection):
        raise DedxAnalysisError(
            "No entries survive the evaluation selection. Check PID list or thresholds."
        )

    dedx_sel = dedx[selection]
    momentum_sel = momentum[selection]
    pid_sel = pid[selection]
    p_signed = momentum_sel * np.sign(pid_sel)

    score_matrix = _compute_score_matrix(p_signed, dedx_sel, species, band_models)
    valid_scores_mask = np.isfinite(score_matrix).any(axis=0)

    with np.errstate(invalid="ignore"):
        score_matrix = np.where(np.isfinite(score_matrix), score_matrix, -np.inf)

    predicted_indices = score_matrix.argmax(axis=0)
    predicted_indices = np.where(valid_scores_mask, predicted_indices, -1)

    predicted_species = np.where(
        predicted_indices >= 0,
        np.take(np.array(species), predicted_indices, mode="clip"),
        -1,
    )

    bin_edges = np.linspace(momentum_range[0], momentum_range[1], momentum_bins + 1)
    bin_indices = np.digitize(p_signed, bin_edges) - 1
    valid_bins_mask = (bin_indices >= 0) & (bin_indices < momentum_bins)
    bin_indices = np.where(valid_bins_mask, bin_indices, -1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    momentum_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    results: list[SpeciesEvaluationResult] = []

    for pid_value in species:
        metrics_df = _accumulate_metrics(
            pid_value,
            bin_indices,
            momentum_centers,
            pid_sel,
            predicted_species,
        )

        csv_path = output_path / f"{figure_prefix}_{pid_value}_metrics.csv"
        plot_path = output_path / f"{figure_prefix}_{pid_value}_metrics.png"
        metrics_df.to_csv(csv_path, index=False)

        _plot_metrics(
            momentum_centers,
            metrics_df["efficiency"].to_numpy(),
            metrics_df["purity"].to_numpy(),
            pid_value,
            plot_path,
        )

        results.append(
            SpeciesEvaluationResult(
                pid=pid_value,
                csv_path=csv_path.resolve(),
                plot_path=plot_path.resolve(),
            )
        )

    return results


def plot_combined_bands(
    band_results: Sequence[PIDBandResult],
    *,
    output_path: str | Path,
    title: str | None = None,
) -> Path:
    """Overlay multiple PID bands on the same figure."""

    if not band_results:
        raise DedxAnalysisError("No band results provided for combined plotting")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    for index, result in enumerate(sorted(band_results, key=lambda item: item.pid)):
        df = pd.read_csv(result.csv_path)
        required_columns = {"momentum", "dedx_mean", "dedx_sigma"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise DedxAnalysisError(
                f"Band file '{result.csv_path}' missing columns: {', '.join(sorted(missing))}"
            )

        momentum = df["momentum"].to_numpy()
        mean = df["dedx_mean"].to_numpy()
        sigma = df["dedx_sigma"].to_numpy()

        valid = np.isfinite(momentum) & np.isfinite(mean) & np.isfinite(sigma)
        if not np.any(valid):
            continue

        momentum = momentum[valid]
        mean = mean[valid]
        sigma = np.where(sigma[valid] > 0, sigma[valid], np.nan)

        order = np.argsort(momentum)
        momentum = momentum[order]
        mean = mean[order]
        sigma = sigma[order]

        lower = mean - sigma
        upper = mean + sigma

        label = f"PID {result.pid}"
        ax.plot(momentum, mean, label=label)
        with np.errstate(invalid="ignore"):
            band_mask = np.isfinite(lower) & np.isfinite(upper)
            if np.any(band_mask):
                ax.fill_between(
                    momentum[band_mask],
                    lower[band_mask],
                    upper[band_mask],
                    alpha=0.2,
                )

    ax.set_xlabel("Momentum × charge [GeV/c]")
    ax.set_ylabel("dE/dx")
    ax.grid(True, alpha=0.3)
    ax.legend()
    if title:
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path.resolve()


def _load_band_models(band_results: Iterable[PIDBandResult]) -> dict[int, _BandModel]:
    models: dict[int, _BandModel] = {}
    for result in band_results:
        df = pd.read_csv(result.csv_path)
        required_columns = {"momentum", "dedx_mean", "dedx_sigma"}
        missing = required_columns.difference(df.columns)
        if missing:
            raise DedxAnalysisError(
                f"Band file '{result.csv_path}' missing columns: {', '.join(sorted(missing))}"
            )
        models[result.pid] = _BandModel(
            momentum=df["momentum"].to_numpy(),
            mean=df["dedx_mean"].to_numpy(),
            sigma=df["dedx_sigma"].to_numpy(),
        )
    return models


def _compute_score_matrix(
    p_signed: np.ndarray,
    dedx: np.ndarray,
    species: Sequence[int],
    band_models: dict[int, _BandModel],
) -> np.ndarray:
    scores = []
    for pid_value in species:
        model = band_models[pid_value]
        mean, sigma = model.evaluate(p_signed)
        with np.errstate(divide="ignore", invalid="ignore"):
            score = -np.abs((dedx - mean) / sigma)
        scores.append(score)
    return np.vstack(scores)


def _accumulate_metrics(
    pid_value: int,
    bin_indices: np.ndarray,
    momentum_centers: np.ndarray,
    pid_truth: np.ndarray,
    predicted_species: np.ndarray,
) -> pd.DataFrame:
    num_bins = momentum_centers.size

    truth_mask = (np.abs(pid_truth) == pid_value) & (bin_indices >= 0)
    pred_mask = (predicted_species == pid_value) & (bin_indices >= 0)
    correct_mask = truth_mask & pred_mask

    truth_counts = np.bincount(bin_indices[truth_mask], minlength=num_bins)
    predicted_counts = np.bincount(bin_indices[pred_mask], minlength=num_bins)
    correct_counts = np.bincount(bin_indices[correct_mask], minlength=num_bins)

    with np.errstate(divide="ignore", invalid="ignore"):
        efficiency = np.divide(
            correct_counts,
            truth_counts,
            out=np.zeros_like(correct_counts, dtype=float),
            where=truth_counts > 0,
        )
        purity = np.divide(
            correct_counts,
            predicted_counts,
            out=np.zeros_like(correct_counts, dtype=float),
            where=predicted_counts > 0,
        )

    data = {
        "momentum": momentum_centers,
        "truth_count": truth_counts,
        "predicted_count": predicted_counts,
        "correct_count": correct_counts,
        "efficiency": efficiency,
        "purity": purity,
    }
    return pd.DataFrame(data)


def _plot_metrics(
    momentum_centers: np.ndarray,
    efficiency: np.ndarray,
    purity: np.ndarray,
    pid_value: int,
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    axes[0].plot(momentum_centers, efficiency, color="tab:blue", label="Efficiency")
    axes[0].set_ylabel("Efficiency")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="lower right")

    axes[1].plot(momentum_centers, purity, color="tab:green", label="Purity")
    axes[1].set_ylabel("Purity")
    axes[1].set_xlabel("Momentum × charge [GeV/c]")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(f"PID {pid_value} performance")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
