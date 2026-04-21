"""Pipeline helpers for running multi-PID dE/dx analysis and evaluations."""
from __future__ import annotations

from dataclasses import dataclass, field
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
    roc_csv_path: Path
    roc_plot_path: Path
    roc_bin_csv_paths: list[Path] = field(default_factory=list)
    roc_bin_plot_path: Optional[Path] = None


@dataclass(slots=True)
class PriorDistributionResult:
    """Summary of files generated for a single PID prior distribution."""

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
        mean = np.interp(momentum_values, self._momentum, self._mean)
        sigma = np.interp(momentum_values, self._momentum, self._sigma)

        lo, hi = self._momentum[0], self._momentum[-1]

        left_mask = momentum_values < lo
        if np.any(left_mask):
            dm = self._momentum[1] - self._momentum[0]
            mean[left_mask] = self._mean[0] + (self._mean[1] - self._mean[0]) / dm * (momentum_values[left_mask] - lo)
            sigma[left_mask] = self._sigma[0] + (self._sigma[1] - self._sigma[0]) / dm * (momentum_values[left_mask] - lo)

        right_mask = momentum_values > hi
        if np.any(right_mask):
            dm = self._momentum[-1] - self._momentum[-2]
            mean[right_mask] = self._mean[-1] + (self._mean[-1] - self._mean[-2]) / dm * (momentum_values[right_mask] - hi)
            sigma[right_mask] = self._sigma[-1] + (self._sigma[-1] - self._sigma[-2]) / dm * (momentum_values[right_mask] - hi)

        sigma = np.where(sigma > 0, sigma, np.nan)
        return mean, sigma


class _PriorDistribution:
    """Piecewise-constant prior probability lookup across momentum bins."""

    def __init__(self, bin_edges: np.ndarray, probabilities: np.ndarray) -> None:
        self._bin_edges = bin_edges
        self._probabilities = probabilities

    def evaluate(self, momentum_values: np.ndarray) -> np.ndarray:
        indices = np.digitize(momentum_values, self._bin_edges) - 1
        valid = (indices >= 0) & (indices < self._probabilities.size)
        result = np.full(momentum_values.shape, np.nan, dtype=float)
        result[valid] = self._probabilities[indices[valid]]
        return result


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


def generate_prior_distributions(
    *,
    input_file: str,
    pids: Sequence[int],
    tree_name: str,
    dedx_branch: str,
    momentum_branch: str,
    pid_branch: str,
    dedx_max: float,
    momentum_range: tuple[float, float],
    momentum_bins: int,
    analysis_momentum_range: Optional[tuple[float, float]] = (-0.3, 0.3),
    max_events: int = 0,
    output_dir: str | Path,
) -> list[PriorDistributionResult]:
    """Generate momentum priors for multiple PIDs using ROOT inputs."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: list[PriorDistributionResult] = []
    arrays = load_root_branches(
        input_file=input_file,
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
    base_selection = finite_mask & dedx_mask

    p_signed_all = momentum * np.sign(pid)
    if analysis_momentum_range is not None:
        range_min, range_max = sorted(analysis_momentum_range)
        momentum_window = (p_signed_all >= range_min) & (p_signed_all <= range_max)
        base_selection &= momentum_window

    selected_indices = np.nonzero(base_selection)[0]
    if selected_indices.size == 0:
        raise DedxAnalysisError(
            "No entries available after base selection to build prior distributions"
        )
    if max_events and max_events > 0 and selected_indices.size > max_events:
        rng = np.random.default_rng()
        selected_indices = rng.choice(selected_indices, size=max_events, replace=False)

    selected_momentum = p_signed_all[selected_indices]
    selected_pid = pid[selected_indices]

    bin_edges = np.linspace(momentum_range[0], momentum_range[1], momentum_bins + 1)
    bin_lower = bin_edges[:-1]
    bin_upper = bin_edges[1:]
    bin_centers = 0.5 * (bin_lower + bin_upper)

    total_counts, _ = np.histogram(selected_momentum, bins=bin_edges)

    for pid_value in pids:
        pid_target = np.isclose(np.abs(selected_pid), float(pid_value), atol=0.5)
        if not np.any(pid_target):
            raise DedxAnalysisError(
                f"No entries available to build prior distribution for PID {pid_value}"
            )

        pid_momentum = selected_momentum[pid_target]
        counts, _ = np.histogram(pid_momentum, bins=bin_edges)
        with np.errstate(divide="ignore", invalid="ignore"):
            probabilities = np.divide(
                counts,
                total_counts,
                out=np.zeros_like(counts, dtype=float),
                where=total_counts > 0,
            )

        df = pd.DataFrame(
            {
                "momentum_bin_lower": bin_lower,
                "momentum_bin_upper": bin_upper,
                "momentum_center": bin_centers,
                "count": counts,
                "total_count": total_counts,
                "probability": probabilities,
            }
        )

        csv_path = output_path / f"prior_{pid_value}.csv"
        df.to_csv(csv_path, index=False)
        plot_path = output_path / f"prior_{pid_value}.png"
        _plot_prior_distribution(
            bin_lower=bin_lower,
            bin_upper=bin_upper,
            probabilities=probabilities,
            pid_value=pid_value,
            output_path=plot_path,
        )
        results.append(
            PriorDistributionResult(
                pid=pid_value,
                csv_path=csv_path.resolve(),
                plot_path=plot_path.resolve(),
            )
        )

    return results


def plot_prior_distribution_from_csv(
    *,
    pid: int,
    csv_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Render a prior distribution plot from an existing CSV file."""

    df = pd.read_csv(csv_path)
    required_columns = {
        "momentum_bin_lower",
        "momentum_bin_upper",
        "probability",
    }
    missing = required_columns.difference(df.columns)
    if missing:
        raise DedxAnalysisError(
            f"Prior file '{csv_path}' missing columns: {', '.join(sorted(missing))}"
        )

    bin_lower = df["momentum_bin_lower"].to_numpy()
    bin_upper = df["momentum_bin_upper"].to_numpy()
    probabilities = df["probability"].to_numpy()

    if (
        bin_lower.size != probabilities.size
        or bin_upper.size != probabilities.size
    ):
        raise DedxAnalysisError(
            f"Prior file '{csv_path}' has inconsistent column lengths"
        )

    plot_path = Path(output_path)
    _plot_prior_distribution(
        bin_lower=bin_lower,
        bin_upper=bin_upper,
        probabilities=probabilities,
        pid_value=pid,
        output_path=plot_path,
    )
    return plot_path.resolve()


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
    prior_results: Sequence[PriorDistributionResult] | None = None,
    force_sigma_one: bool = True,
    analysis_momentum_range: Optional[tuple[float, float]] = None,
    roc_momentum_bins: Optional[tuple[float, float, float]] = None,
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

    bin_edges = np.linspace(momentum_range[0], momentum_range[1], momentum_bins + 1)
    prior_models = (
        _load_prior_models(prior_results, bin_edges) if prior_results else {}
    )

    score_matrix = _compute_score_matrix(
        p_signed,
        dedx_sel,
        species,
        band_models,
        prior_models if prior_models else None,
        force_sigma_one=force_sigma_one,
    )
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

    score_frac_matrix = _compute_score_frac_matrix(
        p_signed,
        dedx_sel,
        species,
        band_models,
        prior_models if prior_models else None,
        force_sigma_one=force_sigma_one,
    )

    if analysis_momentum_range is not None:
        amr_min, amr_max = sorted(analysis_momentum_range)
        analysis_mask = (p_signed >= amr_min) & (p_signed <= amr_max)
    else:
        analysis_mask = np.ones(p_signed.size, dtype=bool)

    bin_indices = np.digitize(p_signed, bin_edges) - 1
    valid_bins_mask = (bin_indices >= 0) & (bin_indices < momentum_bins)
    bin_indices = np.where(valid_bins_mask, bin_indices, -1)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    momentum_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    results: list[SpeciesEvaluationResult] = []

    for pid_idx, pid_value in enumerate(species):
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

        thresholds, eff_roc, pur_roc = _compute_roc_curve(
            score_frac_matrix[pid_idx],
            pid_sel,
            pid_value,
            analysis_mask,
        )
        auc = _compute_auc(eff_roc, pur_roc)

        roc_csv_path = output_path / f"{figure_prefix}_{pid_value}_roc.csv"
        roc_plot_path = output_path / f"{figure_prefix}_{pid_value}_roc.png"

        pd.DataFrame(
            {
                "threshold": thresholds,
                "efficiency": eff_roc,
                "purity": pur_roc,
                "auc": auc,
            }
        ).to_csv(roc_csv_path, index=False)

        _plot_roc_curve(eff_roc, pur_roc, auc, pid_value, roc_plot_path, thresholds=thresholds)

        roc_bin_csv_paths: list[Path] = []
        roc_bin_plot_path: Optional[Path] = None

        if roc_momentum_bins is not None:
            start, stop, step = roc_momentum_bins
            edges = np.arange(start, stop + step * 0.5, step)
            sub_ranges = [(float(edges[i]), float(edges[i + 1])) for i in range(len(edges) - 1)]

            roc_bin_data: list[tuple[np.ndarray, np.ndarray, float, str]] = []
            for lo, hi in sub_ranges:
                bin_mask = (p_signed >= lo) & (p_signed < hi)
                th_b, eff_b, pur_b = _compute_roc_curve(
                    score_frac_matrix[pid_idx], pid_sel, pid_value, bin_mask
                )
                auc_b = _compute_auc(eff_b, pur_b)
                label = f"p \u2208 [{lo:.1f}, {hi:.1f})"
                bin_csv = output_path / f"{figure_prefix}_{pid_value}_roc_{lo:.1f}_{hi:.1f}.csv"
                pd.DataFrame(
                    {"threshold": th_b, "efficiency": eff_b, "purity": pur_b, "auc": auc_b}
                ).to_csv(bin_csv, index=False)
                roc_bin_csv_paths.append(bin_csv.resolve())
                roc_bin_data.append((eff_b, pur_b, auc_b, label))

            _roc_bin_plot = output_path / f"{figure_prefix}_{pid_value}_roc_bins.png"
            _plot_roc_curves_multi(roc_bin_data, pid_value, _roc_bin_plot)
            roc_bin_plot_path = _roc_bin_plot.resolve()

        results.append(
            SpeciesEvaluationResult(
                pid=pid_value,
                csv_path=csv_path.resolve(),
                plot_path=plot_path.resolve(),
                roc_csv_path=roc_csv_path.resolve(),
                roc_plot_path=roc_plot_path.resolve(),
                roc_bin_csv_paths=roc_bin_csv_paths,
                roc_bin_plot_path=roc_bin_plot_path,
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
    prior_models: Optional[dict[int, _PriorDistribution]] = None,
    force_sigma_one: bool = True,
) -> np.ndarray:
    scores = []
    for pid_value in species:
        model = band_models[pid_value]
        mean, sigma = model.evaluate(p_signed)
        if force_sigma_one:
            sigma = np.ones_like(sigma, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = np.abs((dedx - mean) / sigma)

        if prior_models and pid_value in prior_models:
            prior = prior_models[pid_value].evaluate(p_signed)
            prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                score = np.divide(
                    prior,
                    deviation,
                    out=np.full_like(
                        deviation,
                        np.finfo(float).max,
                        dtype=float,
                    ),
                    where=deviation != 0,
                )
        else:
            score = -deviation
        scores.append(score)
    return np.vstack(scores)


def _compute_score_frac_matrix(
    p_signed: np.ndarray,
    dedx: np.ndarray,
    species: Sequence[int],
    band_models: dict[int, _BandModel],
    prior_models: Optional[dict[int, _PriorDistribution]] = None,
    force_sigma_one: bool = True,
) -> np.ndarray:
    """Compute Gaussian-likelihood fractional scores, shape (n_species, n_tracks).

    Each column sums to 1; values represent the fraction of total likelihood
    attributed to each PID hypothesis.
    """
    likelihoods = []
    for pid_value in species:
        model = band_models[pid_value]
        mean, sigma = model.evaluate(p_signed)
        if force_sigma_one:
            sigma = np.ones_like(sigma, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            deviation = np.abs((dedx - mean) / sigma)
        likelihood = np.exp(-0.5 * deviation ** 2)
        likelihood = np.nan_to_num(likelihood, nan=0.0, posinf=0.0, neginf=0.0)
        if prior_models and pid_value in prior_models:
            prior = prior_models[pid_value].evaluate(p_signed)
            prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
            likelihood = likelihood * prior
        likelihoods.append(likelihood)

    score_matrix = np.vstack(likelihoods)  # (n_species, n_tracks)
    total = score_matrix.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        score_frac = np.where(total > 0, score_matrix / total, 1.0 / len(species))
    return score_frac


def _compute_roc_curve(
    score_frac: np.ndarray,
    pid_truth: np.ndarray,
    pid_value: int,
    analysis_mask: np.ndarray,
    max_thresholds: int = 1000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sweep score_frac thresholds and return (thresholds, efficiency, purity).

    Thresholds are sampled from the sorted unique score values within
    analysis_mask (at most max_thresholds points), giving smooth curves
    without a fixed-grid staircase effect.
    Efficiency and purity are integrated over the tracks selected by analysis_mask.
    """
    truth_mask = (np.abs(pid_truth) == pid_value) & analysis_mask
    total_truth = int(np.sum(truth_mask))

    unique_scores = np.unique(score_frac[analysis_mask])
    if len(unique_scores) > max_thresholds:
        idx = np.linspace(0, len(unique_scores) - 1, max_thresholds, dtype=int)
        unique_scores = unique_scores[idx]
    thresholds = np.concatenate([[0.0], unique_scores, [1.0]])
    n = len(thresholds)
    efficiencies = np.zeros(n)
    purities = np.zeros(n)

    for i, th in enumerate(thresholds):
        if th == 0.0:
            efficiencies[i] = 1.0
            purities[i] = 0.0
            continue
        if th == 1.0:
            efficiencies[i] = 0.0
            purities[i] = 1.0
            continue
        pred_mask = (score_frac > th) & analysis_mask
        correct_mask = truth_mask & pred_mask
        total_predicted = int(np.sum(pred_mask))
        total_correct = int(np.sum(correct_mask))
        if total_truth > 0:
            efficiencies[i] = total_correct / total_truth
        if total_predicted > 0:
            purities[i] = total_correct / total_predicted

    return thresholds, efficiencies, purities


def _compute_auc(efficiency: np.ndarray, purity: np.ndarray) -> float:
    """Area under the efficiency-purity curve via the trapezoidal rule."""
    order = np.argsort(purity)
    return float(np.trapz(efficiency[order], purity[order]))


def _plot_roc_curve(
    efficiency: np.ndarray,
    purity: np.ndarray,
    auc: float,
    pid_value: int,
    output_path: Path,
    thresholds: Optional[np.ndarray] = None,
) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    if thresholds is not None:
        sc = ax.scatter(purity, efficiency, c=thresholds, cmap="viridis", s=20, zorder=3)
        cbar = fig.colorbar(sc, ax=ax)
        cbar.set_label("score_frac threshold")
    else:
        ax.scatter(purity, efficiency, s=20, zorder=3)
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"PID {pid_value} ROC curve  (AUC = {auc:.3f})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _plot_roc_curves_multi(
    roc_data: list[tuple[np.ndarray, np.ndarray, float, str]],
    pid_value: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(roc_data)))
    for (eff, pur, auc, label), color in zip(roc_data, colors):
        ax.plot(pur, eff, label=f"{label}  AUC={auc:.3f}", color=color, linewidth=1.5)
    ax.set_xlabel("Purity")
    ax.set_ylabel("Efficiency")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"PID {pid_value} ROC curves by momentum bin")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)


def _load_prior_models(
    prior_results: Sequence[PriorDistributionResult] | None,
    expected_bin_edges: np.ndarray,
) -> dict[int, _PriorDistribution]:
    if not prior_results:
        return {}

    models: dict[int, _PriorDistribution] = {}
    for result in prior_results:
        df = pd.read_csv(result.csv_path)
        required_columns = {
            "momentum_bin_lower",
            "momentum_bin_upper",
            "probability",
        }
        missing = required_columns.difference(df.columns)
        if missing:
            raise DedxAnalysisError(
                f"Prior file '{result.csv_path}' missing columns: {', '.join(sorted(missing))}"
            )

        bin_lower = df["momentum_bin_lower"].to_numpy()
        bin_upper = df["momentum_bin_upper"].to_numpy()
        probabilities = df["probability"].to_numpy()

        if bin_lower.size != probabilities.size or bin_upper.size != probabilities.size:
            raise DedxAnalysisError(
                f"Prior file '{result.csv_path}' has inconsistent bin definitions"
            )

        reconstructed_edges = np.concatenate(
            (bin_lower, np.array([bin_upper[-1]], dtype=float))
        )
        if reconstructed_edges.size != expected_bin_edges.size or not np.allclose(
            reconstructed_edges, expected_bin_edges
        ):
            raise DedxAnalysisError(
                f"Prior file '{result.csv_path}' momentum bins do not match evaluation bins"
            )

        models[result.pid] = _PriorDistribution(
            bin_edges=expected_bin_edges,
            probabilities=probabilities,
        )
    return models


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


def _plot_prior_distribution(
    *,
    bin_lower: np.ndarray,
    bin_upper: np.ndarray,
    probabilities: np.ndarray,
    pid_value: int,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    bin_centers = 0.5 * (bin_lower + bin_upper)
    widths = bin_upper - bin_lower
    widths = np.where(widths > 0, widths, 1e-3)
    ax.bar(
        bin_centers,
        probabilities,
        width=widths,
        align="center",
        color="tab:purple",
        alpha=0.7,
    )
    ax.set_xlabel("Momentum × charge [GeV/c]")
    ax.set_ylabel("Prior probability")
    ax.set_title(f"PID {pid_value} prior distribution")
    ax.set_ylim(bottom=0.0)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
