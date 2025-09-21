"""Command line entry point for the dedx analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import DedxAnalysisError
from .pipeline import (
    PIDBandResult,
    evaluate_pid_bands,
    generate_pid_bands,
    plot_combined_bands,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean and sigma of TPC dE/dx vs momentum")
    parser.add_argument("--input-file", required=True, help="Path to the input ROOT file")
    parser.add_argument(
        "--pid-list",
        nargs="+",
        type=int,
        required=True,
        help="Absolute PID values to process (e.g. 211 321 2212)",
    )
    parser.add_argument("--tree-name", default="T", help="Name of the TTree inside the ROOT file")
    parser.add_argument("--dedx-branch", default="tpc_seeds_dedx", help="Branch name for dE/dx")
    parser.add_argument(
        "--momentum-branch",
        default="tpc_seeds_maxparticle_p",
        help="Branch name for momentum",
    )
    parser.add_argument(
        "--pid-branch",
        default="tpc_seeds_maxparticle_pid",
        help="Branch name for particle ID",
    )
    parser.add_argument(
        "--dedx-max",
        default=1000.0,
        type=float,
        help="Upper bound for dE/dx values to include",
    )
    parser.add_argument(
        "--momentum-bins",
        default=200,
        type=int,
        help="Number of bins along the momentum axis",
    )
    parser.add_argument(
        "--dedx-bins",
        default=400,
        type=int,
        help="Number of bins along the dE/dx axis",
    )
    parser.add_argument(
        "--momentum-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(-3.0, 3.0),
        help="Momentum range (GeV/c) for the histogram",
    )
    parser.add_argument(
        "--momentum-gap",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(-0.1, 0.1),
        help="Momentum × charge window to exclude from GPR fits",
    )
    parser.add_argument(
        "--analysis-momentum-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(-0.3, 0.3),
        help="Signed momentum window retained for analysis",
    )
    parser.add_argument(
        "--dedx-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(0.0, 1000.0),
        help="dE/dx range for the histogram",
    )
    parser.add_argument(
        "--output-csv",
        default="dedx_summary.csv",
        help="Path for the CSV summary output",
    )
    parser.add_argument(
        "--output-plot",
        default="dedx_summary.png",
        help="Path for the visualization output",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Maximum number of events to use after selection (0 means all)",
    )
    parser.add_argument(
        "--band-output-dir",
        default="bands",
        help="Directory to store multi-PID band outputs",
    )
    parser.add_argument(
        "--band-prefix",
        default="dedx_band",
        help="Filename prefix for generated band CSV/PNG files",
    )
    parser.add_argument(
        "--combined-band-plot",
        help="Path to write a single plot overlaying all generated bands",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Regenerate PID bands even if outputs already exist",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip evaluation step when running with --pid-list",
    )
    parser.add_argument(
        "--evaluation-file",
        help="ROOT file to use for evaluation (defaults to --input-file)",
    )
    parser.add_argument(
        "--evaluation-output-dir",
        default="evaluation",
        help="Directory for evaluation metrics",
    )
    parser.add_argument(
        "--evaluation-prefix",
        default="pid_performance",
        help="Filename prefix for evaluation artefacts",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5000,
        help="Number of samples to use for GPR (set <=0 to disable subsampling)",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for subsampling",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bars",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    sample_size = args.sample_size if args.sample_size and args.sample_size > 0 else None
    pid_values = [abs(pid) for pid in args.pid_list]

    band_dir = Path(args.band_output_dir)
    band_dir.mkdir(parents=True, exist_ok=True)

    band_results: list[PIDBandResult] = []
    missing_pids: list[int] = []
    cached_pids: list[int] = []

    for pid in pid_values:
        csv_path = band_dir / f"{args.band_prefix}_{pid}.csv"
        plot_path = band_dir / f"{args.band_prefix}_{pid}.png"

        if not args.force_regenerate and csv_path.exists() and plot_path.exists():
            band_results.append(
                PIDBandResult(
                    pid=pid,
                    csv_path=csv_path.resolve(),
                    plot_path=plot_path.resolve(),
                )
            )
            cached_pids.append(pid)
        else:
            missing_pids.append(pid)

    if missing_pids:
        if args.force_regenerate:
            print("Force regeneration requested; recomputing PID bands.")
        else:
            print(
                "Generating PID bands for missing PIDs: "
                + ", ".join(str(pid) for pid in missing_pids)
            )
        try:
            generated = generate_pid_bands(
                input_file=args.input_file,
                pids=missing_pids,
                tree_name=args.tree_name,
                dedx_branch=args.dedx_branch,
                momentum_branch=args.momentum_branch,
                pid_branch=args.pid_branch,
                dedx_max=args.dedx_max,
                momentum_bins=args.momentum_bins,
                dedx_bins=args.dedx_bins,
                momentum_range=tuple(args.momentum_range),
                dedx_range=tuple(args.dedx_range),
                momentum_gap=tuple(args.momentum_gap),
                analysis_momentum_range=tuple(args.analysis_momentum_range),
                max_events=args.max_events,
                show_progress=not args.no_progress,
                output_dir=args.band_output_dir,
                band_prefix=args.band_prefix,
                sample_size=sample_size,
                random_seed=args.random_seed,
            )
        except DedxAnalysisError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1

        band_results.extend(generated)
        print(
            "Band generation complete for PIDs: "
            + ", ".join(str(pid) for pid in missing_pids)
        )
    else:
        print("All requested PID bands already exist; skipping band generation.")

    if cached_pids and not args.force_regenerate:
        print(
            "Using existing band outputs for PIDs: "
            + ", ".join(str(pid) for pid in cached_pids)
        )

    if not band_results:
        print("Error: No PID band outputs are available.", file=sys.stderr)
        return 1

    band_results.sort(key=lambda result: result.pid)
    print("Available PID bands:")
    for result in band_results:
        print(f"  PID {result.pid}: CSV {result.csv_path}, plot {result.plot_path}")

    if args.combined_band_plot:
        try:
            combined_path = plot_combined_bands(
                band_results,
                output_path=args.combined_band_plot,
                title="dE/dx bands",
            )
        except DedxAnalysisError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        print(f"Combined band plot saved to: {combined_path}")
    else:
        print("Combined band plot skipped (no output path provided).")

    if args.skip_evaluation:
        print("Evaluation skipped (--skip-evaluation).")
        return 0

    evaluation_file = args.evaluation_file or args.input_file
    try:
        evaluation_results = evaluate_pid_bands(
            band_results,
            evaluation_file=evaluation_file,
            tree_name=args.tree_name,
            dedx_branch=args.dedx_branch,
            momentum_branch=args.momentum_branch,
            pid_branch=args.pid_branch,
            dedx_max=args.dedx_max,
            momentum_range=tuple(args.momentum_range),
            momentum_bins=args.momentum_bins,
            output_dir=args.evaluation_output_dir,
            figure_prefix=args.evaluation_prefix,
        )
    except DedxAnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("Evaluation complete")
    for result in evaluation_results:
        print(f"PID {result.pid}: metrics {result.csv_path}, plot {result.plot_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
