"""Command line entry point for the dedx analysis."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .analysis import DedxAnalysisConfig, DedxAnalysisError, run_analysis


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract mean and sigma of TPC dE/dx vs momentum")
    parser.add_argument("--input-file", required=True, help="Path to the input ROOT file")
    parser.add_argument("--pid", required=True, type=int, help="Absolute PID value to select (e.g. 211)")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))

    sample_size = args.sample_size if args.sample_size and args.sample_size > 0 else None

    config = DedxAnalysisConfig(
        input_file=args.input_file,
        pid=args.pid,
        tree_name=args.tree_name,
        dedx_branch=args.dedx_branch,
        momentum_branch=args.momentum_branch,
        pid_branch=args.pid_branch,
        dedx_max=args.dedx_max,
        momentum_bins=args.momentum_bins,
        dedx_bins=args.dedx_bins,
        momentum_range=tuple(args.momentum_range),
        dedx_range=tuple(args.dedx_range),
        output_csv=args.output_csv,
        output_plot=args.output_plot,
        sample_size=sample_size,
        random_seed=args.random_seed,
    )

    try:
        results = run_analysis(config)
    except DedxAnalysisError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    summary_path = Path(results["csv_path"]).resolve()
    plot_path = Path(results["plot_path"]).resolve()

    print("Analysis complete")
    print(f"CSV summary written to: {summary_path}")
    print(f"Plot saved to: {plot_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
