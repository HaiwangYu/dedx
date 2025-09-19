"""Module entry point enabling `python -m dedx_analysis`."""
from .cli import main


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
