"""Export all report figures as 300-dpi PNGs to ``reports/figures/``.

Usage::

    python scripts/export_figures.py              # all figures (includes model CV, ~20 s)
    python scripts/export_figures.py --no-models  # skip the churn-model figures
    python scripts/export_figures.py --dpi 150 --out path/to/dir
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config, ingestion, static_figures  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Run the CLI and return a process exit code."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--out", type=Path, default=config.FIGURES_DIR)
    parser.add_argument("--dpi", type=int, default=config.EXPORT_DPI)
    parser.add_argument("--no-models", action="store_true", help="Skip model figures")
    args = parser.parse_args(argv)

    result = ingestion.load_bundled_cached()
    if not result.is_valid or result.data is None:
        print(
            "Dataset missing or invalid; run python scripts/fetch_data.py first.", file=sys.stderr
        )
        return 1
    print(
        "Computing analytics, segmentation" + ("" if args.no_models else " and model CV") + " ..."
    )
    inputs = static_figures.prepare_inputs(result.data, with_models=not args.no_models)
    paths = static_figures.export_all(inputs, args.out, args.dpi)
    for path in paths:
        desc = static_figures.FIGURES[path.stem][1]
        print(f"  {path.name:<30} {desc}")
    print(f"Exported {len(paths)} figures at {args.dpi} dpi to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
