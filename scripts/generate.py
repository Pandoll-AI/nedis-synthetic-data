#!/usr/bin/env python3
"""
NEDIS Synthetic Data Generator (canonical CLI)

Practical, single-entry command to generate vectorized synthetic data.
Provides detailed help and sensible defaults. Internally delegates to
the pipeline implementation in scripts/run_vectorized_pipeline.py.
"""

import sys
import argparse
from pathlib import Path

# Make sure project root and src are on the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from scripts.run_vectorized_pipeline import run_vectorized_pipeline  # type: ignore


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate.py",
        description=(
            "Generate vectorized synthetic NEDIS data.\n\n"
            "By default, learns distributions from an original table (auto-detected),\n"
            "assigns dates/times for the requested year, and writes results into the\n"
            "target database under schema 'nedis_synthetic'.\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Minimal (learn from main.nedis2017 if present)\n"
            "  python scripts/generate.py --database nedis_synth_2018.duckdb --year 2018\n\n"
            "  # Learn from specific DB and use its original size for synthetic records\n"
            "  python scripts/generate.py --source-database nedis_data.duckdb \\\n"
            "      --database nedis_synth_2018.duckdb --year 2018 --use-original-size\n\n"
            "  # Enable overflow redistribution (requires nedis_meta.hospital_capacity)\n"
            "  python scripts/generate.py --enable-overflow-redistribution \\\n"
            "      --overflow-redistribution-method same_region_first\n"
        ),
    )

    # IO and scale
    p.add_argument(
        "--database",
        default="nedis_synth.duckdb",
        help="Target DuckDB file to write synthetic tables (default: nedis_synth.duckdb)",
    )
    p.add_argument(
        "--source-database",
        default=None,
        help=(
            "Optional DuckDB file to read original distributions from (ATTACH as read-only).\n"
            "If omitted, the script scans current connection schemas for nedis{YEAR}.")
    )
    p.add_argument(
        "--use-original-size",
        action="store_true",
        help=(
            "If set, total synthetic records match the original table row count.\n"
            "Otherwise, --total-records (or default) is used.")
    )
    p.add_argument(
        "--total-records",
        type=int,
        default=322_573,
        help="Total synthetic records to generate when not using --use-original-size (default: 322,573)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Batch size for chunked processing (default: 100,000)",
    )
    p.add_argument(
        "--memory-efficient",
        action="store_true",
        default=True,
        help="Enable chunked generation to reduce memory use (default: enabled)",
    )
    p.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )

    # Temporal
    p.add_argument(
        "--year",
        type=int,
        default=2018,
        help=(
            "Target year for assigned dates/times (default: 2018).\n"
            "If no original table for that year exists, patterns fall back to a detected year (e.g., 2017)"
        ),
    )
    p.add_argument(
        "--time-resolution",
        choices=["daily", "hourly"],
        default="hourly",
        help="Time assignment resolution (default: hourly)",
    )
    p.add_argument("--preserve-seasonality", action="store_true", default=True, help="Preserve monthly seasonal patterns (default: on)")
    p.add_argument("--preserve-weekly-pattern", action="store_true", default=True, help="Preserve weekday patterns (default: on)")
    p.add_argument("--preserve-holiday-effects", action="store_true", default=True, help="Preserve holiday effects (default: on)")

    # Capacity (post-processing)
    p.add_argument(
        "--enable-overflow-redistribution",
        action="store_true",
        default=False,
        help=(
            "Enable capacity overflow redistribution (default: disabled).\n"
            "Requires nedis_meta.hospital_capacity; otherwise will be skipped gracefully."),
    )
    p.add_argument(
        "--overflow-redistribution-method",
        choices=["random_available", "same_region_first", "second_choice_probability"],
        default="same_region_first",
        help="Overflow redistribution strategy when enabled (default: same_region_first)",
    )
    p.add_argument("--base-capacity-multiplier", type=float, default=1.0, help="Capacity baseline multiplier (default: 1.0)")
    p.add_argument("--weekend-capacity-multiplier", type=float, default=0.8, help="Weekend capacity multiplier (default: 0.8)")
    p.add_argument("--holiday-capacity-multiplier", type=float, default=0.7, help="Holiday capacity multiplier (default: 0.7)")
    p.add_argument("--safety-margin", type=float, default=1.2, help="Safety margin for daily capacity limits (default: 1.2)")

    # Validation / reporting
    p.add_argument("--validate-temporal", action="store_true", help="Validate temporal assignment and log summary")
    p.add_argument("--generate-capacity-report", action="store_true", help="Generate capacity processing report")
    p.add_argument("--quality-gate-threshold", type=float, default=0.0, help="Fail pipeline if combined quality score is below this (default: 0.0)")
    p.add_argument("--clean-existing-data", action="store_true", help="Clean existing rows in nedis_synthetic.* tables before writing")

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    ok = run_vectorized_pipeline(args)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

