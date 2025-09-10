#!/usr/bin/env python3
"""
Canonical CLI entrypoint for the NEDIS vectorized synthetic data pipeline.

This thin wrapper delegates to the legacy script module to preserve behavior
while providing a clearer, stable command name.
"""

import sys
from pathlib import Path

# Ensure project root and src are on the path (match legacy behavior)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from scripts import run_vectorized_pipeline as _pipeline  # type: ignore


def main() -> int:
    return _pipeline.main()


if __name__ == "__main__":
    raise SystemExit(main())

