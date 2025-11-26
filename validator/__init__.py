"""
NEDIS Synthetic Data Validation Suite

A comprehensive validation platform for synthetic medical data quality assessment.

This package provides advanced statistical validation, pattern analysis,
clinical validation, and temporal analysis capabilities for evaluating
synthetic data quality against original datasets.

Main Features:
- Statistical distribution validation (KS test, Chi-square, correlation analysis)
- Dynamic pattern discovery with hierarchical fallback
- Clinical data validation and medical protocol compliance
- Temporal pattern analysis and time series validation
- Interactive web dashboard and automated report generation
- REST API and real-time validation capabilities
- Supabase integration for cloud-based validation
- tRPC integration for type-safe API calls

Example Usage:
    # Basic validation
    from validator import validate

    result = validate(
        original_db="original.duckdb",
        synthetic_db="synthetic.duckdb",
        validation_type="comprehensive"
    )

    print(f"Validation score: {result.overall_score}")

    # Advanced validation with custom config
    from validator.core import ValidationOrchestrator, ValidationConfig

    config = ValidationConfig.from_yaml("validation_config.yaml")
    orchestrator = ValidationOrchestrator(config)

    result = orchestrator.validate_comprehensive(
        "original.duckdb",
        "synthetic.duckdb",
        sample_size=100000
    )

For more information, see the README.md file.
"""

from .core.validator import ValidationOrchestrator, validate, validate_async
from .core.config import ValidationConfig, get_config
from .core.database import DatabaseManager, get_database_manager
from .utils.logging import setup_logging, get_logger
from .utils.metrics import get_performance_tracker

__version__ = "2.0.0"
__author__ = "NEDIS Validation Team"
__description__ = "Advanced validation platform for synthetic medical data"

__all__ = [
    # Main validation functions
    'validate',
    'validate_async',
    'ValidationOrchestrator',

    # Configuration
    'ValidationConfig',
    'get_config',

    # Database management
    'DatabaseManager',
    'get_database_manager',

    # Utilities
    'setup_logging',
    'get_logger',
    'get_performance_tracker',

    # Version info
    '__version__',
    '__author__',
    '__description__'
]
