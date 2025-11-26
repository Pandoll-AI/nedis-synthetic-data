"""
Main validation orchestrator for NEDIS synthetic data validation.

This module orchestrates the entire validation pipeline including:
- Statistical validation
- Pattern analysis
- Clinical validation
- Temporal validation
- Report generation
- Performance monitoring
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from pathlib import Path
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .config import ValidationConfig, get_config
from .database import DatabaseManager, get_database_manager
from ..analysis.statistical import StatisticalValidator
from ..analysis.pattern import PatternAnalyzer
from ..analysis.clinical import ClinicalValidator
from ..analysis.temporal import TemporalValidator
# Import reports at runtime to avoid circular imports
from ..utils.metrics import PerformanceTracker


class ValidationResult:
    """Container for validation results"""

    def __init__(self, validation_type: str, overall_score: float = 0.0):
        self.validation_type = validation_type
        self.overall_score = overall_score
        self.start_time = datetime.now()
        self.end_time = None
        self.duration = None
        self.results = {}
        self.metadata = {}
        self.errors = []
        self.warnings = []

    def add_result(self, category: str, result: Dict[str, Any]):
        """Add result for specific category"""
        self.results[category] = result

    def add_metadata(self, key: str, value: Any):
        """Add metadata"""
        self.metadata[key] = value

    def add_error(self, error: str):
        """Add error message"""
        self.errors.append({
            'timestamp': datetime.now().isoformat(),
            'error': error
        })

    def add_warning(self, warning: str):
        """Add warning message"""
        self.warnings.append({
            'timestamp': datetime.now().isoformat(),
            'warning': warning
        })

    def finalize(self):
        """Finalize validation results"""
        self.end_time = datetime.now()
        self.duration = (self.end_time - self.start_time).total_seconds()

    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary"""
        return {
            'validation_type': self.validation_type,
            'overall_score': self.overall_score,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': self.duration,
            'results': self.results,
            'metadata': self.metadata,
            'errors': self.errors,
            'warnings': self.warnings
        }


class ValidationOrchestrator:
    """Main orchestrator for the validation system"""

    def __init__(self, config: Optional[ValidationConfig] = None,
                 db_manager: Optional[DatabaseManager] = None):
        """
        Initialize validation orchestrator

        Args:
            config: Validation configuration
            db_manager: Database manager instance
        """
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.config = config or get_config()

        # Database manager
        self.db_manager = db_manager or get_database_manager()

        # Analysis modules
        self.statistical_validator = StatisticalValidator(self.db_manager, self.config)
        self.pattern_analyzer = PatternAnalyzer(self.db_manager, self.config)
        self.clinical_validator = ClinicalValidator(self.db_manager, self.config)
        self.temporal_validator = TemporalValidator(self.db_manager, self.config)

        # Visualization - import at runtime to avoid circular imports
        from ..visualization.reports import ReportGenerator
        self.report_generator = ReportGenerator(self.config)

        # Performance tracking
        self.performance_tracker = PerformanceTracker()

        # Thread pool for concurrent validation
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.performance.get('max_concurrent_validations', 3),
            thread_name_prefix='validator'
        )

        # Validation results storage
        self._results_cache: Dict[str, ValidationResult] = {}

        self.logger.info("Validation orchestrator initialized")

    async def validate_comprehensive(self, original_db: str, synthetic_db: str,
                                   sample_size: Optional[int] = None) -> ValidationResult:
        """
        Perform comprehensive validation

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            sample_size: Sample size for validation

        Returns:
            ValidationResult object
        """
        validation_id = f"comprehensive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ValidationResult('comprehensive')

        self.logger.info(f"Starting comprehensive validation: {validation_id}")

        try:
            # Set sample size
            if sample_size is None:
                sample_size = self.config.sample_size

            # Add metadata
            result.add_metadata('original_database', original_db)
            result.add_metadata('synthetic_database', synthetic_db)
            result.add_metadata('sample_size', sample_size)

            # Run validations concurrently
            tasks = [
                self._validate_statistical_async(original_db, synthetic_db, sample_size, result),
                self._validate_patterns_async(original_db, synthetic_db, result),
                self._validate_clinical_async(original_db, synthetic_db, result),
                self._validate_temporal_async(original_db, synthetic_db, result)
            ]

            # Wait for all validations to complete
            await asyncio.gather(*tasks, return_exceptions=True)

            # Calculate overall score
            overall_score = self._calculate_overall_score(result.results)
            result.overall_score = overall_score

            # Generate report
            self._generate_validation_report(result)

            # Cache result
            self._results_cache[validation_id] = result

            result.finalize()
            self.logger.info(f"Comprehensive validation completed: {validation_id}, Score: {overall_score:.2f}")

        except Exception as e:
            error_msg = f"Comprehensive validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
            result.overall_score = 0.0

        return result

    def validate_statistical(self, original_db: str, synthetic_db: str,
                           sample_size: Optional[int] = None) -> ValidationResult:
        """
        Perform statistical validation only

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            sample_size: Sample size for validation

        Returns:
            ValidationResult object
        """
        validation_id = f"statistical_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ValidationResult('statistical')

        try:
            if sample_size is None:
                sample_size = self.config.sample_size

            # Perform statistical validation
            statistical_result = self.statistical_validator.validate_distributions(
                sample_size,
                original_db,
                synthetic_db
            )

            result.add_result('statistical', statistical_result)
            result.overall_score = statistical_result.get('overall_score', 0.0)

            # Generate report
            self._generate_validation_report(result)

            result.finalize()
            self._results_cache[validation_id] = result

        except Exception as e:
            error_msg = f"Statistical validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
            result.overall_score = 0.0

        return result

    def validate_patterns(self, original_db: str, synthetic_db: str) -> ValidationResult:
        """
        Perform pattern analysis validation

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            ValidationResult object
        """
        validation_id = f"patterns_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        result = ValidationResult('patterns')

        try:
            # Perform pattern analysis
            pattern_result = self.pattern_analyzer.analyze_patterns(original_db, synthetic_db)

            result.add_result('patterns', pattern_result)
            result.overall_score = pattern_result.get('overall_score', 0.0)

            # Generate report
            self._generate_validation_report(result)

            result.finalize()
            self._results_cache[validation_id] = result

        except Exception as e:
            error_msg = f"Pattern validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)
            result.overall_score = 0.0

        return result

    async def _validate_statistical_async(self, original_db: str, synthetic_db: str,
                                        sample_size: int, result: ValidationResult):
        """Statistical validation in async context"""
        try:
            loop = asyncio.get_event_loop()
            statistical_result = await loop.run_in_executor(
                self.thread_pool,
                self.statistical_validator.validate_distributions,
                sample_size,
                original_db,
                synthetic_db
            )

            result.add_result('statistical', statistical_result)

        except Exception as e:
            error_msg = f"Statistical validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)

    async def _validate_patterns_async(self, original_db: str, synthetic_db: str,
                                     result: ValidationResult):
        """Pattern validation in async context"""
        try:
            loop = asyncio.get_event_loop()
            pattern_result = await loop.run_in_executor(
                self.thread_pool,
                self.pattern_analyzer.analyze_patterns,
                original_db,
                synthetic_db
            )

            result.add_result('patterns', pattern_result)

        except Exception as e:
            error_msg = f"Pattern validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)

    async def _validate_clinical_async(self, original_db: str, synthetic_db: str,
                                     result: ValidationResult):
        """Clinical validation in async context"""
        try:
            loop = asyncio.get_event_loop()
            clinical_result = await loop.run_in_executor(
                self.thread_pool,
                self.clinical_validator.validate_clinical_patterns,
                original_db,
                synthetic_db
            )

            result.add_result('clinical', clinical_result)

        except Exception as e:
            error_msg = f"Clinical validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)

    async def _validate_temporal_async(self, original_db: str, synthetic_db: str,
                                     result: ValidationResult):
        """Temporal validation in async context"""
        try:
            loop = asyncio.get_event_loop()
            temporal_result = await loop.run_in_executor(
                self.thread_pool,
                self.temporal_validator.validate_temporal_patterns,
                original_db,
                synthetic_db
            )

            result.add_result('temporal', temporal_result)

        except Exception as e:
            error_msg = f"Temporal validation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)

    def _calculate_overall_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall validation score

        Args:
            results: Dictionary of validation results by category

        Returns:
            Overall score (0-100)
        """
        weights = {
            'statistical': 0.4,
            'patterns': 0.3,
            'clinical': 0.2,
            'temporal': 0.1
        }

        overall_score = 0.0
        total_weight = 0.0

        for category, weight in weights.items():
            if category in results:
                category_result = results[category]
                category_score = category_result.get('overall_score', 0.0)
                overall_score += category_score * weight
                total_weight += weight

        if total_weight > 0:
            overall_score = overall_score / total_weight

        return min(max(overall_score, 0.0), 100.0)

    def _generate_validation_report(self, result: ValidationResult):
        """Generate validation report"""
        try:
            # Generate comprehensive report
            report_path = self.report_generator.generate_comprehensive_report(
                result, self.config.visualization.get('report_formats', ['html'])
            )

            if report_path:
                result.add_metadata('report_path', report_path)

        except Exception as e:
            error_msg = f"Report generation failed: {e}"
            self.logger.error(error_msg)
            result.add_error(error_msg)

    def get_validation_history(self, limit: int = 10) -> List[ValidationResult]:
        """
        Get validation history

        Args:
            limit: Maximum number of results to return

        Returns:
            List of recent validation results
        """
        # Return results from cache, sorted by start time
        sorted_results = sorted(
            self._results_cache.values(),
            key=lambda x: x.start_time,
            reverse=True
        )

        return sorted_results[:limit]

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_validations = len(self._results_cache)

        if total_validations == 0:
            return {'total_validations': 0}

        scores = [result.overall_score for result in self._results_cache.values()]
        avg_score = sum(scores) / len(scores)

        # Count validations by type
        type_counts = {}
        for result in self._results_cache.values():
            validation_type = result.validation_type
            type_counts[validation_type] = type_counts.get(validation_type, 0) + 1

        return {
            'total_validations': total_validations,
            'average_score': avg_score,
            'validation_types': type_counts,
            'performance_stats': self.performance_tracker.get_stats()
        }

    def clear_cache(self):
        """Clear validation results cache"""
        self._results_cache.clear()
        self.logger.info("Validation cache cleared")

    def __del__(self):
        """Cleanup resources"""
        try:
            if hasattr(self, 'thread_pool'):
                self.thread_pool.shutdown(wait=True)
        except:
            pass


# Global orchestrator instance
_default_orchestrator: Optional[ValidationOrchestrator] = None
_orchestrator_lock = threading.Lock()


def get_orchestrator() -> ValidationOrchestrator:
    """Get global orchestrator instance"""
    global _default_orchestrator

    if _default_orchestrator is None:
        with _orchestrator_lock:
            if _default_orchestrator is None:
                _default_orchestrator = ValidationOrchestrator()

    return _default_orchestrator


async def validate_async(original_db: str, synthetic_db: str,
                        validation_type: str = 'comprehensive',
                        sample_size: Optional[int] = None) -> ValidationResult:
    """
    Async validation function

    Args:
        original_db: Path to original database
        synthetic_db: Path to synthetic database
        validation_type: Type of validation to perform
        sample_size: Sample size for validation

    Returns:
        ValidationResult object
    """
    orchestrator = get_orchestrator()

    if validation_type == 'comprehensive':
        return await orchestrator.validate_comprehensive(original_db, synthetic_db, sample_size)
    elif validation_type == 'statistical':
        return orchestrator.validate_statistical(original_db, synthetic_db, sample_size)
    elif validation_type == 'patterns':
        return orchestrator.validate_patterns(original_db, synthetic_db)
    else:
        raise ValueError(f"Unknown validation type: {validation_type}")


def validate(original_db: str, synthetic_db: str,
            validation_type: str = 'comprehensive',
            sample_size: Optional[int] = None) -> ValidationResult:
    """
    Synchronous validation function

    Args:
        original_db: Path to original database
        synthetic_db: Path to synthetic database
        validation_type: Type of validation to perform
        sample_size: Sample size for validation

    Returns:
        ValidationResult object
    """
    # Create new event loop for async function
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is already running, create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run,
                    validate_async(original_db, synthetic_db, validation_type, sample_size)
                )
                return future.result()
        else:
            return loop.run_until_complete(
                validate_async(original_db, synthetic_db, validation_type, sample_size)
            )
    except RuntimeError:
        # No event loop, create new one
        return asyncio.run(
            validate_async(original_db, synthetic_db, validation_type, sample_size)
        )
