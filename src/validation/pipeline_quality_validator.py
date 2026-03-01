#!/usr/bin/env python3
"""Pipeline-level quality validator."""

from __future__ import annotations

from typing import Any, Dict, Optional
import logging
import math

import pandas as pd

from ..core.database import DatabaseManager
from ..core.config import ConfigManager
from ..temporal.comprehensive_time_gap_synthesizer import ComprehensiveTimeGapSynthesizer
from .clinical_validator import ClinicalRuleValidator
from .statistical_validator import StatisticalValidator
from .correlation_balance_validator import CorrelationBalanceValidator


class PipelineQualityValidator:
    """Run clinical, statistical, temporal, correlation, and capacity checks."""

    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.clinical_validator = ClinicalRuleValidator(db_manager, config)
        self.time_gap_validator = ComprehensiveTimeGapSynthesizer(db_manager, config)
        self._weights = self._load_weights()

    def evaluate(
        self,
        synthetic_df: pd.DataFrame,
        source_table: Optional[str] = None,
        capacity_report: Optional[Dict[str, Any]] = None,
        temporal_validation: Optional[Dict[str, Any]] = None,
        correlation_sample_size: int = 100_000,
        quality_gate: float = 0.0,
        clinical_sample_size: int = 100_000,
        statistical_sample_size: int = 50_000,
    ) -> Dict[str, Any]:
        """Run full pipeline quality checks.

        Args:
            synthetic_df: In-memory synthetic data generated in the current run.
            source_table: Source table to compare against for statistical checks.
            capacity_report: Output from CapacityConstraintPostProcessor.
            temporal_validation: Optional temporal validation report from TemporalPatternAssigner.
            quality_gate: Minimum acceptable overall score.
        """
        if quality_gate > 1.0:
            quality_gate = quality_gate / 100.0

        self.logger.info("Starting pipeline quality evaluation")

        results: Dict[str, Any] = {}
        component_scores: Dict[str, float] = {}
        component_weights: Dict[str, float] = {}
        details: Dict[str, Any] = {}

        # 1) Clinical rules
        clinical_results = self.clinical_validator.validate_all_clinical_rules(sample_size=clinical_sample_size)
        clinical_score = self._score_clinical(clinical_results)
        component_scores["clinical"] = clinical_score
        component_weights["clinical"] = self._weights["clinical"]
        details["clinical"] = clinical_results
        details["clinical"]["score"] = clinical_score

        # 2) Statistical distribution checks
        stats_validator = StatisticalValidator(self.db, self.config, source_table=source_table)
        statistical_results = stats_validator.validate_distributions(
            sample_size=statistical_sample_size,
            synthetic_df=synthetic_df
        )
        statistical_score = (
            float(statistical_results.get("overall_score", 0.0))
            if isinstance(statistical_results, dict)
            else 0.0
        )
        component_scores["statistical"] = self._clamp_score(statistical_score)
        component_weights["statistical"] = self._weights["statistical"]
        details["statistical"] = statistical_results
        details["statistical"]["score"] = component_scores["statistical"]

        # 3) Temporal consistency
        temporal_results = self._evaluate_temporal_component(
            synthetic_df=synthetic_df,
            temporal_validation=temporal_validation
        )
        temporal_score = self._score_temporal(temporal_results)
        if temporal_score is not None:
            component_scores["temporal"] = temporal_score
            component_weights["temporal"] = self._weights["temporal"]
            details["temporal"] = temporal_results
            details["temporal"]["score"] = temporal_score

        # 4) Correlation balance checks
        correlation_validator = CorrelationBalanceValidator(self.db, self.config, source_table=source_table)
        correlation_results = correlation_validator.validate_pairwise_correlations(
            sample_size=correlation_sample_size,
            synthetic_df=synthetic_df,
        )
        correlation_score = self._score_correlation(correlation_results)
        if correlation_score is not None:
            component_scores["correlation"] = correlation_score
            component_weights["correlation"] = self._weights["correlation"]
            details["correlation"] = correlation_results
            details["correlation"]["score"] = correlation_score

        # 5) Capacity report quality
        capacity_score = self._score_capacity(capacity_report)
        if capacity_score is not None:
            component_scores["capacity"] = capacity_score
            component_weights["capacity"] = self._weights["capacity"]
            details["capacity"] = {
                "report": capacity_report,
                "score": capacity_score,
            }

        weighted_scores = 0.0
        weighted_weights = 0.0
        for key, score in component_scores.items():
            weight = component_weights.get(key, 0.0)
            if weight <= 0:
                continue
            weighted_scores += score * weight
            weighted_weights += weight

        overall_score = (weighted_scores / weighted_weights) if weighted_weights > 0 else 0.0
        overall_score = self._clamp_score(overall_score)

        passed = overall_score >= quality_gate
        summary = {
            "overall_score": overall_score,
            "passed": passed,
            "quality_gate": quality_gate,
            "components": component_scores,
            "weights": component_weights,
        }
        results["overall_score"] = overall_score
        results["details"] = details
        results["summary"] = summary
        results["passed"] = passed
        results["component_scores"] = component_scores

        self.logger.info(
            "Pipeline quality evaluation completed: score=%.4f, passed=%s", overall_score, passed
        )
        return results

    def _evaluate_temporal_component(
        self,
        synthetic_df: pd.DataFrame,
        temporal_validation: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Return a merged temporal validation report."""
        if temporal_validation is None:
            return self.time_gap_validator.validate_time_consistency(synthetic_df)

        validation = dict(temporal_validation)
        # Add datetime consistency check from gap model for stronger temporal guarantees.
        gap_validation = self.time_gap_validator.validate_time_consistency(synthetic_df)
        validation["time_gap_consistency"] = gap_validation.get("consistency_checks", {})
        return validation

    def _score_clinical(self, results: Dict[str, Any]) -> float:
        """Clinical score based on compliance and critical violations."""
        if not results.get("success", False):
            return 0.0

        overall_compliance = float(results.get("overall_compliance_rate", 0.0))
        if overall_compliance < 0.0:
            return 0.0
        if overall_compliance > 1.0:
            overall_compliance = 1.0

        violations = results.get("violations_summary", {})
        sample_size = int(results.get("sample_size", 0) or 0)
        total_rules = 0
        for category in results.get("rule_categories", {}).values():
            total_rules += len(category.get("rules", []))
        if total_rules <= 0:
            return overall_compliance

        critical = int(violations.get("critical_violations", 0))
        warning = int(violations.get("warning_violations", 0))
        total_checks = sample_size * max(total_rules, 1)

        critical_penalty = min(0.5, (critical / total_checks) * 10) if total_checks > 0 else 0.0
        warning_penalty = min(0.25, (warning / total_checks) * 2) if total_checks > 0 else 0.0

        return self._clamp_score(overall_compliance - critical_penalty - warning_penalty)

    def _score_temporal(self, results: Dict[str, Any]) -> Optional[float]:
        """Score temporal checks from 0 to 1."""
        if not isinstance(results, dict):
            return None

        # Temporal assignment-level validation (from TemporalPatternAssigner)
        summary_score = 1.0
        summary = results.get("summary", {})
        if isinstance(summary, dict):
            total_patients = int(summary.get("total_patients", 0) or 0)
            unique_dates = int(summary.get("unique_dates", 0) or 0)
            if total_patients <= 0:
                return 0.0
            if unique_dates <= 1:
                summary_score *= 0.7
            else:
                summary_score *= min(1.0, unique_dates / 300.0)

        # Time gap consistency checks
        consistency = results.get("consistency_checks", {})
        if not consistency:
            consistency = results.get("time_gap_consistency", {})

        if isinstance(consistency, dict) and consistency:
            percentages = []
            for check in consistency.values():
                if isinstance(check, dict) and "percentage" in check:
                    percentages.append(float(check["percentage"]) / 100.0)
            if percentages:
                consistency_score = min(percentages)
            else:
                consistency_score = 1.0
        else:
            consistency_score = 1.0

        # Temporal assignment range guard (if exists)
        date_range = results.get("date_range", {})
        range_score = 1.0
        if isinstance(date_range, dict) and "range_valid" in date_range:
            range_score = 1.0 if bool(date_range["range_valid"]) else 0.5

        return self._clamp_score((summary_score * 0.4 + consistency_score * 0.6) * range_score)

    def _score_correlation(self, results: Dict[str, Any]) -> Optional[float]:
        """Use pairwise correlation balance score as-is."""
        if not isinstance(results, dict):
            return None

        value = results.get("overall_score")
        if value is None:
            return None
        try:
            return self._clamp_score(float(value))
        except (TypeError, ValueError):
            return None

    def _score_capacity(self, capacity_report: Optional[Dict[str, Any]]) -> Optional[float]:
        """Score capacity redistribution behavior."""
        if capacity_report is None:
            return None

        redistribution_rate = capacity_report.get("redistribution_rate")
        if redistribution_rate is None:
            return None

        redis_rate = float(redistribution_rate)
        if math.isnan(redis_rate) or math.isinf(redis_rate):
            return 0.0

        if redis_rate <= 0:
            return 1.0

        tolerance = float(self.config.get("validation.max_redistribution_rate", 15.0))
        tolerance = max(1.0, tolerance)
        if redis_rate <= tolerance:
            return 1.0

        # Linear decay to 0 at 100%
        return self._clamp_score(1.0 - (redis_rate - tolerance) / (100.0 - tolerance))

    def _clamp_score(self, value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _load_weights(self) -> Dict[str, float]:
        """Load component weights from config."""
        defaults = {
            "clinical": 0.25,
            "statistical": 0.35,
            "temporal": 0.2,
            "correlation": 0.15,
            "capacity": 0.05,
        }

        configured = self.config.get("validation.pipeline_component_weights", defaults)
        if not isinstance(configured, dict):
            return defaults

        weights = dict(defaults)
        for key in defaults:
            configured_weight = configured.get(key, defaults[key])
            try:
                weights[key] = float(configured_weight)
            except (TypeError, ValueError):
                weights[key] = defaults[key]

        total = sum(weights.values())
        if total <= 0:
            return defaults
        return weights
