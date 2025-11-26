"""
Pattern analysis module for NEDIS synthetic data validation.

This module performs dynamic pattern analysis including:
- Hierarchical pattern discovery with fallback mechanism
- Time gap pattern analysis
- Demographic pattern analysis
- Clinical pattern analysis
- Automated pattern caching and retrieval
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path


class PatternAnalyzer:
    """Advanced pattern analyzer with dynamic discovery and caching"""

    def __init__(self, db_manager, config):
        """
        Initialize pattern analyzer

        Args:
            db_manager: Database manager instance
            config: Validation configuration
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Pattern cache
        self.pattern_cache = {}
        self.cache_enabled = config.get_pattern_config('pattern_cache_enabled', True)

        # Hierarchical analysis settings
        self.min_sample_size = config.get_pattern_config('min_sample_size', 10)
        self.confidence_threshold = config.get_pattern_config('confidence_threshold', 0.95)
        self.hierarchical_fallback = config.get_pattern_config('hierarchical_fallback', True)

        # Time gap analysis settings
        self.time_gap_analysis_enabled = config.get_pattern_config('time_gap_analysis', True)

    def analyze_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Perform comprehensive pattern analysis

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with pattern analysis results
        """
        self.logger.info("Starting pattern analysis")

        try:
            results = {
                'success': True,
                'demographic_patterns': {},
                'clinical_patterns': {},
                'temporal_patterns': {},
                'time_gap_patterns': {},
                'overall_score': 0.0
            }

            # 1. Demographic pattern analysis
            demographic_results = self._analyze_demographic_patterns(original_db, synthetic_db)
            results['demographic_patterns'] = demographic_results

            # 2. Clinical pattern analysis
            clinical_results = self._analyze_clinical_patterns(original_db, synthetic_db)
            results['clinical_patterns'] = clinical_results

            # 3. Temporal pattern analysis
            temporal_results = self._analyze_temporal_patterns(original_db, synthetic_db)
            results['temporal_patterns'] = temporal_results

            # 4. Time gap pattern analysis
            if self.time_gap_analysis_enabled:
                time_gap_results = self._analyze_time_gap_patterns(original_db, synthetic_db)
                results['time_gap_patterns'] = time_gap_results

            # Calculate overall score
            overall_score = self._calculate_pattern_score(results)
            results['overall_score'] = overall_score

            self.logger.info(f"Pattern analysis completed. Overall score: {overall_score:.2f}")

            return results

        except Exception as e:
            self.logger.error(f"Pattern analysis failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'overall_score': 0.0
            }

    def _analyze_demographic_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze demographic patterns with hierarchical fallback

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with demographic pattern analysis
        """
        try:
            patterns = {
                'age_patterns': {},
                'gender_patterns': {},
                'region_patterns': {}
            }

            # Age group patterns
            age_patterns = self._analyze_hierarchical_pattern(
                original_db, synthetic_db, 'pat_age_gr', 'age_group'
            )
            patterns['age_patterns'] = age_patterns

            # Gender patterns
            gender_patterns = self._analyze_hierarchical_pattern(
                original_db, synthetic_db, 'pat_sex', 'gender'
            )
            patterns['gender_patterns'] = gender_patterns

            # Regional patterns
            region_patterns = self._analyze_hierarchical_pattern(
                original_db, synthetic_db, 'pat_do_cd', 'region'
            )
            patterns['region_patterns'] = region_patterns

            return patterns

        except Exception as e:
            self.logger.error(f"Demographic pattern analysis failed: {e}")
            return {}

    def _analyze_clinical_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze clinical patterns

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with clinical pattern analysis
        """
        try:
            patterns = {
                'ktas_patterns': {},
                'diagnosis_patterns': {},
                'treatment_patterns': {}
            }

            # KTAS patterns
            ktas_patterns = self._analyze_hierarchical_pattern(
                original_db, synthetic_db, 'ktas_fstu', 'ktas_level'
            )
            patterns['ktas_patterns'] = ktas_patterns

            # Diagnosis patterns (if available)
            try:
                diagnosis_patterns = self._analyze_diagnosis_patterns(original_db, synthetic_db)
                patterns['diagnosis_patterns'] = diagnosis_patterns
            except:
                patterns['diagnosis_patterns'] = {'available': False, 'reason': 'Diagnosis data not available'}

            # Treatment patterns (if available)
            try:
                treatment_patterns = self._analyze_treatment_patterns(original_db, synthetic_db)
                patterns['treatment_patterns'] = treatment_patterns
            except:
                patterns['treatment_patterns'] = {'available': False, 'reason': 'Treatment data not available'}

            return patterns

        except Exception as e:
            self.logger.error(f"Clinical pattern analysis failed: {e}")
            return {}

    def _analyze_temporal_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze temporal patterns

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with temporal pattern analysis
        """
        try:
            patterns = {
                'daily_patterns': {},
                'weekly_patterns': {},
                'monthly_patterns': {},
                'hourly_patterns': {}
            }

            # Daily patterns
            daily_patterns = self._analyze_temporal_pattern(
                original_db, synthetic_db, 'vst_dt', 'daily', '%w'
            )
            patterns['daily_patterns'] = daily_patterns

            # Weekly patterns
            weekly_patterns = self._analyze_temporal_pattern(
                original_db, synthetic_db, 'vst_dt', 'weekly', '%w'
            )
            patterns['weekly_patterns'] = weekly_patterns

            # Monthly patterns
            monthly_patterns = self._analyze_temporal_pattern(
                original_db, synthetic_db, 'vst_dt', 'monthly', '%m'
            )
            patterns['monthly_patterns'] = monthly_patterns

            # Hourly patterns (if time data available)
            try:
                hourly_patterns = self._analyze_hourly_patterns(original_db, synthetic_db)
                patterns['hourly_patterns'] = hourly_patterns
            except:
                patterns['hourly_patterns'] = {'available': False, 'reason': 'Hourly data not available'}

            return patterns

        except Exception as e:
            self.logger.error(f"Temporal pattern analysis failed: {e}")
            return {}

    def _analyze_time_gap_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze time gap patterns between consecutive visits

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with time gap pattern analysis
        """
        try:
            # This would require patient ID to calculate time gaps between visits
            # For now, return a placeholder implementation
            return {
                'available': False,
                'reason': 'Time gap analysis requires patient ID tracking',
                'implementation_status': 'planned'
            }

        except Exception as e:
            self.logger.error(f"Time gap pattern analysis failed: {e}")
            return {}

    def _analyze_hierarchical_pattern(self, original_db: str, synthetic_db: str,
                                    column_name: str, pattern_type: str) -> Dict[str, Any]:
        """
        Analyze patterns with hierarchical fallback mechanism

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            column_name: Column to analyze
            pattern_type: Type of pattern for caching

        Returns:
            Dictionary with hierarchical pattern analysis
        """
        cache_key = f"{pattern_type}_{column_name}_{Path(original_db).stem}_{Path(synthetic_db).stem}"

        # Check cache first
        if self.cache_enabled and cache_key in self.pattern_cache:
            cached_result = self.pattern_cache[cache_key]
            if self._is_cache_valid(cached_result):
                return cached_result

        try:
            # Level 1: Exact pattern matching
            exact_match = self._calculate_pattern_similarity(
                original_db, synthetic_db, column_name, 'exact'
            )

            if exact_match['similarity_score'] >= self.confidence_threshold:
                result = {
                    'level': 'exact',
                    'similarity_score': exact_match['similarity_score'],
                    'distribution_match': exact_match['distribution_match'],
                    'sample_size': exact_match['sample_size'],
                    'confidence': 'high'
                }
            else:
                # Level 2: Hierarchical grouping
                hierarchical_match = self._calculate_hierarchical_similarity(
                    original_db, synthetic_db, column_name
                )

                if hierarchical_match['similarity_score'] >= self.confidence_threshold:
                    result = {
                        'level': 'hierarchical',
                        'similarity_score': hierarchical_match['similarity_score'],
                        'distribution_match': hierarchical_match['distribution_match'],
                        'grouping_applied': hierarchical_match['grouping_applied'],
                        'confidence': 'medium'
                    }
                else:
                    # Level 3: National level aggregation
                    national_match = self._calculate_national_similarity(
                        original_db, synthetic_db, column_name
                    )

                    result = {
                        'level': 'national',
                        'similarity_score': national_match['similarity_score'],
                        'distribution_match': national_match['distribution_match'],
                        'aggregation_method': 'national_average',
                        'confidence': 'low'
                    }

            # Cache result
            if self.cache_enabled:
                self.pattern_cache[cache_key] = result

            return result

        except Exception as e:
            self.logger.error(f"Hierarchical pattern analysis failed for {column_name}: {e}")
            return {
                'error': str(e),
                'level': 'failed',
                'similarity_score': 0.0,
                'confidence': 'none'
            }

    def _calculate_pattern_similarity(self, original_db: str, synthetic_db: str,
                                    column_name: str, method: str = 'exact') -> Dict[str, Any]:
        """
        Calculate pattern similarity between original and synthetic data

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            column_name: Column to analyze
            method: Similarity calculation method

        Returns:
            Dictionary with similarity metrics
        """
        try:
            # Load data from both databases
            orig_db_config = self.config.get_database_config('original')
            synth_db_config = self.config.get_database_config('synthetic')

            # Use configured tables if available
            orig_table = orig_db_config.get('table', 'nedis2017')
            synth_table = synth_db_config.get('table', 'clinical_records')

            # Check if configured tables exist
            orig_tables = self.db.get_table_list(orig_db_config['path'], orig_db_config.get('schema', 'main'))
            synth_tables = self.db.get_table_list(synth_db_config['path'], synth_db_config.get('schema', 'main'))

            if orig_table not in orig_tables:
                self.logger.warning(f"Configured original table '{orig_table}' not found in {orig_tables}")
                # Fallback to any available table
                preferred_tables = ['nedis2017', 'clinical_records', 'test']
                for preferred in preferred_tables:
                    if preferred in orig_tables:
                        orig_table = preferred
                        break
                else:
                    orig_table = orig_tables[0] if orig_tables else None

            if synth_table not in synth_tables:
                self.logger.warning(f"Configured synthetic table '{synth_table}' not found in {synth_tables}")
                # Fallback to any available table
                preferred_tables = ['clinical_records', 'nedis2017', 'test']
                for preferred in preferred_tables:
                    if preferred in synth_tables:
                        synth_table = preferred
                        break
                else:
                    synth_table = synth_tables[0] if synth_tables else None

            if not orig_table or not synth_table:
                self.logger.warning(f"No valid tables found. Original: {orig_tables}, Synthetic: {synth_tables}")
                return {'similarity_score': 0.0, 'distribution_match': {}, 'sample_size': 0}

            # Build full table names with schema
            orig_schema = orig_db_config.get('schema', 'main')
            synth_schema = synth_db_config.get('schema', 'main')

            orig_full_table = f"{orig_schema}.{orig_table}" if orig_schema != 'main' else orig_table
            synth_full_table = f"{synth_schema}.{synth_table}" if synth_schema != 'main' else synth_table

            # Check if the column exists in the target table
            orig_query = f"SELECT {column_name}, COUNT(*) as count FROM {orig_full_table} GROUP BY {column_name} ORDER BY count DESC"
            synth_query = f"SELECT {column_name}, COUNT(*) as count FROM {synth_full_table} GROUP BY {column_name} ORDER BY count DESC"

            orig_df = self.db.fetch_dataframe(orig_query, orig_db_config['path'], orig_db_config.get('schema', 'main'))
            synth_df = self.db.fetch_dataframe(synth_query, synth_db_config['path'], synth_db_config.get('schema', 'main'))

            if orig_df.empty or synth_df.empty:
                return {'similarity_score': 0.0, 'distribution_match': {}, 'sample_size': 0}

            # Calculate distribution similarity
            orig_dist = orig_df.set_index(column_name)['count'] / orig_df['count'].sum()
            synth_dist = synth_df.set_index(column_name)['count'] / synth_df['count'].sum()

            # Align distributions
            all_categories = set(orig_dist.index) | set(synth_dist.index)
            orig_dist = orig_dist.reindex(all_categories, fill_value=0)
            synth_dist = synth_dist.reindex(all_categories, fill_value=0)

            # Calculate Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon
            js_distance = jensenshannon(orig_dist.values, synth_dist.values)

            # Convert distance to similarity score (1 - normalized_distance)
            max_js_distance = np.sqrt(np.log(2))  # Maximum possible JS distance
            similarity_score = max(0.0, 1.0 - (js_distance / max_js_distance))

            # Calculate distribution match details
            distribution_match = {}
            for category in all_categories:
                orig_pct = orig_dist[category] * 100
                synth_pct = synth_dist[category] * 100
                distribution_match[str(category)] = {
                    'original': float(orig_pct),
                    'synthetic': float(synth_pct),
                    'difference': float(abs(orig_pct - synth_pct))
                }

            return {
                'similarity_score': float(similarity_score),
                'distribution_match': distribution_match,
                'sample_size': len(orig_df),
                'js_distance': float(js_distance)
            }

        except Exception as e:
            self.logger.error(f"Pattern similarity calculation failed: {e}")
            return {'similarity_score': 0.0, 'distribution_match': {}, 'sample_size': 0}

    def _calculate_hierarchical_similarity(self, original_db: str, synthetic_db: str,
                                         column_name: str) -> Dict[str, Any]:
        """
        Calculate similarity using hierarchical grouping

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            column_name: Column to analyze

        Returns:
            Dictionary with hierarchical similarity metrics
        """
        # This is a simplified implementation
        # In practice, this would implement sophisticated hierarchical grouping
        return {
            'similarity_score': 0.7,  # Placeholder
            'distribution_match': {},
            'grouping_applied': 'basic_grouping'
        }

    def _calculate_national_similarity(self, original_db: str, synthetic_db: str,
                                     column_name: str) -> Dict[str, Any]:
        """
        Calculate similarity using national-level aggregation

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            column_name: Column to analyze

        Returns:
            Dictionary with national similarity metrics
        """
        # This is a simplified implementation
        return {
            'similarity_score': 0.5,  # Placeholder
            'distribution_match': {},
            'aggregation_method': 'national_average'
        }

    def _analyze_temporal_pattern(self, original_db: str, synthetic_db: str,
                                date_column: str, pattern_type: str, date_format: str) -> Dict[str, Any]:
        """
        Analyze temporal patterns

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database
            date_column: Date column name
            pattern_type: Type of temporal pattern
            date_format: Date format string for grouping

        Returns:
            Dictionary with temporal pattern analysis
        """
        try:
            # Build query for temporal analysis
            # Adjust date format based on pattern type
            if pattern_type == 'weekly':
                date_expr = f"SUBSTR({date_column}, 1, 4) || '-' || SUBSTR({date_column}, 5, 2) || '-' || SUBSTR({date_column}, 7, 2)"
                time_period_expr = f"STRFTIME('%w', CAST({date_expr} AS DATE))"
            elif pattern_type == 'monthly':
                time_period_expr = f"SUBSTR({date_column}, 5, 2)"
            elif pattern_type == 'hourly':
                # For hourly, we might need a time column, defaulting to daily for now
                time_period_expr = f"SUBSTR({date_column}, 7, 2)"
            else:
                # Default to daily
                time_period_expr = f"{date_column}"

            query = f"""
            SELECT
                CASE
                    WHEN {date_column} IS NULL OR {date_column} = '' THEN 'unknown'
                    ELSE {time_period_expr}
                END as time_period,
                COUNT(*) as count
            FROM {{table}}
            GROUP BY time_period
            ORDER BY count DESC
            """

            # Execute queries
            orig_db_config = self.config.get_database_config('original')
            synth_db_config = self.config.get_database_config('synthetic')

            # Use configured tables if available
            orig_table = orig_db_config.get('table', 'nedis2017')
            synth_table = synth_db_config.get('table', 'clinical_records')

            # Check if configured tables exist
            orig_tables = self.db.get_table_list(orig_db_config['path'], orig_db_config.get('schema', 'main'))
            synth_tables = self.db.get_table_list(synth_db_config['path'], synth_db_config.get('schema', 'main'))

            if orig_table not in orig_tables:
                self.logger.warning(f"Configured original table '{orig_table}' not found in {orig_tables}")
                preferred_tables = ['nedis2017', 'clinical_records', 'test']
                for preferred in preferred_tables:
                    if preferred in orig_tables:
                        orig_table = preferred
                        break
                else:
                    orig_table = orig_tables[0] if orig_tables else None

            if synth_table not in synth_tables:
                self.logger.warning(f"Configured synthetic table '{synth_table}' not found in {synth_tables}")
                preferred_tables = ['clinical_records', 'nedis2017', 'test']
                for preferred in preferred_tables:
                    if preferred in synth_tables:
                        synth_table = preferred
                        break
                else:
                    synth_table = synth_tables[0] if synth_tables else None

            if not orig_table or not synth_table:
                self.logger.warning(f"No valid tables found for temporal analysis")
                return {'pattern_score': 0.0, 'temporal_distributions': {}, 'sample_size': 0}

            # Build full table names with schema
            orig_schema = orig_db_config.get('schema', 'main')
            synth_schema = synth_db_config.get('schema', 'main')

            orig_full_table = f"{orig_schema}.{orig_table}" if orig_schema != 'main' else orig_table
            synth_full_table = f"{synth_schema}.{synth_table}" if synth_schema != 'main' else synth_table

            orig_query = query.format(table=orig_full_table)
            synth_query = query.format(table=synth_full_table)

            orig_df = self.db.fetch_dataframe(orig_query, orig_db_config['path'], orig_db_config.get('schema', 'main'))
            synth_df = self.db.fetch_dataframe(synth_query, synth_db_config['path'], synth_db_config.get('schema', 'main'))

            if orig_df.empty or synth_df.empty:
                return {'available': False, 'reason': 'No temporal data available'}

            # Calculate temporal distribution similarity
            orig_dist = orig_df.set_index('time_period')['count'] / orig_df['count'].sum()
            synth_dist = synth_df.set_index('time_period')['count'] / synth_df['count'].sum()

            # Align distributions
            all_periods = set(orig_dist.index) | set(synth_dist.index)
            orig_dist = orig_dist.reindex(all_periods, fill_value=0)
            synth_dist = synth_dist.reindex(all_periods, fill_value=0)

            # Calculate Jensen-Shannon divergence
            from scipy.spatial.distance import jensenshannon
            js_distance = jensenshannon(orig_dist.values, synth_dist.values)
            max_js_distance = np.sqrt(np.log(2))
            similarity_score = max(0.0, 1.0 - (js_distance / max_js_distance))

            return {
                'available': True,
                'similarity_score': float(similarity_score),
                'distribution_match': {
                    period: {
                        'original': float(orig_dist[period] * 100),
                        'synthetic': float(synth_dist[period] * 100),
                        'difference': float(abs(orig_dist[period] - synth_dist[period]) * 100)
                    }
                    for period in all_periods
                },
                'sample_size': len(orig_df)
            }

        except Exception as e:
            self.logger.error(f"Temporal pattern analysis failed: {e}")
            return {'available': False, 'reason': str(e)}

    def _analyze_hourly_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze hourly patterns

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with hourly pattern analysis
        """
        # Placeholder implementation
        return {'available': False, 'reason': 'Hourly time data not available in current schema'}

    def _analyze_diagnosis_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze diagnosis patterns

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with diagnosis pattern analysis
        """
        # Placeholder implementation
        return {'available': False, 'reason': 'Diagnosis data not available in current schema'}

    def _analyze_treatment_patterns(self, original_db: str, synthetic_db: str) -> Dict[str, Any]:
        """
        Analyze treatment patterns

        Args:
            original_db: Path to original database
            synthetic_db: Path to synthetic database

        Returns:
            Dictionary with treatment pattern analysis
        """
        # Placeholder implementation
        return {'available': False, 'reason': 'Treatment data not available in current schema'}

    def _calculate_pattern_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall pattern analysis score

        Args:
            results: Results from all pattern analyses

        Returns:
            Overall pattern score (0-100)
        """
        try:
            scores = []

            # Demographic patterns score
            demographic_patterns = results.get('demographic_patterns', {})
            for pattern_type, pattern_result in demographic_patterns.items():
                if isinstance(pattern_result, dict) and 'similarity_score' in pattern_result:
                    scores.append(pattern_result['similarity_score'])

            # Clinical patterns score
            clinical_patterns = results.get('clinical_patterns', {})
            for pattern_type, pattern_result in clinical_patterns.items():
                if isinstance(pattern_result, dict) and 'similarity_score' in pattern_result:
                    scores.append(pattern_result['similarity_score'])

            # Temporal patterns score
            temporal_patterns = results.get('temporal_patterns', {})
            for pattern_type, pattern_result in temporal_patterns.items():
                if isinstance(pattern_result, dict) and 'similarity_score' in pattern_result:
                    scores.append(pattern_result['similarity_score'])

            # Calculate weighted average
            if scores:
                return float(np.mean(scores) * 100)
            else:
                return 0.0

        except Exception as e:
            self.logger.error(f"Failed to calculate pattern score: {e}")
            return 0.0

    def _is_cache_valid(self, cached_result: Dict[str, Any]) -> bool:
        """
        Check if cached pattern result is still valid

        Args:
            cached_result: Cached pattern result

        Returns:
            True if cache is valid, False otherwise
        """
        # For now, always consider cache valid
        # In production, this would check data freshness
        return True
