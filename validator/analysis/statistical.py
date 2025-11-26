"""
Statistical validation module for NEDIS synthetic data.

This module performs comprehensive statistical validation including:
- Distribution comparison using KS test and Wasserstein distance
- Categorical variable analysis using Chi-square test
- Correlation analysis using Pearson/Spearman coefficients
- Multivariate analysis and PCA-based comparisons
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.stats import wasserstein_distance, ks_2samp, chi2_contingency
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
import warnings


class StatisticalValidator:
    """Advanced statistical validator for synthetic data"""

    def __init__(self, db_manager, config):
        """
        Initialize statistical validator

        Args:
            db_manager: Database manager instance
            config: Validation configuration
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Variable definitions
        self.continuous_variables = [
            'vst_sbp', 'vst_dbp', 'vst_per_pu', 'vst_per_br',
            'vst_bdht', 'vst_oxy', 'vst_bt', 'vst_wt'
        ]

        self.categorical_variables = [
            'pat_age_gr', 'pat_sex', 'pat_do_cd', 'ktas_fstu',
            'emtrt_rust', 'vst_meth', 'msypt', 'main_trt_p'
        ]

        # Validation thresholds
        self.ks_threshold = config.get_statistical_config('ks_threshold', 0.05)
        self.chi2_threshold = config.get_statistical_config('chi2_threshold', 0.05)
        self.correlation_threshold = config.get_statistical_config('correlation_threshold', 0.1)
        self.wasserstein_threshold = config.get_statistical_config('wasserstein_threshold', 0.1)

    def validate_distributions(
        self,
        sample_size: int = 50000,
        original_db_path: Optional[str] = None,
        synthetic_db_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical validation

        Args:
            sample_size: Sample size for validation

        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Starting statistical validation with sample size: {sample_size}")

        try:
            # Load sample data
            original_sample = self._load_sample_data('original', sample_size, original_db_path)
            synthetic_sample = self._load_sample_data('synthetic', sample_size, synthetic_db_path)

            if original_sample.empty or synthetic_sample.empty:
                return {
                    'success': False,
                    'reason': 'No data available',
                    'sample_sizes': {'original': len(original_sample), 'synthetic': len(synthetic_sample)}
                }

            results = {
                'success': True,
                'sample_sizes': {'original': len(original_sample), 'synthetic': len(synthetic_sample)},
                'continuous_tests': {},
                'categorical_tests': {},
                'correlation_analysis': {},
                'multivariate_analysis': {},
                'overall_score': 0.0
            }

            # 1. Continuous variable validation
            continuous_results = self._validate_continuous_variables(original_sample, synthetic_sample)
            results['continuous_tests'] = continuous_results

            # 2. Categorical variable validation
            categorical_results = self._validate_categorical_variables(original_sample, synthetic_sample)
            results['categorical_tests'] = categorical_results

            # 3. Correlation analysis
            correlation_results = self._validate_correlations(original_sample, synthetic_sample)
            results['correlation_analysis'] = correlation_results

            # 4. Multivariate analysis
            multivariate_results = self._validate_multivariate_distributions(original_sample, synthetic_sample)
            results['multivariate_analysis'] = multivariate_results

            # 5. Column profiling for dashboard display
            results['column_profiles'] = self._build_column_profiles(original_sample, synthetic_sample)

            # Calculate overall score
            overall_score = self._calculate_statistical_score(results)
            results['overall_score'] = overall_score

            self.logger.info(f"Statistical validation completed. Overall score: {overall_score:.2f}")

            return results

        except Exception as e:
            self.logger.error(f"Statistical validation failed: {e}")
            return {
                'success': False,
                'reason': str(e),
                'sample_sizes': {'original': 0, 'synthetic': 0}
            }

    def _load_sample_data(self, data_type: str, sample_size: int,
                          path_override: Optional[str] = None) -> pd.DataFrame:
        """
        Load sample data from database

        Args:
            data_type: 'original' or 'synthetic'
            sample_size: Number of samples to load

        Returns:
            pandas DataFrame with sample data
        """
        try:
            db_config = self.config.get_database_config(data_type)
            if not db_config and not path_override:
                self.logger.error(f"No database config found for {data_type}")
                return pd.DataFrame()

            # Get main clinical table
            schema = db_config.get('schema', 'main') if db_config else 'main'
            db_path = path_override or (db_config.get('path') if db_config else None)
            if not db_path:
                self.logger.error(f"No database path available for {data_type}")
                return pd.DataFrame()

            # Try to get table name from config first
            table_name = db_config.get('table') if db_config else None

            if not table_name:
                # Fallback to automatic detection
                available_tables = self.db.get_table_list(db_path, schema)
                preferred_tables = ['nedis2017', 'clinical_records', 'test']
                for preferred in preferred_tables:
                    if preferred in available_tables:
                        table_name = preferred
                        break

                if table_name is None and available_tables:
                    table_name = available_tables[0]
                elif table_name is None:
                    self.logger.error(f"No tables found in {data_type} database")
                    return pd.DataFrame()

            sample_df = self.db.sample_data(table_name, sample_size, db_path, schema)

            if sample_df.empty:
                self.logger.warning(f"No data sampled from {data_type} database")
                return pd.DataFrame()

            self.logger.info(f"Loaded {len(sample_df)} samples from {data_type} database")
            return sample_df

        except Exception as e:
            self.logger.error(f"Failed to load sample data from {data_type}: {e}")
            return pd.DataFrame()

    def _validate_continuous_variables(self, original_df: pd.DataFrame,
                                     synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate continuous variables using multiple statistical tests

        Args:
            original_df: Original data DataFrame
            synthetic_df: Synthetic data DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {}

        for var in self.continuous_variables:
            if var not in original_df.columns or var not in synthetic_df.columns:
                results[var] = {
                    'available': False,
                    'reason': 'Column not found in one or both datasets'
                }
                continue

            try:
                original_data = original_df[var].dropna()
                synthetic_data = synthetic_df[var].dropna()

                if len(original_data) < 10 or len(synthetic_data) < 10:
                    results[var] = {
                        'available': False,
                        'reason': 'Insufficient data for statistical testing'
                    }
                    continue

                # Kolmogorov-Smirnov test
                ks_stat, ks_pvalue = ks_2samp(original_data, synthetic_data)

                # Wasserstein distance (Earth Mover's Distance)
                wasserstein_dist = wasserstein_distance(original_data, synthetic_data)

                # Basic statistics
                orig_mean, orig_std = original_data.mean(), original_data.std()
                synth_mean, synth_std = synthetic_data.mean(), synthetic_data.std()

                # Calculate scores
                ks_score = 1.0 if ks_pvalue > self.ks_threshold else max(0.0, ks_pvalue / self.ks_threshold)
                wasserstein_score = 1.0 if wasserstein_dist < self.wasserstein_threshold else max(0.0, 1.0 - (wasserstein_dist / self.wasserstein_threshold))

                # Overall variable score
                variable_score = (ks_score + wasserstein_score) / 2.0

                results[var] = {
                    'available': True,
                    'ks_test': {
                        'statistic': float(ks_stat),
                        'p_value': float(ks_pvalue),
                        'score': float(ks_score)
                    },
                    'wasserstein_distance': {
                        'distance': float(wasserstein_dist),
                        'score': float(wasserstein_score)
                    },
                    'statistics': {
                        'original': {
                            'mean': float(orig_mean),
                            'std': float(orig_std),
                            'count': len(original_data)
                        },
                        'synthetic': {
                            'mean': float(synth_mean),
                            'std': float(synth_std),
                            'count': len(synthetic_data)
                        }
                    },
                    'score': float(variable_score)
                }

            except Exception as e:
                self.logger.warning(f"Failed to validate continuous variable {var}: {e}")
                results[var] = {
                    'available': False,
                    'reason': str(e)
                }

        return results

    def _validate_categorical_variables(self, original_df: pd.DataFrame,
                                      synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate categorical variables using Chi-square and other tests

        Args:
            original_df: Original data DataFrame
            synthetic_df: Synthetic data DataFrame

        Returns:
            Dictionary with validation results
        """
        results = {}

        for var in self.categorical_variables:
            if var not in original_df.columns or var not in synthetic_df.columns:
                results[var] = {
                    'available': False,
                    'reason': 'Column not found in one or both datasets'
                }
                continue

            try:
                original_data = original_df[var].dropna().astype(str)
                synthetic_data = synthetic_df[var].dropna().astype(str)

                if len(original_data) < 10 or len(synthetic_data) < 10:
                    results[var] = {
                        'available': False,
                        'reason': 'Insufficient data for statistical testing'
                    }
                    continue

                # Create contingency table
                contingency_table = pd.crosstab(original_data, synthetic_data)

                if contingency_table.size == 0:
                    results[var] = {
                        'available': False,
                        'reason': 'No overlapping categories'
                    }
                    continue

                # Chi-square test
                chi2_stat, chi2_pvalue, _, _ = chi2_contingency(contingency_table)

                # Calculate Cramer's V
                n = contingency_table.sum().sum()
                cramers_v = np.sqrt(chi2_stat / (n * (min(contingency_table.shape) - 1)))

                # Category distributions
                orig_dist = original_data.value_counts(normalize=True)
                synth_dist = synthetic_data.value_counts(normalize=True)

                # Calculate score
                chi2_score = 1.0 if chi2_pvalue > self.chi2_threshold else max(0.0, chi2_pvalue / self.chi2_threshold)
                cramers_score = 1.0 - min(cramers_v, 1.0)  # Lower Cramer's V is better for similarity

                variable_score = (chi2_score + cramers_score) / 2.0

                results[var] = {
                    'available': True,
                    'chi2_test': {
                        'statistic': float(chi2_stat),
                        'p_value': float(chi2_pvalue),
                        'score': float(chi2_score)
                    },
                    'cramers_v': {
                        'coefficient': float(cramers_v),
                        'score': float(cramers_score)
                    },
                    'distributions': {
                        'original': orig_dist.to_dict(),
                        'synthetic': synth_dist.to_dict()
                    },
                    'score': float(variable_score)
                }

            except Exception as e:
                self.logger.warning(f"Failed to validate categorical variable {var}: {e}")
                results[var] = {
                    'available': False,
                    'reason': str(e)
                }

        return results

    def _validate_correlations(self, original_df: pd.DataFrame,
                             synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate correlation structures between variables

        Args:
            original_df: Original data DataFrame
            synthetic_df: Synthetic data DataFrame

        Returns:
            Dictionary with correlation validation results
        """
        try:
            # Select numeric columns
            numeric_cols = []
            for col in original_df.columns:
                if col in synthetic_df.columns:
                    try:
                        original_df[col] = pd.to_numeric(original_df[col], errors='coerce')
                        synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce')
                        if not original_df[col].isna().all() and not synthetic_df[col].isna().all():
                            numeric_cols.append(col)
                    except:
                        continue

            if len(numeric_cols) < 2:
                return {
                    'available': False,
                    'reason': 'Insufficient numeric columns for correlation analysis'
                }

            # Calculate correlation matrices
            orig_corr = original_df[numeric_cols].corr(method='pearson')
            synth_corr = synthetic_df[numeric_cols].corr(method='pearson')

            # Calculate correlation differences
            corr_diff = orig_corr - synth_corr
            mse = mean_squared_error(orig_corr.values.flatten(), synth_corr.values.flatten())

            # Calculate score based on correlation similarity
            max_possible_mse = 4.0  # Maximum possible MSE for correlation matrices (-1 to 1 range)
            correlation_score = max(0.0, 1.0 - (mse / max_possible_mse))

            return {
                'available': True,
                'correlation_mse': float(mse),
                'correlation_score': float(correlation_score),
                'original_correlation_matrix': orig_corr.to_dict(),
                'synthetic_correlation_matrix': synth_corr.to_dict(),
                'correlation_difference': corr_diff.to_dict()
            }

        except Exception as e:
            self.logger.error(f"Correlation analysis failed: {e}")
            return {
                'available': False,
                'reason': str(e)
            }

    def _validate_multivariate_distributions(self, original_df: pd.DataFrame,
                                           synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform multivariate distribution analysis

        Args:
            original_df: Original data DataFrame
            synthetic_df: Synthetic data DataFrame

        Returns:
            Dictionary with multivariate validation results
        """
        try:
            # Select numeric columns
            numeric_cols = []
            for col in original_df.columns:
                if col in synthetic_df.columns and col in self.continuous_variables:
                    try:
                        original_df[col] = pd.to_numeric(original_df[col], errors='coerce')
                        synthetic_df[col] = pd.to_numeric(synthetic_df[col], errors='coerce')
                        if not original_df[col].isna().all() and not synthetic_df[col].isna().all():
                            numeric_cols.append(col)
                    except:
                        continue

            if len(numeric_cols) < 2:
                return {
                    'available': False,
                    'reason': 'Insufficient numeric columns for multivariate analysis'
                }

            # Prepare data for PCA
            orig_data = original_df[numeric_cols].dropna()
            synth_data = synthetic_df[numeric_cols].dropna()

            if len(orig_data) < 10 or len(synth_data) < 10:
                return {
                    'available': False,
                    'reason': 'Insufficient data for multivariate analysis'
                }

            # Standardize data
            scaler = StandardScaler()
            orig_scaled = scaler.fit_transform(orig_data)
            synth_scaled = scaler.transform(synth_data)

            # PCA analysis
            pca = PCA(n_components=min(5, len(numeric_cols) - 1))
            orig_pca = pca.fit_transform(orig_scaled)
            synth_pca = pca.transform(synth_scaled)

            # Compare PCA components
            pca_similarity_scores = []
            for i in range(orig_pca.shape[1]):
                if i < synth_pca.shape[1]:
                    # Compare explained variance
                    orig_var = pca.explained_variance_ratio_[i]
                    synth_var = np.var(synth_pca[:, i]) / np.var(orig_pca[:, i]) if np.var(orig_pca[:, i]) > 0 else 0

                    # Calculate similarity score for this component
                    component_score = 1.0 - abs(orig_var - synth_var)
                    pca_similarity_scores.append(component_score)

            # Overall multivariate score
            multivariate_score = np.mean(pca_similarity_scores) if pca_similarity_scores else 0.0

            return {
                'available': True,
                'pca_similarity_score': float(multivariate_score),
                'explained_variance_original': pca.explained_variance_ratio_.tolist(),
                'explained_variance_synthetic': (np.var(synth_pca, axis=0) / np.sum(np.var(synth_pca, axis=0))).tolist(),
                'numeric_columns_used': numeric_cols
            }

        except Exception as e:
            self.logger.error(f"Multivariate analysis failed: {e}")
            return {
                'available': False,
                'reason': str(e)
            }

    def _build_column_profiles(self, original_df: pd.DataFrame,
                               synthetic_df: pd.DataFrame) -> Dict[str, Any]:
        """Create per-column statistical summaries for both datasets"""
        profiles: Dict[str, Any] = {}

        shared_columns = [col for col in original_df.columns if col in synthetic_df.columns]

        for column in shared_columns:
            try:
                original_series = original_df[column]
                synthetic_series = synthetic_df[column]

                original_profile = self._profile_series(original_series)
                synthetic_profile = self._profile_series(synthetic_series)

                kind = original_profile.get('kind')
                if synthetic_profile.get('kind') != kind:
                    kind = 'mixed'

                profiles[column] = {
                    'kind': kind,
                    'original': original_profile,
                    'synthetic': synthetic_profile
                }

            except Exception as exc:
                self.logger.debug(f"Failed to profile column {column}: {exc}")

        return profiles

    def _profile_series(self, series: pd.Series) -> Dict[str, Any]:
        """Return statistics for a given series"""
        total_count = int(len(series))
        missing_count = int(series.isna().sum())
        non_null = series.dropna()

        profile: Dict[str, Any] = {
            'dtype': str(series.dtype),
            'total_count': total_count,
            'missing_count': missing_count,
            'count': int(len(non_null))
        }

        if non_null.empty:
            profile['kind'] = 'empty'
            return profile

        if pd.api.types.is_numeric_dtype(non_null):
            numeric_profile = self._profile_numeric(non_null)
            profile.update(numeric_profile)
        else:
            categorical_profile = self._profile_categorical(non_null)
            profile.update(categorical_profile)

        return profile

    def _profile_numeric(self, series: pd.Series) -> Dict[str, Any]:
        """Profile numeric data"""
        numeric = pd.to_numeric(series, errors='coerce').dropna()

        if numeric.empty:
            return {'kind': 'numeric', 'count': 0}

        q1 = self._safe_float(numeric.quantile(0.25))
        median = self._safe_float(numeric.quantile(0.5))
        q3 = self._safe_float(numeric.quantile(0.75))

        profile: Dict[str, Any] = {
            'kind': 'numeric',
            'count': int(numeric.count()),
            'mean': self._safe_float(numeric.mean()),
            'std': self._safe_float(numeric.std()),
            'min': self._safe_float(numeric.min()),
            'max': self._safe_float(numeric.max()),
            'median': median,
            'q1': q1,
            'q3': q3,
            'iqr': self._safe_float(q3 - q1 if q1 is not None and q3 is not None else None)
        }

        return profile

    def _profile_categorical(self, series: pd.Series) -> Dict[str, Any]:
        """Profile categorical data"""
        categorical = series.astype(str)
        value_counts = categorical.value_counts(dropna=False)
        total = int(value_counts.sum())

        top_values: List[Dict[str, Any]] = []
        if total > 0:
            for value, count in value_counts.head(5).items():
                percentage = (count / total) * 100 if total else 0.0
                top_values.append({
                    'value': value,
                    'count': int(count),
                    'percentage': self._safe_float(percentage)
                })

        profile: Dict[str, Any] = {
            'kind': 'categorical',
            'count': total,
            'unique_count': int(categorical.nunique(dropna=True)),
            'top_values': top_values
        }

        return profile

    def _safe_float(self, value: Any) -> Optional[float]:
        """Convert to float, returning None for invalid values"""
        if value is None:
            return None
        try:
            float_value = float(value)
            if np.isnan(float_value) or np.isinf(float_value):
                return None
            return float_value
        except (TypeError, ValueError):
            return None

    def _calculate_statistical_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate overall statistical validation score

        Args:
            results: Results from all statistical tests

        Returns:
            Overall statistical score (0-100)
        """
        try:
            # Continuous variables score
            continuous_scores = []
            continuous_tests = results.get('continuous_tests', {})
            for var, result in continuous_tests.items():
                if result.get('available', False):
                    continuous_scores.append(result.get('score', 0.0))

            continuous_score = np.mean(continuous_scores) if continuous_scores else 0.0

            # Categorical variables score
            categorical_scores = []
            categorical_tests = results.get('categorical_tests', {})
            for var, result in categorical_tests.items():
                if result.get('available', False):
                    categorical_scores.append(result.get('score', 0.0))

            categorical_score = np.mean(categorical_scores) if categorical_scores else 0.0

            # Correlation score
            correlation_result = results.get('correlation_analysis', {})
            correlation_score = correlation_result.get('correlation_score', 0.0) if correlation_result.get('available', False) else 0.0

            # Multivariate score
            multivariate_result = results.get('multivariate_analysis', {})
            multivariate_score = multivariate_result.get('pca_similarity_score', 0.0) if multivariate_result.get('available', False) else 0.0

            # Weighted overall score
            weights = {
                'continuous': 0.4,
                'categorical': 0.3,
                'correlation': 0.2,
                'multivariate': 0.1
            }

            overall_score = (
                continuous_score * weights['continuous'] +
                categorical_score * weights['categorical'] +
                correlation_score * weights['correlation'] +
                multivariate_score * weights['multivariate']
            )

            return min(max(overall_score * 100, 0.0), 100.0)

        except Exception as e:
            self.logger.error(f"Failed to calculate statistical score: {e}")
            return 0.0
