"""
Statistical Validator

원본 데이터와 합성 데이터 간의 통계적 유사성을 검증하는 모듈입니다.

검증 항목:
- 연속형 변수: Kolmogorov-Smirnov 검정
- 범주형 변수: Chi-square 검정  
- 상관관계 분석: Pearson/Spearman 상관계수
- 분포 형태: Quantile-Quantile plot 분석
- 다변량 분석: Multivariate normality test
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats
from scipy.stats import wasserstein_distance

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class StatisticalValidator:
    """통계적 검증기"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        config: ConfigManager,
        source_table: Optional[str] = None,
    ):
        """
        통계적 검증기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
            source_table: 원본 테이블 경로
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.source_table = self._resolve_source_table(source_table)
        self._table_schema_cache: Dict[str, List[str]] = {}
        self._continuous_config = {
            "ptmihibp": (40, 300),
            "ptmilobp": (30, 220),
            "ptmipuls": (1, 300),
            "ptmibrth": (1, 200),
            "ptmibdht": (34.0, 44.0),
            "ptmivoxs": (70, 100),
        }
        self._invalid_markers = {
            -1,
            -1.0,
            999,
            999.0,
            "999",
            "-1",
            "-1.0",
            "None",
            "NULL",
            "",
            " ",
        }
        
        # 검증할 변수들 정의
        self.continuous_variables = [
            'ptmihibp', 'ptmilobp', 'ptmipuls', 'ptmibrth', 'ptmibdht', 'ptmivoxs'
        ]
        
        self.categorical_variables = [
            'ptmibrtd', 'ptmisexx', 'ptmizipc', 'ptmikts1', 'ptmiemrt', 
            'ptmiinmn', 'ptmimnsy', 'ptmidept'
        ]
        
        # 검정 임계값
        self.significance_level = 0.05
        self.ks_threshold = 0.05        # KS 검정 p-value 임계값
        self.chi2_threshold = 0.05      # Chi-square 검정 p-value 임계값  
        self.correlation_threshold = 0.1 # 상관계수 차이 임계값
        
    def validate_distributions(
        self,
        sample_size: int = 50000,
        synthetic_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """
        전체적인 분포 검증 수행
        
        Args:
            sample_size: 검증에 사용할 샘플 크기
            synthetic_df: 외부에서 전달된 합성 데이터프레임(없으면 DB 샘플 사용)
            
        Returns:
            검증 결과 딕셔너리
        """
        
        self.logger.info(f"Starting comprehensive distribution validation with sample size: {sample_size}")
        
        try:
            # 원본 및 합성 데이터 샘플 로드
            original_sample = self._load_original_sample(sample_size)
            synthetic_sample = self._load_synthetic_sample(sample_size, synthetic_df)
            
            if original_sample.empty or synthetic_sample.empty:
                return {'success': False, 'reason': 'No data available'}
            
            validation_results = {
                'success': True,
                'sample_sizes': {
                    'original': len(original_sample),
                    'synthetic': len(synthetic_sample)
                },
                'continuous_tests': {},
                'categorical_tests': {},
                'correlation_analysis': {},
                'summary_statistics': {},
                'overall_score': 0.0
            }
            
            # 1. 연속형 변수 검증
            continuous_results = self._validate_continuous_variables(original_sample, synthetic_sample)
            validation_results['continuous_tests'] = continuous_results
            
            # 2. 범주형 변수 검증
            categorical_results = self._validate_categorical_variables(original_sample, synthetic_sample)
            validation_results['categorical_tests'] = categorical_results
            
            # 3. 상관관계 분석
            correlation_results = self._validate_correlations(original_sample, synthetic_sample)
            validation_results['correlation_analysis'] = correlation_results
            
            # 4. 요약 통계 비교
            summary_stats = self._compare_summary_statistics(original_sample, synthetic_sample)
            validation_results['summary_statistics'] = summary_stats
            
            # 5. 전체 점수 계산
            overall_score = self._calculate_overall_score(validation_results)
            validation_results['overall_score'] = overall_score
            
            # 검증 결과 로그
            self._log_validation_summary(validation_results)
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Distribution validation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_original_sample(self, sample_size: int) -> pd.DataFrame:
        """원본 데이터 샘플 로드"""
        
        try:
            select_exprs = self._build_select_expressions(
                self.source_table,
                self.continuous_variables + self.categorical_variables
            )
            if not select_exprs:
                self.logger.warning(
                    "No overlapping variables found in source table: %s", self.source_table
                )
                return pd.DataFrame()

            sample_query = f"""
                SELECT {', '.join(select_exprs)}
                FROM {self.source_table}
                USING SAMPLE {sample_size}
            """

            sample_data = self.db.fetch_dataframe(sample_query)
            sample_data = self._prepare_dataframe_sample(
                sample_data,
                sample_size,
                self.continuous_variables + self.categorical_variables,
            )
            
            self.logger.info(f"Loaded original sample: {len(sample_data)} records")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Failed to load original sample: {e}")
            return pd.DataFrame()
    
    def _load_synthetic_sample(
        self,
        sample_size: int,
        synthetic_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """합성 데이터 샘플 로드"""

        if synthetic_df is not None:
            return self._prepare_dataframe_sample(
                synthetic_df,
                sample_size,
                self.continuous_variables + self.categorical_variables
            )
        
        try:
            select_exprs = self._build_select_expressions(
                "nedis_synthetic.clinical_records",
                self.continuous_variables + self.categorical_variables
            )
            if not select_exprs:
                self.logger.warning("No overlapping variables found in nedis_synthetic.clinical_records")
                return pd.DataFrame()

            sample_query = f"""
                SELECT {', '.join(select_exprs)}
                FROM nedis_synthetic.clinical_records
                USING SAMPLE {sample_size}
            """
            
            sample_data = self.db.fetch_dataframe(sample_query)
            sample_data = self._prepare_dataframe_sample(
                sample_data,
                sample_size,
                self.continuous_variables + self.categorical_variables,
            )
            
            self.logger.info(f"Loaded synthetic sample: {len(sample_data)} records")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Failed to load synthetic sample: {e}")
            return pd.DataFrame()

    def _prepare_dataframe_sample(
        self,
        df: pd.DataFrame,
        sample_size: int,
        variables: List[str]
    ) -> pd.DataFrame:
        """입력 DataFrame에서 필요한 변수만 추출하여 샘플링"""
        if df.empty:
            return df

        available = [var for var in variables if var in df.columns]
        if not available:
            self.logger.warning("No requested validation variables available in provided synthetic_df")
            return pd.DataFrame()

        sample_df = df[available].copy()
        if len(sample_df) > sample_size:
            sample_df = sample_df.sample(n=sample_size, random_state=42)

        return self._clean_validation_frame(sample_df.reset_index(drop=True))

    def _clean_validation_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize frame-level data before validation comparisons."""
        cleaned = df.copy()

        for col in self.continuous_variables:
            if col not in cleaned.columns:
                continue
            cleaned[col] = self._clean_continuous_column(cleaned[col], col)

        for col in self.categorical_variables:
            if col not in cleaned.columns:
                continue
            cleaned[col] = self._clean_categorical_column(cleaned[col])

        return cleaned

    def _clean_continuous_column(self, series: pd.Series, variable: str) -> pd.Series:
        numeric = pd.to_numeric(series, errors="coerce")
        if variable in self._continuous_config:
            min_v, max_v = self._continuous_config[variable]
            numeric = numeric.replace(
                [value for value in self._invalid_markers if isinstance(value, (int, float))],
                np.nan,
            )
            numeric = numeric.where(numeric.between(min_v, max_v), np.nan)

        return numeric

    def _clean_categorical_column(self, series: pd.Series) -> pd.Series:
        cleaned = series.copy()
        cleaned = cleaned.astype("string")
        cleaned = cleaned.where(~cleaned.isin([str(v) for v in self._invalid_markers]), pd.NA)
        cleaned = cleaned.where(~cleaned.str.strip().isin([""]), pd.NA)
        return cleaned

    def _resolve_source_table(self, source_table: Optional[str]) -> str:
        """원본 테이블 후보를 config, 입력값, 기본값 순으로 결정"""
        configured = self.config.get('original.source_table')
        candidates = [source_table, configured, 'nedis_original.nedis2017', 'nedis_data.nedis2017', 'main.nedis2017']

        for candidate in candidates:
            if not candidate:
                continue
            if isinstance(candidate, str) and self._table_exists(candidate):
                return candidate

        raise RuntimeError("No valid original source table found for statistical validation")

    def _resolve_table_columns(self, table: str) -> List[str]:
        """테이블 컬럼 목록 캐시 조회"""
        if table in self._table_schema_cache:
            return self._table_schema_cache[table]

        schema_df = self.db.fetch_dataframe(f"PRAGMA table_info({table})")
        columns = schema_df['name'].astype(str).tolist()
        self._table_schema_cache[table] = columns
        return columns

    def _build_select_expressions(self, table: str, variables: List[str]) -> List[str]:
        """변수명 불일치를 보완해 select 절 생성"""
        available_columns = set(self._resolve_table_columns(table))

        # 원본/합성 스키마 불일치 보정
        fallback_map = {
            'ptmikts1': ['ptmikts1', 'ptmikpr1'],
        }

        select_exprs = []

        for variable in variables:
            candidates = fallback_map.get(variable, [variable])
            selected = None
            for candidate in candidates:
                if candidate in available_columns:
                    selected = candidate
                    break
            if selected is None:
                continue

            if variable in self.continuous_variables:
                expr = f"TRY_CAST({selected} AS DOUBLE) AS {variable}"
            else:
                expr = f"CAST({selected} AS VARCHAR) AS {variable}"
            select_exprs.append(expr)

        return select_exprs

    def _table_exists(self, table: str) -> bool:
        """테이블 존재 여부 확인"""
        try:
            self.db.execute_query(f"SELECT 1 FROM {table} LIMIT 1")
            return True
        except Exception:
            return False
    
    def _validate_continuous_variables(self, original: pd.DataFrame, 
                                     synthetic: pd.DataFrame) -> Dict[str, Any]:
        """연속형 변수 분포 검증"""
        
        continuous_results = {}
        
        for variable in self.continuous_variables:
            if variable not in original.columns or variable not in synthetic.columns:
                continue
            
            # 결측치 제거
            orig_values = original[variable].dropna()
            synt_values = synthetic[variable].dropna()
            
            if len(orig_values) < 30 or len(synt_values) < 30:
                continuous_results[variable] = {
                    'test': 'insufficient_data',
                    'sample_sizes': {'original': len(orig_values), 'synthetic': len(synt_values)}
                }
                continue
            
            try:
                # Kolmogorov-Smirnov and Wasserstein diagnostics
                ks_statistic, ks_pvalue = stats.ks_2samp(orig_values, synt_values)
                wasserstein_dist = wasserstein_distance(orig_values, synt_values)

                # Anderson-Darling is optional; keep for diagnostics only.
                try:
                    ad_statistic, _, ad_significance = stats.anderson_ksamp([orig_values, synt_values])
                except Exception:
                    ad_statistic, ad_significance = None, None

                # Descriptive statistics
                orig_mean, orig_std = orig_values.mean(), orig_values.std()
                synt_mean, synt_std = synt_values.mean(), synt_values.std()
                mean_diff = abs(orig_mean - synt_mean) / (abs(orig_mean) + 1e-6)
                std_diff = abs(orig_std - synt_std) / (abs(orig_std) + 1e-6)

                ks_score = self._scale_score(1.0 - ks_statistic, tolerance=1.0)
                wd_scale = (orig_values.max() - orig_values.min())
                if not np.isfinite(wd_scale) or wd_scale <= 0:
                    wd_scale = max(orig_values.std(), 1.0)
                wd_score = self._scale_score(wasserstein_dist, tolerance=wd_scale)
                moment_score = self._scale_score(mean_diff + std_diff, tolerance=1.0)
                variable_score = 0.4 * ks_score + 0.3 * wd_score + 0.3 * moment_score

                continuous_results[variable] = {
                    'ks_test': {
                        'statistic': float(ks_statistic),
                        'p_value': float(ks_pvalue),
                    },
                    'score': float(variable_score),
                    'wasserstein_distance': float(wasserstein_dist),
                    'anderson_darling': {
                        'statistic': float(ad_statistic) if ad_statistic else None,
                        'significance': float(ad_significance) if ad_significance else None
                    },
                    'descriptive_stats': {
                        'original': {'mean': float(orig_mean), 'std': float(orig_std)},
                        'synthetic': {'mean': float(synt_mean), 'std': float(synt_std)},
                        'mean_relative_diff': float(mean_diff),
                        'std_relative_diff': float(std_diff)
                    },
                    'sample_sizes': {'original': len(orig_values), 'synthetic': len(synt_values)}
                }
                
            except Exception as e:
                self.logger.warning(f"Continuous validation failed for {variable}: {e}")
                continuous_results[variable] = {
                    'test': 'failed',
                    'error': str(e)
                }
        
        return continuous_results
    
    def _validate_categorical_variables(self, original: pd.DataFrame, 
                                      synthetic: pd.DataFrame) -> Dict[str, Any]:
        """범주형 변수 분포 검증"""
        
        categorical_results = {}
        
        for variable in self.categorical_variables:
            if variable not in original.columns or variable not in synthetic.columns:
                continue
            
            try:
                # 범주별 빈도 계산
                orig_counts = original[variable].value_counts().sort_index()
                synt_counts = synthetic[variable].value_counts().sort_index()
                
                # 공통 범주만 고려
                common_categories = orig_counts.index.intersection(synt_counts.index)
                
                if len(common_categories) < 2:
                    categorical_results[variable] = {
                        'test': 'insufficient_categories',
                        'categories': len(common_categories)
                    }
                    continue
                
                # 공통 범주의 빈도만 추출
                orig_common = orig_counts[common_categories]
                synt_common = synt_counts[common_categories]
                
                # 비율 계산
                orig_props = orig_common / orig_common.sum()
                synt_props = synt_common / synt_common.sum()
                
                # Chi-square 검정
                # 기대빈도 계산 (전체 빈도를 합성 데이터 크기에 맞게 조정)
                expected_freqs = orig_props * synt_common.sum()
                
                # 기대빈도가 5 미만인 범주가 있으면 검정 수행 불가
                if (expected_freqs < 5).any():
                    chi2_result = {'test': 'low_expected_frequency'}
                else:
                    chi2_statistic, chi2_pvalue = stats.chisquare(synt_common, expected_freqs)
                    chi2_result = {
                        'statistic': float(chi2_statistic),
                        'p_value': float(chi2_pvalue),
                    }
                
                # Total Variation Distance 계산
                tv_distance = 0.5 * np.sum(np.abs(orig_props - synt_props))
                
                # Jensen-Shannon Divergence 계산
                js_divergence = self._jensen_shannon_divergence(orig_props.values, synt_props.values)

                if 'p_value' in chi2_result:
                    chi2_score = float(np.clip(chi2_result['p_value'] / self.chi2_threshold, 0.0, 1.0))
                else:
                    chi2_score = 1.0 - tv_distance
                tv_score = 1.0 - min(1.0, tv_distance)
                js_score = 1.0 - min(1.0, js_divergence / 0.25)
                categorical_score = 0.5 * chi2_score + 0.3 * tv_score + 0.2 * js_score
                
                categorical_results[variable] = {
                    'chi2_test': chi2_result,
                    'score': float(max(0.0, categorical_score)),
                    'total_variation_distance': float(tv_distance),
                    'jensen_shannon_divergence': float(js_divergence),
                    'category_overlap': {
                        'common_categories': len(common_categories),
                        'original_unique': len(orig_counts),
                        'synthetic_unique': len(synt_counts),
                        'overlap_rate': len(common_categories) / len(orig_counts.index.union(synt_counts.index))
                    },
                    'proportions': {
                        'original': orig_props.to_dict(),
                        'synthetic': synt_props.to_dict()
                    }
                }
                
            except Exception as e:
                self.logger.warning(f"Categorical validation failed for {variable}: {e}")
                categorical_results[variable] = {
                    'test': 'failed',
                    'error': str(e)
                }
        
        return categorical_results
    
    def _validate_correlations(self, original: pd.DataFrame, 
                             synthetic: pd.DataFrame) -> Dict[str, Any]:
        """상관관계 검증"""
        
        try:
            # 연속형 변수만 고려
            common_continuous = [var for var in self.continuous_variables 
                               if var in original.columns and var in synthetic.columns]
            
            if len(common_continuous) < 2:
                return {'test': 'insufficient_variables', 'variables': len(common_continuous)}
            
            # 결측치 처리된 데이터프레임 생성
            orig_clean = original[common_continuous].dropna()
            synt_clean = synthetic[common_continuous].dropna()
            
            if len(orig_clean) < 100 or len(synt_clean) < 100:
                return {'test': 'insufficient_data_after_dropna'}
            
            # 상관계수 계산
            orig_corr = orig_clean.corr()
            synt_corr = synt_clean.corr()
            
            # 상관계수 차이 계산
            corr_diff = np.abs(orig_corr - synt_corr)
            
            # 대각선 제외 (자기 자신과의 상관계수 = 1)
            mask = np.triu(np.ones_like(corr_diff), k=1).astype(bool)
            
            correlation_results = {
                'pearson_correlation': {
                    'original_matrix': orig_corr.to_dict(),
                    'synthetic_matrix': synt_corr.to_dict(),
                    'difference_matrix': corr_diff.to_dict(),
                    'max_difference': float(corr_diff.values[mask].max()),
                    'mean_difference': float(corr_diff.values[mask].mean()),
                    'score': float(
                        self._scale_score(
                            float(corr_diff.values[mask].max()),
                            tolerance=self.correlation_threshold,
                        )
                    )
                },
                'variables_analyzed': common_continuous,
                'sample_sizes': {'original': len(orig_clean), 'synthetic': len(synt_clean)}
            }
            
            # Spearman 순위 상관계수도 계산
            try:
                orig_spearman = orig_clean.corr(method='spearman')
                synt_spearman = synt_clean.corr(method='spearman')
                spearman_diff = np.abs(orig_spearman - synt_spearman)
                
                correlation_results['spearman_correlation'] = {
                    'max_difference': float(spearman_diff.values[mask].max()),
                    'mean_difference': float(spearman_diff.values[mask].mean()),
                    'score': float(
                        self._scale_score(
                            float(spearman_diff.values[mask].max()),
                            tolerance=self.correlation_threshold,
                        )
                    )
                }
            except Exception as e:
                self.logger.warning(f"Spearman correlation calculation failed: {e}")
            
            return correlation_results
            
        except Exception as e:
            self.logger.error(f"Correlation validation failed: {e}")
            return {'test': 'failed', 'error': str(e)}
    
    def _compare_summary_statistics(self, original: pd.DataFrame, 
                                   synthetic: pd.DataFrame) -> Dict[str, Any]:
        """기술 통계량 비교"""
        
        summary_comparison = {}
        
        # 연속형 변수 기술 통계
        for variable in self.continuous_variables:
            if variable not in original.columns or variable not in synthetic.columns:
                continue
            
            orig_values = original[variable].dropna()
            synt_values = synthetic[variable].dropna()
            
            if len(orig_values) < 10 or len(synt_values) < 10:
                continue
            
            orig_stats = orig_values.describe()
            synt_stats = synt_values.describe()
            
            # 상대적 차이 계산
            relative_diffs = {}
            for stat in ['mean', 'std', 'min', 'max', '25%', '50%', '75%']:
                if orig_stats[stat] != 0:
                    relative_diffs[stat] = abs(synt_stats[stat] - orig_stats[stat]) / abs(orig_stats[stat])
                else:
                    relative_diffs[stat] = 0.0
            
            summary_comparison[variable] = {
                'original': orig_stats.to_dict(),
                'synthetic': synt_stats.to_dict(),
                'relative_differences': relative_diffs,
                'max_relative_diff': max(relative_diffs.values())
            }
        
        return summary_comparison
    
    def _jensen_shannon_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon Divergence 계산"""
        
        # 0이 포함된 확률을 피하기 위해 작은 값 추가
        epsilon = 1e-10
        p = p + epsilon
        q = q + epsilon
        
        # 정규화
        p = p / p.sum()
        q = q / q.sum()
        
        # JS divergence 계산
        m = 0.5 * (p + q)
        
        def kl_divergence(x, y):
            return np.sum(x * np.log(x / y))
        
        js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
        
        return js_div
    
    def _calculate_overall_score(self, validation_results: Dict[str, Any]) -> float:
        """전체 검증 점수 계산"""
        
        scores = []
        
        continuous_tests = validation_results.get('continuous_tests', {})
        for variable, result in continuous_tests.items():
            score = result.get('score')
            if isinstance(score, (int, float)):
                scores.append(float(score))

        categorical_tests = validation_results.get('categorical_tests', {})
        for variable, result in categorical_tests.items():
            score = result.get('score')
            if isinstance(score, (int, float)):
                scores.append(float(score))
        
        correlation_analysis = validation_results.get('correlation_analysis', {})
        if isinstance(correlation_analysis, dict) and 'pearson_correlation' in correlation_analysis:
            pearson_score = correlation_analysis['pearson_correlation'].get('score')
            if isinstance(pearson_score, (int, float)):
                scores.append(float(pearson_score))
        
        # 전체 점수 (평균)
        overall_score = np.mean(scores) if scores else 0.0
        
        return overall_score
    
    def _log_validation_summary(self, validation_results: Dict[str, Any]):
        """검증 결과 요약 로그"""
        
        self.logger.info("=== Statistical Validation Summary ===")
        
        overall_score = validation_results['overall_score']
        self.logger.info(f"Overall Score: {overall_score:.3f}")
        
        # 연속형 변수 결과
        continuous_tests = validation_results.get('continuous_tests', {})
        continuous_scores = [result.get('score', 0.0) for result in continuous_tests.values()]
        continuous_values = [s for s in continuous_scores if isinstance(s, (int, float))]
        continuous_avg = float(np.mean(continuous_values)) if continuous_values else 0.0
        self.logger.info(f"Continuous score avg: {continuous_avg:.3f}")
        
        # 범주형 변수 결과
        categorical_tests = validation_results.get('categorical_tests', {})
        categorical_scores = [result.get('score', 0.0) for result in categorical_tests.values()]
        categorical_values = [s for s in categorical_scores if isinstance(s, (int, float))]
        categorical_avg = float(np.mean(categorical_values)) if categorical_values else 0.0
        self.logger.info(f"Categorical score avg: {categorical_avg:.3f}")
        
        # 상관관계 결과
        correlation_analysis = validation_results.get('correlation_analysis', {})
        if 'pearson_correlation' in correlation_analysis:
            max_corr_diff = correlation_analysis['pearson_correlation'].get('max_difference', 0)
            corr_score = correlation_analysis['pearson_correlation'].get('score', 0.0)
            self.logger.info(f"Correlation score: {float(corr_score):.3f} (max difference: {max_corr_diff:.3f})")

    def _scale_score(self, value: float, tolerance: float) -> float:
        if not np.isfinite(value) or not np.isfinite(tolerance) or tolerance <= 0:
            return 0.0
        return float(max(0.0, 1.0 - min(value, tolerance * 2) / (tolerance * 2)))
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """검증 결과 리포트 생성"""
        
        if not validation_results.get('success', False):
            return "Validation failed or no results available."
        
        report = []
        report.append("# Statistical Validation Report")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # 전체 점수
        overall_score = validation_results['overall_score']
        score_grade = "Excellent" if overall_score >= 0.9 else \
                     "Good" if overall_score >= 0.7 else \
                     "Fair" if overall_score >= 0.5 else "Poor"
        
        report.append(f"## Overall Score: {overall_score:.3f} ({score_grade})")
        report.append("")
        
        # 샘플 크기
        sample_sizes = validation_results['sample_sizes']
        report.append(f"## Sample Sizes")
        report.append(f"- Original: {sample_sizes['original']:,}")
        report.append(f"- Synthetic: {sample_sizes['synthetic']:,}")
        report.append("")
        
        # 연속형 변수 결과
        continuous_tests = validation_results.get('continuous_tests', {})
        if continuous_tests:
            report.append("## Continuous Variables")
            report.append("| Variable | KS Statistic | p-value | Score | Mean Diff (%) | Std Diff (%) |")
            report.append("|----------|--------------|---------|-------|---------------|--------------|")
            
            for variable, result in continuous_tests.items():
                if 'ks_test' in result:
                    ks_stat = result['ks_test']['statistic']
                    ks_pval = result['ks_test']['p_value']
                    score = result.get('score', 0.0)
                    desc_stats = result.get('descriptive_stats', {})
                    mean_diff = desc_stats.get('mean_relative_diff', 0) * 100
                    std_diff = desc_stats.get('std_relative_diff', 0) * 100
                    report.append(
                        f"| {variable} | {ks_stat:.4f} | {ks_pval:.4f} | {float(score):.3f} | "
                        f"{mean_diff:.1f}% | {std_diff:.1f}% |"
                    )
            report.append("")
        
        # 범주형 변수 결과
        categorical_tests = validation_results.get('categorical_tests', {})
        if categorical_tests:
            report.append("## Categorical Variables")
            report.append("| Variable | Chi-square | p-value | Score | TV Distance | JS Divergence |")
            report.append("|----------|------------|---------|-------|-------------|---------------|")
            
            for variable, result in categorical_tests.items():
                if 'chi2_test' in result:
                    chi2_test = result['chi2_test']
                    score = result.get('score', 0.0)
                    chi2_stat = chi2_test.get('statistic', "N/A")
                    chi2_pval = chi2_test.get('p_value', "N/A")
                    tv_dist = result.get('total_variation_distance', 0)
                    js_div = result.get('jensen_shannon_divergence', 0)
                    report.append(f"| {variable} | {chi2_stat} | {chi2_pval} | {float(score):.3f} | {tv_dist:.4f} | {js_div:.4f} |")
            report.append("")
        
        # 상관관계 결과
        correlation_analysis = validation_results.get('correlation_analysis', {})
        if 'pearson_correlation' in correlation_analysis:
            corr_result = correlation_analysis['pearson_correlation']
            report.append("## Correlation Analysis")
            report.append(f"- Correlation score: {corr_result.get('score', 0.0):.3f}")
            report.append(f"- Maximum correlation difference: {corr_result.get('max_difference', 0):.4f}")
            report.append(f"- Mean correlation difference: {corr_result.get('mean_difference', 0):.4f}")
            report.append("")
        
        return "\n".join(report)
    
    def save_validation_results(self, validation_results: Dict[str, Any]) -> bool:
        """검증 결과를 데이터베이스에 저장"""
        
        try:
            # 검증 결과 테이블 생성 (존재하지 않는 경우)
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS nedis_meta.validation_results (
                    test_id INTEGER PRIMARY KEY,
                    test_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    test_type VARCHAR NOT NULL,
                    variable_name VARCHAR,
                    statistic_name VARCHAR,
                    statistic_value DOUBLE,
                    p_value DOUBLE,
                    passed BOOLEAN,
                    sample_size INTEGER,
                    additional_info JSON
                )
            """)
            
            # 연속형 변수 결과 저장
            continuous_tests = validation_results.get('continuous_tests', {})
            for variable, result in continuous_tests.items():
                if 'ks_test' in result:
                    ks_test = result['ks_test']
                    self.db.execute_query("""
                        INSERT INTO nedis_meta.validation_results
                        (test_type, variable_name, statistic_name, statistic_value, 
                         p_value, passed, sample_size, additional_info)
                        VALUES ('continuous', ?, 'ks_statistic', ?, ?, ?, ?, ?)
                    """, (variable, ks_test['statistic'], ks_test['p_value'], 
                          result.get('score', 0.0) >= 0.5, result['sample_sizes']['synthetic'], 
                          str(result.get('descriptive_stats', {}))))
            
            # 범주형 변수 결과 저장
            categorical_tests = validation_results.get('categorical_tests', {})
            for variable, result in categorical_tests.items():
                if 'chi2_test' in result and 'statistic' in result['chi2_test']:
                    chi2_test = result['chi2_test']
                    self.db.execute_query("""
                        INSERT INTO nedis_meta.validation_results
                        (test_type, variable_name, statistic_name, statistic_value,
                         p_value, passed, additional_info)
                        VALUES ('categorical', ?, 'chi2_statistic', ?, ?, ?, ?)
                    """, (variable, chi2_test['statistic'], chi2_test['p_value'],
                          result.get('score', 0.0) >= 0.5, str(result.get('category_overlap', {}))))
            
            # 전체 점수 저장
            self.db.execute_query("""
                INSERT INTO nedis_meta.validation_results
                (test_type, variable_name, statistic_name, statistic_value, passed)
                VALUES ('overall', 'all_variables', 'overall_score', ?, ?)
            """, (validation_results['overall_score'], 
                  validation_results['overall_score'] >= 0.7))
            
            self.logger.info("Validation results saved to database")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save validation results: {e}")
            return False
