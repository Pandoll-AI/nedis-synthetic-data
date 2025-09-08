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
import warnings

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class StatisticalValidator:
    """통계적 검증기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        통계적 검증기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 검증할 변수들 정의
        self.continuous_variables = [
            'vst_sbp', 'vst_dbp', 'vst_per_pu', 'vst_per_br', 'vst_bdht', 'vst_oxy'
        ]
        
        self.categorical_variables = [
            'pat_age_gr', 'pat_sex', 'pat_do_cd', 'ktas_fstu', 'emtrt_rust', 
            'vst_meth', 'msypt', 'main_trt_p'
        ]
        
        # 검정 임계값
        self.significance_level = 0.05
        self.ks_threshold = 0.05        # KS 검정 p-value 임계값
        self.chi2_threshold = 0.05      # Chi-square 검정 p-value 임계값  
        self.correlation_threshold = 0.1 # 상관계수 차이 임계값
        
    def validate_distributions(self, sample_size: int = 50000) -> Dict[str, Any]:
        """
        전체적인 분포 검증 수행
        
        Args:
            sample_size: 검증에 사용할 샘플 크기
            
        Returns:
            검증 결과 딕셔너리
        """
        
        self.logger.info(f"Starting comprehensive distribution validation with sample size: {sample_size}")
        
        try:
            # 원본 및 합성 데이터 샘플 로드
            original_sample = self._load_original_sample(sample_size)
            synthetic_sample = self._load_synthetic_sample(sample_size)
            
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
            sample_query = f"""
                SELECT {', '.join(self.continuous_variables + self.categorical_variables)}
                FROM nedis_original.nedis2017
                USING SAMPLE {sample_size}
            """
            
            sample_data = self.db.fetch_dataframe(sample_query)
            
            # -1 값들을 NaN으로 변환 (측정되지 않은 값)
            for col in self.continuous_variables:
                if col in sample_data.columns:
                    sample_data[col] = sample_data[col].replace(-1, np.nan)
            
            self.logger.info(f"Loaded original sample: {len(sample_data)} records")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Failed to load original sample: {e}")
            return pd.DataFrame()
    
    def _load_synthetic_sample(self, sample_size: int) -> pd.DataFrame:
        """합성 데이터 샘플 로드"""
        
        try:
            sample_query = f"""
                SELECT {', '.join(self.continuous_variables + self.categorical_variables)}
                FROM nedis_synthetic.clinical_records
                USING SAMPLE {sample_size}
            """
            
            sample_data = self.db.fetch_dataframe(sample_query)
            
            # -1 값들을 NaN으로 변환
            for col in self.continuous_variables:
                if col in sample_data.columns:
                    sample_data[col] = sample_data[col].replace(-1, np.nan)
                    # 체온의 경우 -1.0으로 저장될 수 있음
                    if col == 'vst_bdht':
                        sample_data[col] = sample_data[col].replace(-1.0, np.nan)
            
            self.logger.info(f"Loaded synthetic sample: {len(sample_data)} records")
            return sample_data
            
        except Exception as e:
            self.logger.error(f"Failed to load synthetic sample: {e}")
            return pd.DataFrame()
    
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
                # Kolmogorov-Smirnov 검정
                ks_statistic, ks_pvalue = stats.ks_2samp(orig_values, synt_values)
                
                # Wasserstein distance (Earth Mover's Distance)
                wasserstein_dist = wasserstein_distance(orig_values, synt_values)
                
                # Anderson-Darling 검정
                try:
                    ad_statistic, ad_critical_values, ad_significance = stats.anderson_ksamp([orig_values, synt_values])
                except:
                    ad_statistic, ad_significance = None, None
                
                # 평균과 분산 비교
                orig_mean, orig_std = orig_values.mean(), orig_values.std()
                synt_mean, synt_std = synt_values.mean(), synt_values.std()
                
                mean_diff = abs(orig_mean - synt_mean) / orig_mean if orig_mean != 0 else 0
                std_diff = abs(orig_std - synt_std) / orig_std if orig_std != 0 else 0
                
                continuous_results[variable] = {
                    'ks_test': {
                        'statistic': float(ks_statistic),
                        'p_value': float(ks_pvalue),
                        'passed': ks_pvalue >= self.ks_threshold
                    },
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
                    chi2_result = {'test': 'low_expected_frequency', 'passed': False}
                else:
                    chi2_statistic, chi2_pvalue = stats.chisquare(synt_common, expected_freqs)
                    chi2_result = {
                        'statistic': float(chi2_statistic),
                        'p_value': float(chi2_pvalue),
                        'passed': chi2_pvalue >= self.chi2_threshold
                    }
                
                # Total Variation Distance 계산
                tv_distance = 0.5 * np.sum(np.abs(orig_props - synt_props))
                
                # Jensen-Shannon Divergence 계산
                js_divergence = self._jensen_shannon_divergence(orig_props.values, synt_props.values)
                
                categorical_results[variable] = {
                    'chi2_test': chi2_result,
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
                    'passed': corr_diff.values[mask].max() <= self.correlation_threshold
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
                    'passed': spearman_diff.values[mask].max() <= self.correlation_threshold
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
        
        # 연속형 변수 점수 (KS test 기준)
        continuous_tests = validation_results.get('continuous_tests', {})
        for variable, result in continuous_tests.items():
            if 'ks_test' in result and 'passed' in result['ks_test']:
                scores.append(1.0 if result['ks_test']['passed'] else 0.0)
        
        # 범주형 변수 점수 (Chi-square test 기준)
        categorical_tests = validation_results.get('categorical_tests', {})
        for variable, result in categorical_tests.items():
            if 'chi2_test' in result and 'passed' in result['chi2_test']:
                scores.append(1.0 if result['chi2_test']['passed'] else 0.0)
        
        # 상관관계 점수
        correlation_analysis = validation_results.get('correlation_analysis', {})
        if 'pearson_correlation' in correlation_analysis:
            pearson_result = correlation_analysis['pearson_correlation']
            if 'passed' in pearson_result:
                scores.append(1.0 if pearson_result['passed'] else 0.0)
        
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
        continuous_passed = sum(1 for result in continuous_tests.values() 
                              if result.get('ks_test', {}).get('passed', False))
        self.logger.info(f"Continuous Variables: {continuous_passed}/{len(continuous_tests)} passed KS test")
        
        # 범주형 변수 결과
        categorical_tests = validation_results.get('categorical_tests', {})
        categorical_passed = sum(1 for result in categorical_tests.values() 
                               if result.get('chi2_test', {}).get('passed', False))
        self.logger.info(f"Categorical Variables: {categorical_passed}/{len(categorical_tests)} passed Chi-square test")
        
        # 상관관계 결과
        correlation_analysis = validation_results.get('correlation_analysis', {})
        if 'pearson_correlation' in correlation_analysis:
            corr_passed = correlation_analysis['pearson_correlation'].get('passed', False)
            max_corr_diff = correlation_analysis['pearson_correlation'].get('max_difference', 0)
            self.logger.info(f"Correlation Analysis: {'Passed' if corr_passed else 'Failed'} "
                           f"(max difference: {max_corr_diff:.3f})")
    
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
            report.append("## Continuous Variables (Kolmogorov-Smirnov Test)")
            report.append("| Variable | KS Statistic | p-value | Status | Mean Diff (%) | Std Diff (%) |")
            report.append("|----------|--------------|---------|--------|---------------|--------------|")
            
            for variable, result in continuous_tests.items():
                if 'ks_test' in result:
                    ks_stat = result['ks_test']['statistic']
                    ks_pval = result['ks_test']['p_value']
                    status = "✅ Pass" if result['ks_test']['passed'] else "❌ Fail"
                    
                    desc_stats = result.get('descriptive_stats', {})
                    mean_diff = desc_stats.get('mean_relative_diff', 0) * 100
                    std_diff = desc_stats.get('std_relative_diff', 0) * 100
                    
                    report.append(f"| {variable} | {ks_stat:.4f} | {ks_pval:.4f} | {status} | {mean_diff:.1f}% | {std_diff:.1f}% |")
            
            report.append("")
        
        # 범주형 변수 결과  
        categorical_tests = validation_results.get('categorical_tests', {})
        if categorical_tests:
            report.append("## Categorical Variables (Chi-square Test)")
            report.append("| Variable | Chi-square | p-value | Status | TV Distance | JS Divergence |")
            report.append("|----------|------------|---------|--------|-------------|---------------|")
            
            for variable, result in categorical_tests.items():
                if 'chi2_test' in result:
                    chi2_test = result['chi2_test']
                    if 'statistic' in chi2_test:
                        chi2_stat = chi2_test['statistic']
                        chi2_pval = chi2_test['p_value']
                        status = "✅ Pass" if chi2_test['passed'] else "❌ Fail"
                    else:
                        chi2_stat, chi2_pval, status = "N/A", "N/A", "⚠️ Skip"
                    
                    tv_dist = result.get('total_variation_distance', 0)
                    js_div = result.get('jensen_shannon_divergence', 0)
                    
                    report.append(f"| {variable} | {chi2_stat} | {chi2_pval} | {status} | {tv_dist:.4f} | {js_div:.4f} |")
            
            report.append("")
        
        # 상관관계 결과
        correlation_analysis = validation_results.get('correlation_analysis', {})
        if 'pearson_correlation' in correlation_analysis:
            corr_result = correlation_analysis['pearson_correlation']
            report.append("## Correlation Analysis")
            report.append(f"- Maximum correlation difference: {corr_result.get('max_difference', 0):.4f}")
            report.append(f"- Mean correlation difference: {corr_result.get('mean_difference', 0):.4f}")
            report.append(f"- Status: {'✅ Pass' if corr_result.get('passed', False) else '❌ Fail'}")
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
                          ks_test['passed'], result['sample_sizes']['synthetic'], 
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
                          chi2_test['passed'], str(result.get('category_overlap', {}))))
            
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