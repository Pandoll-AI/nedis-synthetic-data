"""
실제 원본 vs 합성 데이터 비교 분석 대시보드 엔진

DuckDB에서 실제 원본 및 합성 데이터를 추출하여 통계적 비교 분석을 수행합니다.
Mock data 사용 금지, 실제 데이터만 사용합니다.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import json
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ComparisonDashboardEngine:
    """원본 vs 합성 데이터 실제 비교 분석 엔진"""
    
    def __init__(self, db_path: str = "nedis_sample.duckdb"):
        """
        비교 분석 엔진 초기화
        
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = db_path
        self.conn = None
        
        # 통계적 임계값 설정
        self.statistical_thresholds = {
            'significance_level': 0.05,
            'effect_size_small': 0.2,
            'effect_size_medium': 0.5,
            'effect_size_large': 0.8,
            'acceptable_deviation': 0.10  # 10% 편차까지 허용
        }
        
        # KTAS 표준 분포 (의료 가이드라인 기반)
        self.ktas_standard = {
            '1': 0.007,  # 소생술
            '2': 0.150,  # 응급 
            '3': 0.350,  # 긴급
            '4': 0.350,  # 준긴급
            '5': 0.143   # 비긴급
        }
    
    def connect_database(self):
        """데이터베이스 연결"""
        try:
            self.conn = duckdb.connect(self.db_path, read_only=True)
            logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    def close_connection(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
    
    def check_data_availability(self) -> Dict[str, Any]:
        """실제 원본 및 합성 데이터 가용성 확인"""
        if not self.conn:
            self.connect_database()
            
        try:
            # 원본 데이터 확인
            original_count = self.conn.execute(
                "SELECT COUNT(*) FROM nedis2017"
            ).fetchone()[0]
            
            # 합성 데이터 확인 시도
            synthetic_count = 0
            synthetic_available = False
            
            try:
                # 여러 가능한 합성 데이터 테이블 확인
                possible_synthetic_tables = [
                    'nedis_synthetic',
                    'synthetic_nedis',
                    'nedis2017_synthetic'
                ]
                
                for table_name in possible_synthetic_tables:
                    try:
                        count = self.conn.execute(
                            f"SELECT COUNT(*) FROM {table_name}"
                        ).fetchone()[0]
                        if count > 0:
                            synthetic_count = count
                            synthetic_available = True
                            break
                    except:
                        continue
                        
            except Exception as e:
                logger.warning(f"Synthetic data not found: {e}")
            
            return {
                'original_available': original_count > 0,
                'original_count': original_count,
                'synthetic_available': synthetic_available,
                'synthetic_count': synthetic_count,
                'can_perform_comparison': original_count > 0 and synthetic_available,
                'status': 'ready' if (original_count > 0 and synthetic_available) else 'partial'
            }
            
        except Exception as e:
            logger.error(f"Data availability check failed: {e}")
            return {
                'original_available': False,
                'synthetic_available': False,
                'can_perform_comparison': False,
                'status': 'error',
                'error': str(e)
            }
    
    def extract_demographic_data(self, data_type: str = 'original') -> Dict[str, Any]:
        """실제 인구학적 데이터 추출"""
        if not self.conn:
            self.connect_database()
            
        try:
            # 테이블 선택
            table_name = 'nedis2017' if data_type == 'original' else 'nedis_synthetic'
            
            # 연령 분포
            age_query = f"""
            SELECT 
                pat_age_gr as age_group,
                COUNT(*) as count
            FROM {table_name}
            WHERE pat_age_gr IS NOT NULL AND pat_age_gr != ''
            GROUP BY pat_age_gr
            ORDER BY pat_age_gr
            """
            
            age_data = self.conn.execute(age_query).fetchall()
            
            # 성별 분포
            sex_query = f"""
            SELECT 
                pat_sex as sex,
                COUNT(*) as count
            FROM {table_name}
            WHERE pat_sex IS NOT NULL AND pat_sex != ''
            GROUP BY pat_sex
            ORDER BY pat_sex
            """
            
            sex_data = self.conn.execute(sex_query).fetchall()
            
            # 지역 분포 (상위 20개)
            region_query = f"""
            SELECT 
                pat_do_cd as region_code,
                COUNT(*) as count
            FROM {table_name}
            WHERE pat_do_cd IS NOT NULL AND pat_do_cd != ''
            GROUP BY pat_do_cd
            ORDER BY count DESC
            LIMIT 20
            """
            
            region_data = self.conn.execute(region_query).fetchall()
            
            return {
                'age_distribution': [
                    {'age_group': row[0], 'count': row[1]} 
                    for row in age_data
                ],
                'sex_distribution': [
                    {'sex': row[0], 'count': row[1]} 
                    for row in sex_data
                ],
                'region_distribution': [
                    {'region_code': row[0], 'count': row[1]} 
                    for row in region_data
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to extract demographic data for {data_type}: {e}")
            return {}
    
    def extract_clinical_data(self, data_type: str = 'original') -> Dict[str, Any]:
        """실제 임상 데이터 추출"""
        if not self.conn:
            self.connect_database()
            
        try:
            table_name = 'nedis2017' if data_type == 'original' else 'nedis_synthetic'
            
            # KTAS 분포
            ktas_query = f"""
            SELECT 
                ktas_no,
                COUNT(*) as count
            FROM {table_name}
            WHERE ktas_no IS NOT NULL AND ktas_no != '' AND ktas_no != '-'
            GROUP BY ktas_no
            ORDER BY ktas_no
            """
            
            ktas_data = self.conn.execute(ktas_query).fetchall()
            
            # 생체징후 평균값
            vitals_query = f"""
            SELECT 
                AVG(CASE WHEN vst_sbp > 0 AND vst_sbp < 300 THEN vst_sbp END) as avg_sbp,
                AVG(CASE WHEN vst_dbp > 0 AND vst_dbp < 200 THEN vst_dbp END) as avg_dbp,
                AVG(CASE WHEN vst_per_pu > 0 AND vst_per_pu < 200 THEN vst_per_pu END) as avg_pulse,
                AVG(CASE WHEN vst_per_br > 0 AND vst_per_br < 60 THEN vst_per_br END) as avg_respiration,
                AVG(CASE WHEN vst_oxy > 0 AND vst_oxy <= 100 THEN vst_oxy END) as avg_oxygen,
                COUNT(CASE WHEN vst_sbp > 0 AND vst_sbp < 300 THEN 1 END) as sbp_records,
                COUNT(CASE WHEN vst_dbp > 0 AND vst_dbp < 200 THEN 1 END) as dbp_records,
                COUNT(CASE WHEN vst_per_pu > 0 AND vst_per_pu < 200 THEN 1 END) as pulse_records,
                COUNT(CASE WHEN vst_per_br > 0 AND vst_per_br < 60 THEN 1 END) as respiration_records,
                COUNT(CASE WHEN vst_oxy > 0 AND vst_oxy <= 100 THEN 1 END) as oxygen_records
            FROM {table_name}
            """
            
            vitals_result = self.conn.execute(vitals_query).fetchone()
            
            # KTAS별 생체징후 패턴
            ktas_vitals_query = f"""
            SELECT 
                ktas_no,
                AVG(CASE WHEN vst_sbp > 0 AND vst_sbp < 300 THEN vst_sbp END) as avg_sbp,
                AVG(CASE WHEN vst_per_pu > 0 AND vst_per_pu < 200 THEN vst_per_pu END) as avg_pulse,
                AVG(CASE WHEN vst_oxy > 0 AND vst_oxy <= 100 THEN vst_oxy END) as avg_oxygen,
                COUNT(*) as count
            FROM {table_name}
            WHERE ktas_no IS NOT NULL AND ktas_no != '' AND ktas_no != '-'
            GROUP BY ktas_no
            ORDER BY ktas_no
            """
            
            ktas_vitals_data = self.conn.execute(ktas_vitals_query).fetchall()
            
            return {
                'ktas_distribution': [
                    {'ktas_no': row[0], 'count': row[1]} 
                    for row in ktas_data
                ],
                'vital_signs_summary': {
                    'avg_sbp': round(vitals_result[0], 2) if vitals_result[0] else None,
                    'avg_dbp': round(vitals_result[1], 2) if vitals_result[1] else None,
                    'avg_pulse': round(vitals_result[2], 2) if vitals_result[2] else None,
                    'avg_respiration': round(vitals_result[3], 2) if vitals_result[3] else None,
                    'avg_oxygen': round(vitals_result[4], 2) if vitals_result[4] else None,
                    'completion_rates': {
                        'sbp': vitals_result[5],
                        'dbp': vitals_result[6],
                        'pulse': vitals_result[7],
                        'respiration': vitals_result[8],
                        'oxygen': vitals_result[9]
                    }
                },
                'ktas_vital_patterns': [
                    {
                        'ktas_no': row[0],
                        'avg_sbp': round(row[1], 2) if row[1] else None,
                        'avg_pulse': round(row[2], 2) if row[2] else None,
                        'avg_oxygen': round(row[3], 2) if row[3] else None,
                        'count': row[4]
                    }
                    for row in ktas_vitals_data
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to extract clinical data for {data_type}: {e}")
            return {}
    
    def extract_temporal_data(self, data_type: str = 'original') -> Dict[str, Any]:
        """실제 시간적 패턴 데이터 추출"""
        if not self.conn:
            self.connect_database()
            
        try:
            table_name = 'nedis2017' if data_type == 'original' else 'nedis_synthetic'
            
            # 월별 패턴
            monthly_query = f"""
            SELECT 
                CAST(SUBSTR(vst_dt, 5, 2) AS INTEGER) as month,
                COUNT(*) as count
            FROM {table_name}
            WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) = 8
            GROUP BY CAST(SUBSTR(vst_dt, 5, 2) AS INTEGER)
            ORDER BY month
            """
            
            monthly_data = self.conn.execute(monthly_query).fetchall()
            
            # 요일별 패턴 (날짜로부터 요일 계산)
            weekday_query = f"""
            SELECT 
                DAYOFWEEK(CAST(vst_dt AS DATE)) as weekday,
                COUNT(*) as count
            FROM {table_name}
            WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) = 8
                AND TRY_CAST(vst_dt AS DATE) IS NOT NULL
            GROUP BY DAYOFWEEK(CAST(vst_dt AS DATE))
            ORDER BY weekday
            """
            
            try:
                weekday_data = self.conn.execute(weekday_query).fetchall()
            except:
                # 대체 방법으로 요일 계산
                weekday_data = []
            
            # 시간별 패턴
            hourly_query = f"""
            SELECT 
                CAST(SUBSTR(vst_tm, 1, 2) AS INTEGER) as hour,
                COUNT(*) as count
            FROM {table_name}
            WHERE vst_tm IS NOT NULL AND LENGTH(vst_tm) >= 4
                AND SUBSTR(vst_tm, 1, 2) BETWEEN '00' AND '23'
            GROUP BY CAST(SUBSTR(vst_tm, 1, 2) AS INTEGER)
            ORDER BY hour
            """
            
            hourly_data = self.conn.execute(hourly_query).fetchall()
            
            return {
                'monthly_pattern': [
                    {'month': row[0], 'count': row[1]} 
                    for row in monthly_data
                ],
                'weekday_pattern': [
                    {'weekday': row[0], 'count': row[1]} 
                    for row in weekday_data
                ],
                'hourly_pattern': [
                    {'hour': row[0], 'count': row[1]} 
                    for row in hourly_data
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to extract temporal data for {data_type}: {e}")
            return {}
    
    def perform_statistical_comparison(self, original_data: List[Dict], 
                                     synthetic_data: List[Dict], 
                                     category_field: str, 
                                     count_field: str,
                                     test_name: str) -> Dict[str, Any]:
        """실제 통계적 비교 수행 (카이제곱 검정, KS 검정 등)"""
        try:
            if not synthetic_data:
                return {
                    'test_name': test_name,
                    'status': 'no_synthetic_data',
                    'message': '합성 데이터가 없어 비교를 수행할 수 없습니다.'
                }
            
            # 데이터 정리
            orig_df = pd.DataFrame(original_data)
            synth_df = pd.DataFrame(synthetic_data)
            
            # 공통 카테고리 추출
            all_categories = set(orig_df[category_field].unique()).union(
                set(synth_df[category_field].unique())
            )
            
            # 관찰값 및 기대값 생성
            orig_counts = []
            synth_counts = []
            comparison_details = []
            
            for category in sorted(all_categories):
                orig_count = orig_df[orig_df[category_field] == category][count_field].sum()
                synth_count = synth_df[synth_df[category_field] == category][count_field].sum()
                
                orig_counts.append(orig_count)
                synth_counts.append(synth_count)
            
            # 카이제곱 검정
            chi2, p_value, dof, expected = chi2_contingency([orig_counts, synth_counts])
            
            # Cramér's V 계산 (연관성 강도)
            total_n = sum(orig_counts) + sum(synth_counts)
            cramers_v = np.sqrt(chi2 / (total_n * (min(2, len(all_categories)) - 1)))
            
            # 비율 비교 및 편차 계산
            orig_total = sum(orig_counts)
            synth_total = sum(synth_counts)
            max_deviation = 0
            
            for i, category in enumerate(sorted(all_categories)):
                orig_pct = (orig_counts[i] / orig_total * 100) if orig_total > 0 else 0
                synth_pct = (synth_counts[i] / synth_total * 100) if synth_total > 0 else 0
                deviation = abs(orig_pct - synth_pct)
                max_deviation = max(max_deviation, deviation)
                
                comparison_details.append({
                    'category': str(category),
                    'original_count': orig_counts[i],
                    'synthetic_count': synth_counts[i], 
                    'original_percentage': round(orig_pct, 2),
                    'synthetic_percentage': round(synth_pct, 2),
                    'absolute_deviation': round(deviation, 2)
                })
            
            # 결과 해석
            is_significant = p_value < self.statistical_thresholds['significance_level']
            is_acceptable = max_deviation <= (self.statistical_thresholds['acceptable_deviation'] * 100)
            has_strong_association = cramers_v > 0.1
            
            return {
                'test_name': test_name,
                'status': 'completed',
                'statistical_tests': {
                    'chi_square': {
                        'statistic': float(chi2),
                        'p_value': float(p_value),
                        'degrees_of_freedom': int(dof),
                        'is_significant': is_significant
                    },
                    'cramers_v': {
                        'value': float(cramers_v),
                        'interpretation': 'strong' if cramers_v > 0.3 else 'moderate' if cramers_v > 0.1 else 'weak'
                    },
                    'max_deviation': {
                        'value': float(max_deviation),
                        'is_acceptable': is_acceptable
                    }
                },
                'comparison_details': comparison_details,
                'summary': {
                    'original_total': orig_total,
                    'synthetic_total': synth_total,
                    'categories_compared': len(all_categories),
                    'quality_score': self._calculate_quality_score(max_deviation, p_value, cramers_v)
                },
                'interpretation': {
                    'distributions_similar': not is_significant and is_acceptable,
                    'clinically_acceptable': is_acceptable,
                    'statistical_notes': self._generate_interpretation_notes(is_significant, is_acceptable, cramers_v)
                }
            }
            
        except Exception as e:
            logger.error(f"Statistical comparison failed for {test_name}: {e}")
            return {
                'test_name': test_name,
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_quality_score(self, max_deviation: float, p_value: float, cramers_v: float) -> float:
        """품질 점수 계산 (0-100)"""
        # 편차 점수 (편차가 적을수록 높은 점수)
        deviation_score = max(0, 100 - (max_deviation * 5))
        
        # 유의성 점수 (유의하지 않을수록 높은 점수)
        significance_score = 100 if p_value >= 0.05 else max(0, 100 - ((0.05 - p_value) * 2000))
        
        # 연관성 점수 (적절한 연관성일 때 높은 점수)
        association_score = 100 if 0.1 <= cramers_v <= 0.3 else max(0, 100 - abs(cramers_v - 0.2) * 500)
        
        # 가중 평균
        return round((deviation_score * 0.5 + significance_score * 0.3 + association_score * 0.2), 2)
    
    def _generate_interpretation_notes(self, is_significant: bool, is_acceptable: bool, cramers_v: float) -> List[str]:
        """해석 노트 생성"""
        notes = []
        
        if is_significant:
            notes.append("분포 간 통계적으로 유의한 차이가 있습니다.")
        else:
            notes.append("분포 간 통계적으로 유의한 차이가 없습니다.")
            
        if is_acceptable:
            notes.append("편차가 허용 가능한 범위(10%) 내에 있습니다.")
        else:
            notes.append("편차가 허용 범위를 초과합니다. 합성 모델 조정이 필요합니다.")
            
        if cramers_v > 0.3:
            notes.append("변수 간 강한 연관성이 있습니다.")
        elif cramers_v > 0.1:
            notes.append("변수 간 중간 정도의 연관성이 있습니다.")
        else:
            notes.append("변수 간 연관성이 약합니다.")
            
        return notes
    
    def generate_comparison_dashboard_data(self) -> Dict[str, Any]:
        """비교 대시보드용 종합 데이터 생성"""
        logger.info("Generating comprehensive comparison dashboard data...")
        
        try:
            self.connect_database()
            
            # 데이터 가용성 확인
            availability = self.check_data_availability()
            
            if not availability['original_available']:
                return {
                    'status': 'error',
                    'message': '원본 데이터를 찾을 수 없습니다.'
                }
            
            # 원본 데이터 추출
            original_demo = self.extract_demographic_data('original')
            original_clinical = self.extract_clinical_data('original')  
            original_temporal = self.extract_temporal_data('original')
            
            dashboard_data = {
                'metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'analysis_type': 'comprehensive_comparison',
                    'data_availability': availability
                },
                'original_data': {
                    'demographics': original_demo,
                    'clinical': original_clinical,
                    'temporal': original_temporal
                }
            }
            
            # 합성 데이터가 있는 경우 비교 분석 수행
            if availability['synthetic_available']:
                logger.info("Synthetic data available, performing comparison analysis...")
                
                synthetic_demo = self.extract_demographic_data('synthetic')
                synthetic_clinical = self.extract_clinical_data('synthetic')
                synthetic_temporal = self.extract_temporal_data('synthetic')
                
                dashboard_data['synthetic_data'] = {
                    'demographics': synthetic_demo,
                    'clinical': synthetic_clinical,
                    'temporal': synthetic_temporal
                }
                
                # 실제 통계적 비교 수행
                comparisons = {}
                
                # 인구학적 비교
                if original_demo.get('age_distribution') and synthetic_demo.get('age_distribution'):
                    comparisons['age_comparison'] = self.perform_statistical_comparison(
                        original_demo['age_distribution'],
                        synthetic_demo['age_distribution'],
                        'age_group', 'count', 'Age Distribution Comparison'
                    )
                
                if original_demo.get('sex_distribution') and synthetic_demo.get('sex_distribution'):
                    comparisons['sex_comparison'] = self.perform_statistical_comparison(
                        original_demo['sex_distribution'],
                        synthetic_demo['sex_distribution'],
                        'sex', 'count', 'Sex Distribution Comparison'
                    )
                
                # KTAS 분포 비교  
                if original_clinical.get('ktas_distribution') and synthetic_clinical.get('ktas_distribution'):
                    comparisons['ktas_comparison'] = self.perform_statistical_comparison(
                        original_clinical['ktas_distribution'],
                        synthetic_clinical['ktas_distribution'],
                        'ktas_no', 'count', 'KTAS Distribution Comparison'
                    )
                
                # 시간적 패턴 비교
                if original_temporal.get('monthly_pattern') and synthetic_temporal.get('monthly_pattern'):
                    comparisons['monthly_comparison'] = self.perform_statistical_comparison(
                        original_temporal['monthly_pattern'],
                        synthetic_temporal['monthly_pattern'],
                        'month', 'count', 'Monthly Pattern Comparison'
                    )
                
                dashboard_data['statistical_comparisons'] = comparisons
                dashboard_data['overall_quality_assessment'] = self._generate_overall_assessment(comparisons)
                
            else:
                logger.warning("No synthetic data available for comparison")
                dashboard_data['message'] = "합성 데이터가 없어 비교 분석을 수행할 수 없습니다. Phase 1-7을 먼저 실행하세요."
            
            return dashboard_data
            
        except Exception as e:
            logger.error(f"Failed to generate comparison dashboard data: {e}")
            return {
                'status': 'error',
                'error': str(e)
            }
        finally:
            self.close_connection()
    
    def _generate_overall_assessment(self, comparisons: Dict) -> Dict[str, Any]:
        """전체적인 품질 평가 생성"""
        if not comparisons:
            return {'status': 'no_data'}
        
        scores = []
        critical_issues = []
        recommendations = []
        
        for comp_name, comp_data in comparisons.items():
            if comp_data.get('status') == 'completed':
                quality_score = comp_data['summary'].get('quality_score', 0)
                scores.append(quality_score)
                
                if quality_score < 70:
                    critical_issues.append(f"{comp_data['test_name']}: 품질 점수 {quality_score}")
                
                if not comp_data['interpretation'].get('clinically_acceptable', True):
                    recommendations.append(f"{comp_data['test_name']}: 편차가 큰 카테고리의 합성 모델 조정 필요")
        
        overall_score = np.mean(scores) if scores else 0
        
        return {
            'overall_quality_score': round(overall_score, 2),
            'grade': 'Excellent' if overall_score >= 90 else 'Good' if overall_score >= 80 else 'Fair' if overall_score >= 70 else 'Poor',
            'critical_issues': critical_issues,
            'recommendations': recommendations,
            'total_comparisons': len([c for c in comparisons.values() if c.get('status') == 'completed'])
        }


def main():
    """메인 실행 함수"""
    logging.basicConfig(level=logging.INFO)
    
    engine = ComparisonDashboardEngine()
    results = engine.generate_comparison_dashboard_data()
    
    # 결과 저장
    output_path = Path("outputs/comparison_dashboard_data.json") 
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Comparison dashboard data generated: {output_path}")
    print(f"Status: {results.get('status', 'completed')}")
    
    if 'data_availability' in results.get('metadata', {}):
        avail = results['metadata']['data_availability']
        print(f"Original data: {avail['original_count']:,} records")
        print(f"Synthetic data: {avail['synthetic_count']:,} records") 
        print(f"Can perform comparison: {avail['can_perform_comparison']}")
    
    return results


if __name__ == "__main__":
    main()