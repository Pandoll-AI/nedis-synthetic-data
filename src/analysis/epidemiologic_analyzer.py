"""
역학 분석 모듈

NEDIS 원본 데이터와 합성 데이터 간의 역학적 패턴을 비교 분석합니다.
의료 역학 지표, 질병 패턴, 위험 요인 등을 분석하여 합성 데이터의 품질을 검증합니다.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
import json

logger = logging.getLogger(__name__)


class EpidemiologicAnalyzer:
    """역학 분석 클래스"""
    
    def __init__(self, sample_db_path: str, synthetic_db_path: str):
        """
        역학 분석기 초기화
        
        Args:
            sample_db_path: 원본 NEDIS 데이터베이스 경로
            synthetic_db_path: 합성 데이터베이스 경로
        """
        self.sample_db_path = sample_db_path
        self.synthetic_db_path = synthetic_db_path
        self.sample_conn = None
        self.synthetic_conn = None
        
        # ICD-10 질병군 매핑 (주요 질병군)
        self.disease_groups = {
            'A00-B99': '감염성 및 기생충 질환',
            'C00-D48': '신생물',
            'D50-D89': '혈액 및 혈액기관 질환',
            'E00-E89': '내분비, 영양 및 대사질환',
            'F00-F99': '정신 및 행동장애',
            'G00-G99': '신경계 질환',
            'H00-H59': '눈 및 부속기관 질환',
            'H60-H95': '귀 및 유돌기관 질환',
            'I00-I99': '순환기계 질환',
            'J00-J99': '호흡기계 질환',
            'K00-K95': '소화기계 질환',
            'L00-L99': '피부 및 피하조직 질환',
            'M00-M99': '근골격계 및 결합조직 질환',
            'N00-N99': '비뇨생식기계 질환',
            'O00-O99': '임신, 출산 및 산욕',
            'P00-P96': '주산기 기원 질환',
            'Q00-Q99': '선천성 기형 및 염색체 이상',
            'R00-R99': '증상, 징후 및 검사소견',
            'S00-T98': '손상, 중독 및 외적요인',
            'V01-Y98': '질병이환 및 사망의 외적요인',
            'Z00-Z99': '건강상태 및 보건서비스 이용'
        }
        
        # KTAS 중증도 분류
        self.ktas_severity = {
            '1': '소생술',
            '2': '응급',
            '3': '긴급',
            '4': '준긴급', 
            '5': '비긴급'
        }
        
    def connect_databases(self):
        """데이터베이스 연결"""
        try:
            self.sample_conn = duckdb.connect(self.sample_db_path, read_only=True)
            self.synthetic_conn = duckdb.connect(self.synthetic_db_path, read_only=True)
            logger.info("Database connections established")
        except Exception as e:
            logger.error(f"Failed to connect to databases: {e}")
            raise
            
    def close_connections(self):
        """데이터베이스 연결 종료"""
        if self.sample_conn:
            self.sample_conn.close()
        if self.synthetic_conn:
            self.synthetic_conn.close()
            
    def _categorize_icd_code(self, icd_code: str) -> str:
        """ICD-10 코드를 질병군으로 분류"""
        if not icd_code or len(icd_code) < 3:
            return '미분류'
            
        first_char = icd_code[0].upper()
        second_char = icd_code[1:3]
        
        try:
            if first_char == 'A' or first_char == 'B':
                return '감염성 및 기생충 질환'
            elif first_char == 'C' or (first_char == 'D' and int(second_char) <= 48):
                return '신생물'
            elif first_char == 'D' and int(second_char) >= 50:
                return '혈액 및 혈액기관 질환'
            elif first_char == 'E':
                return '내분비, 영양 및 대사질환'
            elif first_char == 'F':
                return '정신 및 행동장애'
            elif first_char == 'G':
                return '신경계 질환'
            elif first_char == 'H' and int(second_char) <= 59:
                return '눈 및 부속기관 질환'
            elif first_char == 'H' and int(second_char) >= 60:
                return '귀 및 유돌기관 질환'
            elif first_char == 'I':
                return '순환기계 질환'
            elif first_char == 'J':
                return '호흡기계 질환'
            elif first_char == 'K':
                return '소화기계 질환'
            elif first_char == 'L':
                return '피부 및 피하조직 질환'
            elif first_char == 'M':
                return '근골격계 및 결합조직 질환'
            elif first_char == 'N':
                return '비뇨생식기계 질환'
            elif first_char == 'O':
                return '임신, 출산 및 산욕'
            elif first_char == 'P':
                return '주산기 기원 질환'
            elif first_char == 'Q':
                return '선천성 기형 및 염색체 이상'
            elif first_char == 'R':
                return '증상, 징후 및 검사소견'
            elif first_char in ['S', 'T']:
                return '손상, 중독 및 외적요인'
            elif first_char in ['V', 'W', 'X', 'Y']:
                return '질병이환 및 사망의 외적요인'
            elif first_char == 'Z':
                return '건강상태 및 보건서비스 이용'
            else:
                return '미분류'
        except:
            return '미분류'
    
    def analyze_demographic_patterns(self) -> Dict[str, Any]:
        """인구학적 패턴 분석"""
        logger.info("Analyzing demographic patterns...")
        
        results = {}
        
        # 연령별 분포 분석
        age_query = """
        SELECT 
            CASE 
                WHEN pat_age_gr = '01' THEN '0-9세'
                WHEN pat_age_gr = '09' THEN '10-19세'
                WHEN pat_age_gr = '10' THEN '20-29세'
                WHEN pat_age_gr = '20' THEN '30-39세'
                WHEN pat_age_gr = '30' THEN '40-49세'
                WHEN pat_age_gr = '40' THEN '50-59세'
                WHEN pat_age_gr = '50' THEN '60-69세'
                WHEN pat_age_gr = '60' THEN '70-79세'
                WHEN pat_age_gr = '70' THEN '80-89세'
                WHEN pat_age_gr = '80' THEN '90세 이상'
                ELSE '미분류'
            END as age_group,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM nedis2017 
        WHERE pat_age_gr IS NOT NULL
        GROUP BY pat_age_gr
        ORDER BY pat_age_gr
        """
        
        sample_age = self.sample_conn.execute(age_query).fetchdf()
        results['age_distribution'] = {
            'sample': sample_age.to_dict('records')
        }
        
        # 성별 분포 분석
        sex_query = """
        SELECT 
            CASE 
                WHEN pat_sex_cd = 'M' THEN '남성'
                WHEN pat_sex_cd = 'F' THEN '여성'
                ELSE '미분류'
            END as sex,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM nedis2017 
        WHERE pat_sex_cd IS NOT NULL
        GROUP BY pat_sex_cd
        """
        
        sample_sex = self.sample_conn.execute(sex_query).fetchdf()
        results['sex_distribution'] = {
            'sample': sample_sex.to_dict('records')
        }
        
        # 지역별 분포 분석
        region_query = """
        SELECT 
            pat_do_cd as region_code,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM nedis2017 
        WHERE pat_do_cd IS NOT NULL
        GROUP BY pat_do_cd
        ORDER BY count DESC
        LIMIT 20
        """
        
        sample_region = self.sample_conn.execute(region_query).fetchdf()
        results['region_distribution'] = {
            'sample': sample_region.to_dict('records')
        }
        
        return results
    
    def analyze_disease_epidemiology(self) -> Dict[str, Any]:
        """질병 역학 분석"""
        logger.info("Analyzing disease epidemiology...")
        
        results = {}
        
        # KTAS 중증도 분포
        ktas_query = """
        SELECT 
            ktas_no,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM nedis2017 
        WHERE ktas_no IS NOT NULL AND ktas_no != ''
        GROUP BY ktas_no
        ORDER BY ktas_no
        """
        
        sample_ktas = self.sample_conn.execute(ktas_query).fetchdf()
        results['ktas_distribution'] = {
            'sample': sample_ktas.to_dict('records')
        }
        
        # 연령별 중증도 패턴
        age_ktas_query = """
        SELECT 
            CASE 
                WHEN pat_age_gr = '01' THEN '0-9세'
                WHEN pat_age_gr = '09' THEN '10-19세'
                WHEN pat_age_gr = '10' THEN '20-29세'
                WHEN pat_age_gr = '20' THEN '30-39세'
                WHEN pat_age_gr = '30' THEN '40-49세'
                WHEN pat_age_gr = '40' THEN '50-59세'
                WHEN pat_age_gr = '50' THEN '60-69세'
                WHEN pat_age_gr = '60' THEN '70-79세'
                WHEN pat_age_gr = '70' THEN '80-89세'
                WHEN pat_age_gr = '80' THEN '90세 이상'
                ELSE '미분류'
            END as age_group,
            ktas_no,
            COUNT(*) as count
        FROM nedis2017 
        WHERE pat_age_gr IS NOT NULL AND ktas_no IS NOT NULL AND ktas_no != ''
        GROUP BY pat_age_gr, ktas_no
        ORDER BY pat_age_gr, ktas_no
        """
        
        sample_age_ktas = self.sample_conn.execute(age_ktas_query).fetchdf()
        results['age_ktas_pattern'] = {
            'sample': sample_age_ktas.to_dict('records')
        }
        
        # 성별 중증도 패턴
        sex_ktas_query = """
        SELECT 
            CASE 
                WHEN pat_sex_cd = 'M' THEN '남성'
                WHEN pat_sex_cd = 'F' THEN '여성'
                ELSE '미분류'
            END as sex,
            ktas_no,
            COUNT(*) as count
        FROM nedis2017 
        WHERE pat_sex_cd IS NOT NULL AND ktas_no IS NOT NULL AND ktas_no != ''
        GROUP BY pat_sex_cd, ktas_no
        ORDER BY pat_sex_cd, ktas_no
        """
        
        sample_sex_ktas = self.sample_conn.execute(sex_ktas_query).fetchdf()
        results['sex_ktas_pattern'] = {
            'sample': sample_sex_ktas.to_dict('records')
        }
        
        return results
    
    def analyze_temporal_epidemiology(self) -> Dict[str, Any]:
        """시간 역학 분석"""
        logger.info("Analyzing temporal epidemiology...")
        
        results = {}
        
        # 월별 방문 패턴
        monthly_query = """
        SELECT 
            CAST(SUBSTR(vst_dt, 5, 2) AS INTEGER) as month,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM nedis2017 
        WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) >= 6
        GROUP BY SUBSTR(vst_dt, 5, 2)
        ORDER BY month
        """
        
        sample_monthly = self.sample_conn.execute(monthly_query).fetchdf()
        results['monthly_pattern'] = {
            'sample': sample_monthly.to_dict('records')
        }
        
        # 요일별 방문 패턴 (2017년 기준)
        weekday_query = """
        WITH date_parse AS (
            SELECT 
                vst_dt,
                TRY_CAST(vst_dt AS DATE) as visit_date,
                COUNT(*) as count
            FROM nedis2017 
            WHERE vst_dt IS NOT NULL 
                AND LENGTH(vst_dt) = 8
                AND vst_dt LIKE '2017%'
            GROUP BY vst_dt
        )
        SELECT 
            DAYOFWEEK(visit_date) as dow,
            CASE DAYOFWEEK(visit_date)
                WHEN 1 THEN '일요일'
                WHEN 2 THEN '월요일' 
                WHEN 3 THEN '화요일'
                WHEN 4 THEN '수요일'
                WHEN 5 THEN '목요일'
                WHEN 6 THEN '금요일'
                WHEN 7 THEN '토요일'
            END as weekday,
            SUM(count) as total_count,
            ROUND(SUM(count) * 100.0 / SUM(SUM(count)) OVER(), 2) as percentage
        FROM date_parse
        WHERE visit_date IS NOT NULL
        GROUP BY DAYOFWEEK(visit_date)
        ORDER BY dow
        """
        
        sample_weekday = self.sample_conn.execute(weekday_query).fetchdf()
        results['weekday_pattern'] = {
            'sample': sample_weekday.to_dict('records')
        }
        
        # 시간별 방문 패턴
        hourly_query = """
        SELECT 
            CAST(SUBSTR(vst_tm, 1, 2) AS INTEGER) as hour,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
        FROM nedis2017 
        WHERE vst_tm IS NOT NULL 
            AND LENGTH(vst_tm) >= 2
            AND CAST(SUBSTR(vst_tm, 1, 2) AS INTEGER) BETWEEN 0 AND 23
        GROUP BY SUBSTR(vst_tm, 1, 2)
        ORDER BY hour
        """
        
        sample_hourly = self.sample_conn.execute(hourly_query).fetchdf()
        results['hourly_pattern'] = {
            'sample': sample_hourly.to_dict('records')
        }
        
        return results
    
    def analyze_clinical_epidemiology(self) -> Dict[str, Any]:
        """임상 역학 분석"""
        logger.info("Analyzing clinical epidemiology...")
        
        results = {}
        
        # 생체징후 분석
        vitals_query = """
        SELECT 
            COUNT(*) as total_records,
            COUNT(vst_sbp) as sbp_records,
            COUNT(vst_dbp) as dbp_records,
            COUNT(vst_per_pu) as pulse_records,
            COUNT(vst_per_br) as respiration_records,
            COUNT(vst_oxy) as oxygen_records,
            AVG(CASE WHEN vst_sbp > 0 AND vst_sbp < 300 THEN vst_sbp END) as avg_sbp,
            AVG(CASE WHEN vst_dbp > 0 AND vst_dbp < 200 THEN vst_dbp END) as avg_dbp,
            AVG(CASE WHEN vst_per_pu > 0 AND vst_per_pu < 200 THEN vst_per_pu END) as avg_pulse,
            AVG(CASE WHEN vst_per_br > 0 AND vst_per_br < 60 THEN vst_per_br END) as avg_respiration,
            AVG(CASE WHEN vst_oxy > 0 AND vst_oxy <= 100 THEN vst_oxy END) as avg_oxygen
        FROM nedis2017
        """
        
        sample_vitals = self.sample_conn.execute(vitals_query).fetchdf()
        results['vital_signs'] = {
            'sample': sample_vitals.to_dict('records')[0]
        }
        
        # 중증도별 생체징후 패턴
        ktas_vitals_query = """
        SELECT 
            ktas_no,
            COUNT(*) as count,
            AVG(CASE WHEN vst_sbp > 0 AND vst_sbp < 300 THEN vst_sbp END) as avg_sbp,
            AVG(CASE WHEN vst_dbp > 0 AND vst_dbp < 200 THEN vst_dbp END) as avg_dbp,
            AVG(CASE WHEN vst_per_pu > 0 AND vst_per_pu < 200 THEN vst_per_pu END) as avg_pulse,
            AVG(CASE WHEN vst_oxy > 0 AND vst_oxy <= 100 THEN vst_oxy END) as avg_oxygen
        FROM nedis2017
        WHERE ktas_no IS NOT NULL AND ktas_no != ''
        GROUP BY ktas_no
        ORDER BY ktas_no
        """
        
        sample_ktas_vitals = self.sample_conn.execute(ktas_vitals_query).fetchdf()
        results['ktas_vital_patterns'] = {
            'sample': sample_ktas_vitals.to_dict('records')
        }
        
        return results
    
    def analyze_spatial_epidemiology(self) -> Dict[str, Any]:
        """공간 역학 분석"""
        logger.info("Analyzing spatial epidemiology...")
        
        results = {}
        
        # 지역별 중증도 패턴
        region_ktas_query = """
        SELECT 
            pat_do_cd as region_code,
            ktas_no,
            COUNT(*) as count
        FROM nedis2017
        WHERE pat_do_cd IS NOT NULL AND ktas_no IS NOT NULL AND ktas_no != ''
        GROUP BY pat_do_cd, ktas_no
        ORDER BY pat_do_cd, ktas_no
        """
        
        sample_region_ktas = self.sample_conn.execute(region_ktas_query).fetchdf()
        results['region_ktas_pattern'] = {
            'sample': sample_region_ktas.to_dict('records')
        }
        
        # 병원별 환자 분포
        hospital_query = """
        SELECT 
            emorg_cd as hospital_code,
            COUNT(*) as patient_count,
            COUNT(DISTINCT pat_do_cd) as region_count,
            ROUND(AVG(CASE WHEN pat_age IS NOT NULL AND pat_age != '' AND pat_age != '-'
                          AND TRY_CAST(pat_age AS INTEGER) IS NOT NULL
                          THEN CAST(pat_age AS INTEGER) END), 1) as avg_age
        FROM nedis2017
        WHERE emorg_cd IS NOT NULL
        GROUP BY emorg_cd
        ORDER BY patient_count DESC
        LIMIT 20
        """
        
        sample_hospital = self.sample_conn.execute(hospital_query).fetchdf()
        results['hospital_distribution'] = {
            'sample': sample_hospital.to_dict('records')
        }
        
        return results
    
    def perform_statistical_tests(self, original_data: Dict, synthetic_data: Dict) -> Dict[str, Any]:
        """통계적 검정 수행"""
        logger.info("Performing statistical tests...")
        
        test_results = {}
        
        # 카이제곱 검정을 위한 헬퍼 함수
        def chi_square_test(obs1, obs2, test_name):
            try:
                # 두 분포를 동일한 카테고리로 정렬
                combined_data = pd.concat([obs1, obs2], ignore_index=True)
                categories = combined_data.iloc[:, 0].unique()
                
                obs1_counts = []
                obs2_counts = []
                
                for cat in categories:
                    count1 = obs1[obs1.iloc[:, 0] == cat]['count'].sum() if len(obs1[obs1.iloc[:, 0] == cat]) > 0 else 0
                    count2 = obs2[obs2.iloc[:, 0] == cat]['count'].sum() if len(obs2[obs2.iloc[:, 0] == cat]) > 0 else 0
                    obs1_counts.append(count1)
                    obs2_counts.append(count2)
                
                # 카이제곱 검정
                contingency_table = [obs1_counts, obs2_counts]
                chi2, p_value, dof, expected = chi2_contingency(contingency_table)
                
                return {
                    'test_name': test_name,
                    'chi2_statistic': float(chi2),
                    'p_value': float(p_value),
                    'degrees_of_freedom': int(dof),
                    'significant': p_value < 0.05
                }
            except Exception as e:
                logger.warning(f"Chi-square test failed for {test_name}: {e}")
                return {
                    'test_name': test_name,
                    'error': str(e)
                }
        
        # 연령 분포 검정 (원본 데이터만 있는 경우 스킵)
        if 'age_distribution' in original_data and 'synthetic' in original_data['age_distribution']:
            orig_age = pd.DataFrame(original_data['age_distribution']['sample'])
            synt_age = pd.DataFrame(original_data['age_distribution']['synthetic'])
            test_results['age_distribution'] = chi_square_test(orig_age, synt_age, 'Age Distribution')
        
        return test_results
    
    def generate_epidemiologic_summary(self) -> Dict[str, Any]:
        """종합적인 역학 분석 결과 생성"""
        logger.info("Generating comprehensive epidemiologic analysis...")
        
        try:
            self.connect_databases()
            
            summary = {
                'metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'sample_db': self.sample_db_path,
                    'synthetic_db': self.synthetic_db_path
                },
                'demographics': self.analyze_demographic_patterns(),
                'disease_epidemiology': self.analyze_disease_epidemiology(),
                'temporal_epidemiology': self.analyze_temporal_epidemiology(),
                'clinical_epidemiology': self.analyze_clinical_epidemiology(),
                'spatial_epidemiology': self.analyze_spatial_epidemiology()
            }
            
            # 통계적 검정 (합성 데이터가 있는 경우)
            summary['statistical_tests'] = self.perform_statistical_tests(
                summary['demographics'], {}
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in epidemiologic analysis: {e}")
            raise
        finally:
            self.close_connections()


def main():
    """메인 실행 함수"""
    logging.basicConfig(level=logging.INFO)
    
    # 분석기 초기화
    analyzer = EpidemiologicAnalyzer(
        sample_db_path="nedis_sample.duckdb",
        synthetic_db_path="nedis_synthetic.duckdb"
    )
    
    # 분석 실행
    results = analyzer.generate_epidemiologic_summary()
    
    # 결과 저장
    output_path = Path("outputs/epidemiologic_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"Epidemiologic analysis completed. Results saved to: {output_path}")
    return results


if __name__ == "__main__":
    main()