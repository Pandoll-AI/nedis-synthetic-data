"""
원본 vs 합성 데이터 비교 분석 모듈

NEDIS 원본 데이터와 합성 데이터 간의 통계적, 역학적 패턴을 직접 비교 분석합니다.
Mock data를 사용하지 않고 실제 DuckDB 데이터를 기반으로 비교 분석을 수행합니다.
"""

import duckdb
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu
import json

logger = logging.getLogger(__name__)

class ComparativeAnalyzer:
    """원본 vs 합성 데이터 비교 분석 클래스"""
    
    def __init__(self, db_path: str = "nedis_sample.duckdb"):
        """
        비교 분석기 초기화
        
        Args:
            db_path: DuckDB 데이터베이스 경로
        """
        self.db_path = db_path
        self.conn = None
        
    def connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = duckdb.connect(self.db_path)
            logger.info(f"DuckDB 연결 성공: {self.db_path}")
            
            # 스키마 확인
            schemas = self.conn.execute("SELECT schema_name FROM information_schema.schemata").fetchall()
            logger.info(f"사용 가능한 스키마: {[s[0] for s in schemas]}")
            
        except Exception as e:
            logger.error(f"데이터베이스 연결 실패: {e}")
            raise
    
    def disconnect(self):
        """데이터베이스 연결 해제"""
        if self.conn:
            self.conn.close()
            logger.info("DuckDB 연결 해제")
    
    def get_original_demographics(self) -> Dict[str, Any]:
        """원본 데이터의 인구통계학적 분포 가져오기"""
        try:
            # 연령 분포
            age_query = """
            SELECT 
                CASE 
                    WHEN pat_age_gr IS NULL OR pat_age_gr = '' THEN '미분류'
                    ELSE pat_age_gr 
                END as age_group,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017 
            GROUP BY pat_age_gr
            ORDER BY count DESC
            """
            age_df = self.conn.execute(age_query).fetchdf()
            
            # 성별 분포
            sex_query = """
            SELECT 
                CASE 
                    WHEN pat_sex_cd IS NULL OR pat_sex_cd = '' THEN '미분류'
                    WHEN pat_sex_cd = 'M' THEN '남성'
                    WHEN pat_sex_cd = 'F' THEN '여성'
                    ELSE '미분류'
                END as sex,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017 
            GROUP BY pat_sex_cd
            ORDER BY count DESC
            """
            sex_df = self.conn.execute(sex_query).fetchdf()
            
            # 지역 분포
            region_query = """
            SELECT 
                CASE 
                    WHEN pat_do_cd IS NULL OR pat_do_cd = '' THEN '미분류'
                    ELSE pat_do_cd 
                END as region_code,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017 
            GROUP BY pat_do_cd
            ORDER BY count DESC
            LIMIT 20
            """
            region_df = self.conn.execute(region_query).fetchdf()
            
            return {
                "age_distribution": age_df.to_dict('records'),
                "sex_distribution": sex_df.to_dict('records'),
                "region_distribution": region_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"원본 인구통계학적 데이터 조회 실패: {e}")
            return {"age_distribution": [], "sex_distribution": [], "region_distribution": []}
    
    def get_synthetic_demographics(self) -> Dict[str, Any]:
        """합성 데이터의 인구통계학적 분포 가져오기"""
        try:
            # 연령 분포
            age_query = """
            SELECT 
                CASE 
                    WHEN pat_age_gr IS NULL OR pat_age_gr = '' THEN '미분류'
                    ELSE pat_age_gr 
                END as age_group,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_synthetic.clinical_records 
            GROUP BY pat_age_gr
            ORDER BY count DESC
            """
            age_df = self.conn.execute(age_query).fetchdf()
            
            # 성별 분포
            sex_query = """
            SELECT 
                CASE 
                    WHEN pat_sex IS NULL OR pat_sex = '' THEN '미분류'
                    WHEN pat_sex = 'M' THEN '남성'
                    WHEN pat_sex = 'F' THEN '여성'
                    ELSE '미분류'
                END as sex,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_synthetic.clinical_records 
            GROUP BY pat_sex
            ORDER BY count DESC
            """
            sex_df = self.conn.execute(sex_query).fetchdf()
            
            # 지역 분포
            region_query = """
            SELECT 
                CASE 
                    WHEN pat_do_cd IS NULL OR pat_do_cd = '' THEN '미분류'
                    ELSE pat_do_cd 
                END as region_code,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_synthetic.clinical_records 
            GROUP BY pat_do_cd
            ORDER BY count DESC
            LIMIT 20
            """
            region_df = self.conn.execute(region_query).fetchdf()
            
            return {
                "age_distribution": age_df.to_dict('records'),
                "sex_distribution": sex_df.to_dict('records'),
                "region_distribution": region_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"합성 인구통계학적 데이터 조회 실패: {e}")
            return {"age_distribution": [], "sex_distribution": [], "region_distribution": []}
    
    def get_original_clinical_patterns(self) -> Dict[str, Any]:
        """원본 데이터의 임상 패턴 가져오기"""
        try:
            # KTAS 분포 - 올바른 ktas_fstu 컬럼 사용
            ktas_query = """
            SELECT 
                CASE 
                    WHEN ktas_fstu IS NULL OR ktas_fstu = '' OR ktas_fstu = '-' THEN '미분류'
                    WHEN ktas_fstu = '1' THEN '1단계 (소생)'
                    WHEN ktas_fstu = '2' THEN '2단계 (응급)'
                    WHEN ktas_fstu = '3' THEN '3단계 (긴급)'
                    WHEN ktas_fstu = '4' THEN '4단계 (준긴급)'
                    WHEN ktas_fstu = '5' THEN '5단계 (비긴급)'
                    ELSE '기타'
                END as ktas_level,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017 
            GROUP BY ktas_fstu
            ORDER BY count DESC
            """
            ktas_df = self.conn.execute(ktas_query).fetchdf()
            
            return {
                "ktas_distribution": ktas_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"원본 임상 패턴 데이터 조회 실패: {e}")
            return {"ktas_distribution": []}
    
    def get_synthetic_clinical_patterns(self) -> Dict[str, Any]:
        """합성 데이터의 임상 패턴 가져오기"""
        try:
            # KTAS 분포
            ktas_query = """
            SELECT 
                CASE 
                    WHEN ktas_fstu IS NULL OR ktas_fstu = '' OR ktas_fstu = '-' THEN '미분류'
                    WHEN ktas_fstu = '1' THEN '1단계 (소생)'
                    WHEN ktas_fstu = '2' THEN '2단계 (응급)'
                    WHEN ktas_fstu = '3' THEN '3단계 (긴급)'
                    WHEN ktas_fstu = '4' THEN '4단계 (준긴급)'
                    WHEN ktas_fstu = '5' THEN '5단계 (비긴급)'
                    ELSE '기타'
                END as ktas_level,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_synthetic.clinical_records 
            GROUP BY ktas_fstu
            ORDER BY count DESC
            """
            ktas_df = self.conn.execute(ktas_query).fetchdf()
            
            return {
                "ktas_distribution": ktas_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"합성 임상 패턴 데이터 조회 실패: {e}")
            return {"ktas_distribution": []}
    
    def get_original_temporal_patterns(self) -> Dict[str, Any]:
        """원본 데이터의 시간 패턴 가져오기"""
        try:
            # 월별 분포
            monthly_query = """
            SELECT 
                CAST(SUBSTRING(vst_dt, 5, 2) AS INTEGER) as month,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017 
            WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) = 8
            GROUP BY CAST(SUBSTRING(vst_dt, 5, 2) AS INTEGER)
            ORDER BY month
            """
            monthly_df = self.conn.execute(monthly_query).fetchdf()
            
            # 요일별 분포  
            weekday_query = """
            SELECT 
                DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d')) as weekday,
                CASE DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d'))
                    WHEN 1 THEN '일요일'
                    WHEN 2 THEN '월요일'
                    WHEN 3 THEN '화요일'
                    WHEN 4 THEN '수요일'
                    WHEN 5 THEN '목요일'
                    WHEN 6 THEN '금요일'
                    WHEN 7 THEN '토요일'
                END as weekday_name,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017 
            WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) = 8
            GROUP BY DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d'))
            ORDER BY weekday
            """
            weekday_df = self.conn.execute(weekday_query).fetchdf()
            
            return {
                "monthly_distribution": monthly_df.to_dict('records'),
                "weekday_distribution": weekday_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"원본 시간 패턴 데이터 조회 실패: {e}")
            return {"monthly_distribution": [], "weekday_distribution": []}
    
    def get_synthetic_temporal_patterns(self) -> Dict[str, Any]:
        """합성 데이터의 시간 패턴 가져오기"""
        try:
            # 월별 분포
            monthly_query = """
            SELECT 
                CAST(SUBSTRING(vst_dt, 5, 2) AS INTEGER) as month,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_synthetic.clinical_records 
            WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) = 8
            GROUP BY CAST(SUBSTRING(vst_dt, 5, 2) AS INTEGER)
            ORDER BY month
            """
            monthly_df = self.conn.execute(monthly_query).fetchdf()
            
            # 요일별 분포
            weekday_query = """
            SELECT 
                DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d')) as weekday,
                CASE DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d'))
                    WHEN 1 THEN '일요일'
                    WHEN 2 THEN '월요일'
                    WHEN 3 THEN '화요일'
                    WHEN 4 THEN '수요일'
                    WHEN 5 THEN '목요일'
                    WHEN 6 THEN '금요일'
                    WHEN 7 THEN '토요일'
                END as weekday_name,
                COUNT(*) as count,
                ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_synthetic.clinical_records 
            WHERE vst_dt IS NOT NULL AND LENGTH(vst_dt) = 8
            GROUP BY DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d'))
            ORDER BY weekday
            """
            weekday_df = self.conn.execute(weekday_query).fetchdf()
            
            return {
                "monthly_distribution": monthly_df.to_dict('records'),
                "weekday_distribution": weekday_df.to_dict('records')
            }
            
        except Exception as e:
            logger.error(f"합성 시간 패턴 데이터 조회 실패: {e}")
            return {"monthly_distribution": [], "weekday_distribution": []}
    
    def calculate_statistical_tests(self, original_data: Dict, synthetic_data: Dict) -> Dict[str, Any]:
        """통계적 검정 수행"""
        results = {}
        
        try:
            # 연령 분포 비교
            if original_data.get("age_distribution") and synthetic_data.get("age_distribution"):
                orig_age = pd.DataFrame(original_data["age_distribution"])
                synth_age = pd.DataFrame(synthetic_data["age_distribution"])
                
                if len(orig_age) > 0 and len(synth_age) > 0:
                    # 공통 연령군 추출
                    common_ages = set(orig_age['age_group']) & set(synth_age['age_group'])
                    if common_ages:
                        orig_counts = orig_age[orig_age['age_group'].isin(common_ages)]['count'].values
                        synth_counts = synth_age[synth_age['age_group'].isin(common_ages)]['count'].values
                        
                        if len(orig_counts) > 1 and len(synth_counts) > 1:
                            chi2_stat, p_value = stats.chi2_contingency([orig_counts, synth_counts])[:2]
                            results['age_chi2'] = {
                                'statistic': float(chi2_stat),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05)
                            }
            
            # 성별 분포 비교
            if original_data.get("sex_distribution") and synthetic_data.get("sex_distribution"):
                orig_sex = pd.DataFrame(original_data["sex_distribution"])
                synth_sex = pd.DataFrame(synthetic_data["sex_distribution"])
                
                if len(orig_sex) > 0 and len(synth_sex) > 0:
                    common_sexes = set(orig_sex['sex']) & set(synth_sex['sex'])
                    if common_sexes:
                        orig_counts = orig_sex[orig_sex['sex'].isin(common_sexes)]['count'].values
                        synth_counts = synth_sex[synth_sex['sex'].isin(common_sexes)]['count'].values
                        
                        if len(orig_counts) > 1 and len(synth_counts) > 1:
                            chi2_stat, p_value = stats.chi2_contingency([orig_counts, synth_counts])[:2]
                            results['sex_chi2'] = {
                                'statistic': float(chi2_stat),
                                'p_value': float(p_value),
                                'significant': bool(p_value < 0.05)
                            }
            
        except Exception as e:
            logger.error(f"통계적 검정 실패: {e}")
        
        return results
    
    def run_comparative_analysis(self) -> Dict[str, Any]:
        """전체 비교 분석 실행"""
        logger.info("원본 vs 합성 데이터 비교 분석 시작")
        
        self.connect()
        
        try:
            # 원본 데이터 분석
            logger.info("원본 데이터 분석 중...")
            original_demographics = self.get_original_demographics()
            original_clinical = self.get_original_clinical_patterns()
            original_temporal = self.get_original_temporal_patterns()
            
            # 합성 데이터 분석
            logger.info("합성 데이터 분석 중...")
            synthetic_demographics = self.get_synthetic_demographics()
            synthetic_clinical = self.get_synthetic_clinical_patterns()
            synthetic_temporal = self.get_synthetic_temporal_patterns()
            
            # 통계적 검정
            logger.info("통계적 검정 수행 중...")
            statistical_tests = self.calculate_statistical_tests(
                {**original_demographics, **original_clinical, **original_temporal},
                {**synthetic_demographics, **synthetic_clinical, **synthetic_temporal}
            )
            
            # 데이터 범위 정보 수집
            orig_date_range = self.conn.execute("SELECT MIN(vst_dt) as min_date, MAX(vst_dt) as max_date, COUNT(DISTINCT vst_dt) as unique_dates FROM nedis_original.nedis2017 WHERE vst_dt IS NOT NULL").fetchone()
            synth_date_range = self.conn.execute("SELECT MIN(vst_dt) as min_date, MAX(vst_dt) as max_date, COUNT(DISTINCT vst_dt) as unique_dates FROM nedis_synthetic.clinical_records WHERE vst_dt IS NOT NULL").fetchone()
            
            # 결과 조합
            results = {
                "metadata": {
                    "analysis_date": datetime.now().isoformat(),
                    "database": self.db_path,
                    "original_record_count": self.conn.execute("SELECT COUNT(*) FROM nedis_original.nedis2017").fetchone()[0],
                    "synthetic_record_count": self.conn.execute("SELECT COUNT(*) FROM nedis_synthetic.clinical_records").fetchone()[0],
                    "original_date_range": {
                        "min_date": orig_date_range[0],
                        "max_date": orig_date_range[1],
                        "unique_dates": orig_date_range[2]
                    },
                    "synthetic_date_range": {
                        "min_date": synth_date_range[0],
                        "max_date": synth_date_range[1],
                        "unique_dates": synth_date_range[2]
                    },
                    "date_coverage_warning": synth_date_range[2] == 1 and orig_date_range[2] > 300
                },
                "demographics": {
                    "original": original_demographics,
                    "synthetic": synthetic_demographics
                },
                "clinical_patterns": {
                    "original": original_clinical,
                    "synthetic": synthetic_clinical
                },
                "temporal_patterns": {
                    "original": original_temporal,
                    "synthetic": synthetic_temporal
                },
                "statistical_tests": statistical_tests
            }
            
            logger.info("비교 분석 완료")
            return results
            
        except Exception as e:
            logger.error(f"비교 분석 실패: {e}")
            raise
        finally:
            self.disconnect()

def main():
    """메인 함수 - 비교 분석 실행"""
    analyzer = ComparativeAnalyzer()
    results = analyzer.run_comparative_analysis()
    
    # 결과 저장
    output_path = Path("outputs/comparative_analysis.json")
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"비교 분석 결과 저장: {output_path}")
    
    # 요약 출력
    metadata = results.get("metadata", {})
    print(f"\n=== NEDIS 원본 vs 합성 데이터 비교 분석 결과 ===")
    print(f"분석 일시: {metadata.get('analysis_date', 'N/A')}")
    print(f"원본 레코드 수: {metadata.get('original_record_count', 'N/A'):,}")
    print(f"합성 레코드 수: {metadata.get('synthetic_record_count', 'N/A'):,}")
    
    # 통계적 검정 결과 요약
    tests = results.get("statistical_tests", {})
    if tests:
        print(f"\n=== 통계적 검정 결과 ===")
        for test_name, result in tests.items():
            print(f"{test_name}: p-value={result.get('p_value', 'N/A'):.6f}, 유의미={'Yes' if result.get('significant', False) else 'No'}")

if __name__ == "__main__":
    main()