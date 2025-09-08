"""
NEDIS 데이터 프로파일링 모듈

원본 NEDIS 데이터에서 인구학적 통계와 병원 용량 정보를 추출하여
합성 데이터 생성의 기초 메타데이터를 생성합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging
from datetime import datetime
import warnings

# 상대 import 대신 절대 import 사용 
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import DatabaseManager
from core.config import ConfigManager


class NEDISProfiler:
    """NEDIS 데이터 프로파일러"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        NEDIS 데이터 프로파일러 초기화
        
        Args:
            db_manager: 데이터베이스 관리자 인스턴스
            config: 설정 관리자 인스턴스
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def extract_population_margins(self) -> bool:
        """
        인구학적 마진 추출 및 저장
        
        원본 데이터에서 (시도, 연령군, 성별) 조합별 방문 통계와
        계절별, 요일별 가중치를 계산하여 population_margins 테이블에 저장
        
        Returns:
            성공 여부
        """
        self.logger.info("Starting population margins extraction")
        
        # 원본 데이터 테이블 존재 확인
        if not self._check_original_data():
            return False
            
        # 기존 데이터 삭제
        self.db.execute_query("DELETE FROM nedis_meta.population_margins")
        
        query = """
        INSERT INTO nedis_meta.population_margins
        SELECT 
            pat_do_cd,
            pat_age_gr,
            pat_sex,
            COUNT(*) as yearly_visits,
            -- 계절별 가중치 계산 (YYYYMMDD 형식을 DATE로 변환)
            SUM(CASE WHEN MONTH(STRPTIME(vst_dt, '%Y%m%d')) IN (3,4,5) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_spring,
            SUM(CASE WHEN MONTH(STRPTIME(vst_dt, '%Y%m%d')) IN (6,7,8) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_summer,
            SUM(CASE WHEN MONTH(STRPTIME(vst_dt, '%Y%m%d')) IN (9,10,11) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_fall,
            SUM(CASE WHEN MONTH(STRPTIME(vst_dt, '%Y%m%d')) IN (12,1,2) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as seasonal_weight_winter,
            -- 요일별 가중치 계산  
            SUM(CASE WHEN DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d')) IN (1,7) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*) as weekend_weight,
            (1 - SUM(CASE WHEN DAYOFWEEK(STRPTIME(vst_dt, '%Y%m%d')) IN (1,7) THEN 1 ELSE 0 END)::DOUBLE / COUNT(*)) as weekday_weight
        FROM nedis_original.nedis2017
        WHERE pat_do_cd != '' AND pat_do_cd IS NOT NULL
          AND pat_age_gr != '' AND pat_age_gr IS NOT NULL
          AND pat_sex IN ('M', 'F')
          AND vst_dt IS NOT NULL AND vst_dt != ''
        GROUP BY pat_do_cd, pat_age_gr, pat_sex
        HAVING COUNT(*) >= 10  -- 최소 10개 레코드가 있는 조합만 포함
        """
        
        try:
            self.db.execute_query(query)
            
            # 결과 검증
            count_query = "SELECT COUNT(*) as total FROM nedis_meta.population_margins"
            total_combinations = self.db.fetch_dataframe(count_query)['total'][0]
            
            # 품질 검증
            quality_issues = self._validate_population_margins()
            
            self.logger.info(f"Created {total_combinations:,} population margin combinations")
            if quality_issues:
                self.logger.warning(f"Quality issues found: {quality_issues}")
                
            return True
            
        except Exception as e:
            self.logger.error(f"Population margins extraction failed: {e}")
            return False
    
    def extract_hospital_statistics(self) -> bool:
        """
        병원별 용량 통계 추출
        
        각 병원별 일일 용량 통계, KTAS별 용량을 계산하여
        hospital_capacity 테이블에 저장
        
        Returns:
            성공 여부
        """
        self.logger.info("Starting hospital statistics extraction")
        
        try:
            # 1. 기본 병원 정보 삽입/업데이트
            hospital_info_query = """
            INSERT OR REPLACE INTO nedis_meta.hospital_capacity 
                (emorg_cd, hospname, gubun, adr)
            SELECT DISTINCT 
                emorg_cd, 
                COALESCE(hospname, '') as hospname, 
                COALESCE(gubun, '') as gubun, 
                COALESCE(adr, '') as adr
            FROM nedis_original.nedis2017
            WHERE emorg_cd != '' AND emorg_cd IS NOT NULL
            """
            
            self.db.execute_query(hospital_info_query)
            
            # 2. 병원별 통계 계산 및 업데이트
            stats_query = """
            UPDATE nedis_meta.hospital_capacity 
            SET 
                daily_capacity_mean = stats.daily_capacity_mean,
                daily_capacity_std = stats.daily_capacity_std,
                ktas1_capacity = stats.ktas1_capacity,
                ktas2_capacity = stats.ktas2_capacity
            FROM (
                WITH daily_counts AS (
                    SELECT 
                        emorg_cd,
                        vst_dt,
                        COUNT(*) as daily_visits,
                        SUM(CASE WHEN ktas_fstu = '1' THEN 1 ELSE 0 END) as ktas1_count,
                        SUM(CASE WHEN ktas_fstu = '2' THEN 1 ELSE 0 END) as ktas2_count
                    FROM nedis_original.nedis2017
                    WHERE emorg_cd != '' AND emorg_cd IS NOT NULL
                      AND vst_dt != '' AND vst_dt IS NOT NULL
                    GROUP BY emorg_cd, vst_dt
                    HAVING COUNT(*) > 0
                )
                SELECT 
                    emorg_cd,
                    ROUND(AVG(daily_visits))::INTEGER as daily_capacity_mean,
                    GREATEST(1, ROUND(STDDEV(daily_visits)))::INTEGER as daily_capacity_std,
                    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ktas1_count))::INTEGER as ktas1_capacity,
                    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY ktas2_count))::INTEGER as ktas2_capacity
                FROM daily_counts
                GROUP BY emorg_cd
                HAVING COUNT(*) >= 30  -- 최소 30일 데이터가 있는 병원만
            ) AS stats
            WHERE nedis_meta.hospital_capacity.emorg_cd = stats.emorg_cd
            """
            
            self.db.execute_query(stats_query)
            
            # 3. 매력도 점수 계산 (중력모형용)
            self._calculate_hospital_attractiveness()
            
            # 4. 결과 검증
            verify_query = """
            SELECT 
                COUNT(*) as hospitals_with_stats,
                AVG(daily_capacity_mean) as avg_capacity,
                MIN(daily_capacity_mean) as min_capacity,
                MAX(daily_capacity_mean) as max_capacity
            FROM nedis_meta.hospital_capacity 
            WHERE daily_capacity_mean IS NOT NULL
            """
            
            result = self.db.fetch_dataframe(verify_query)
            stats_summary = result.iloc[0]
            
            self.logger.info(
                f"Updated statistics for {int(stats_summary['hospitals_with_stats']):,} hospitals"
            )
            self.logger.info(
                f"Capacity range: {int(stats_summary['min_capacity'])} - {int(stats_summary['max_capacity'])}"
                f" (avg: {int(stats_summary['avg_capacity'])})"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Hospital statistics extraction failed: {e}")
            return False
    
    def _calculate_hospital_attractiveness(self):
        """병원 매력도 점수 계산"""
        attractiveness_query = """
        UPDATE nedis_meta.hospital_capacity
        SET attractiveness_score = 
            daily_capacity_mean * 
            CASE 
                WHEN gubun LIKE '%권역%' THEN 2.0
                WHEN gubun LIKE '%지역센터%' THEN 1.5
                WHEN gubun LIKE '%지역기관%' THEN 1.0
                ELSE 1.0
            END
        WHERE daily_capacity_mean IS NOT NULL
        """
        
        self.db.execute_query(attractiveness_query)
        self.logger.debug("Hospital attractiveness scores calculated")
    
    def _check_original_data(self) -> bool:
        """원본 데이터 테이블 존재 및 기본 품질 확인"""
        try:
            # 테이블 존재 확인
            if not self.db.table_exists("nedis_original.nedis2017"):
                self.logger.error("Original data table 'nedis_original.nedis2017' not found")
                return False
                
            # 기본 데이터 개수 확인
            count_query = "SELECT COUNT(*) as total FROM nedis_original.nedis2017"
            total_records = self.db.fetch_dataframe(count_query)['total'][0]
            
            if total_records == 0:
                self.logger.error("Original data table is empty")
                return False
                
            self.logger.info(f"Found {total_records:,} records in original data")
            
            # 필수 컬럼 존재 확인
            required_columns = ['pat_do_cd', 'pat_age_gr', 'pat_sex', 'vst_dt', 'emorg_cd']
            schema_df = self.db.fetch_dataframe("DESCRIBE nedis_original.nedis2017")
            existing_columns = set(schema_df['column_name'].tolist())
            
            missing_columns = set(required_columns) - existing_columns
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Original data validation failed: {e}")
            return False
    
    def _validate_population_margins(self) -> Dict[str, Any]:
        """Population margins 데이터 품질 검증"""
        issues = {}
        
        try:
            # 1. 계절별 가중치 합계 확인
            seasonal_sum_query = """
            SELECT 
                AVG(seasonal_weight_spring + seasonal_weight_summer + 
                    seasonal_weight_fall + seasonal_weight_winter) as avg_seasonal_sum,
                MIN(seasonal_weight_spring + seasonal_weight_summer + 
                    seasonal_weight_fall + seasonal_weight_winter) as min_seasonal_sum,
                MAX(seasonal_weight_spring + seasonal_weight_summer + 
                    seasonal_weight_fall + seasonal_weight_winter) as max_seasonal_sum
            FROM nedis_meta.population_margins
            """
            
            seasonal_stats = self.db.fetch_dataframe(seasonal_sum_query).iloc[0]
            
            if abs(seasonal_stats['avg_seasonal_sum'] - 1.0) > 0.01:
                issues['seasonal_weights'] = f"Average seasonal sum: {seasonal_stats['avg_seasonal_sum']:.3f}"
                
            # 2. 요일별 가중치 합계 확인  
            weekday_sum_query = """
            SELECT 
                AVG(weekday_weight + weekend_weight) as avg_weekday_sum
            FROM nedis_meta.population_margins
            """
            
            weekday_stats = self.db.fetch_dataframe(weekday_sum_query).iloc[0]
            
            if abs(weekday_stats['avg_weekday_sum'] - 1.0) > 0.01:
                issues['weekday_weights'] = f"Average weekday sum: {weekday_stats['avg_weekday_sum']:.3f}"
                
            # 3. 음수 방문수 확인
            negative_visits_query = """
            SELECT COUNT(*) as negative_count
            FROM nedis_meta.population_margins
            WHERE yearly_visits <= 0
            """
            
            negative_count = self.db.fetch_dataframe(negative_visits_query)['negative_count'][0]
            if negative_count > 0:
                issues['negative_visits'] = f"{negative_count} combinations with <= 0 visits"
                
            # 4. 극값 확인
            extreme_values_query = """
            SELECT 
                MIN(yearly_visits) as min_visits,
                MAX(yearly_visits) as max_visits,
                AVG(yearly_visits) as avg_visits
            FROM nedis_meta.population_margins
            """
            
            extreme_stats = self.db.fetch_dataframe(extreme_values_query).iloc[0]
            
            if extreme_stats['max_visits'] > extreme_stats['avg_visits'] * 100:
                issues['extreme_values'] = f"Max visits ({int(extreme_stats['max_visits']):,}) >> avg ({int(extreme_stats['avg_visits']):,})"
                
        except Exception as e:
            issues['validation_error'] = str(e)
            
        return issues
    
    def generate_basic_report(self) -> Dict[str, Any]:
        """
        기본 데이터 프로파일 리포트 생성
        
        Returns:
            프로파일링 결과를 포함한 리포트 딕셔너리
        """
        report = {
            'generation_timestamp': datetime.now().isoformat(),
            'original_data': {},
            'population_margins': {},
            'hospital_capacity': {},
            'data_quality': {}
        }
        
        try:
            # 원본 데이터 통계
            report['original_data'] = self._get_original_data_stats()
            
            # Population margins 통계
            report['population_margins'] = self._get_population_margins_stats()
            
            # Hospital capacity 통계
            report['hospital_capacity'] = self._get_hospital_capacity_stats()
            
            # 데이터 품질 지표
            report['data_quality'] = self._get_data_quality_metrics()
            
            self.logger.info("Basic data profile report generated")
            
        except Exception as e:
            self.logger.error(f"Failed to generate basic report: {e}")
            report['error'] = str(e)
            
        return report
    
    def _get_original_data_stats(self) -> Dict[str, Any]:
        """원본 데이터 기본 통계"""
        stats = {}
        
        try:
            # 전체 레코드 수
            total_query = "SELECT COUNT(*) as total FROM nedis_original.nedis2017"
            stats['total_records'] = self.db.fetch_dataframe(total_query)['total'][0]
            
            # 시도별 분포 (상위 10개)
            region_query = """
            SELECT pat_do_cd, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017
            WHERE pat_do_cd != '' AND pat_do_cd IS NOT NULL
            GROUP BY pat_do_cd
            ORDER BY count DESC
            LIMIT 10
            """
            stats['top_regions'] = self.db.fetch_dataframe(region_query).to_dict('records')
            
            # 성별 분포
            gender_query = """
            SELECT pat_sex, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017
            WHERE pat_sex IN ('M', 'F')
            GROUP BY pat_sex
            ORDER BY pat_sex
            """
            stats['gender_distribution'] = self.db.fetch_dataframe(gender_query).to_dict('records')
            
            # KTAS 분포
            ktas_query = """
            SELECT ktas_fstu, COUNT(*) as count, 
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017
            WHERE ktas_fstu != '' AND ktas_fstu IS NOT NULL
            GROUP BY ktas_fstu
            ORDER BY ktas_fstu
            """
            stats['ktas_distribution'] = self.db.fetch_dataframe(ktas_query).to_dict('records')
            
            # 연령군 분포
            age_query = """
            SELECT pat_age_gr, COUNT(*) as count,
                   ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
            FROM nedis_original.nedis2017
            WHERE pat_age_gr != '' AND pat_age_gr IS NOT NULL
            GROUP BY pat_age_gr
            ORDER BY pat_age_gr
            """
            stats['age_distribution'] = self.db.fetch_dataframe(age_query).to_dict('records')
            
        except Exception as e:
            stats['error'] = str(e)
            
        return stats
    
    def _get_population_margins_stats(self) -> Dict[str, Any]:
        """Population margins 통계"""
        stats = {}
        
        try:
            # 기본 통계
            basic_query = """
            SELECT 
                COUNT(*) as total_combinations,
                SUM(yearly_visits) as total_visits,
                AVG(yearly_visits) as avg_visits,
                MIN(yearly_visits) as min_visits,
                MAX(yearly_visits) as max_visits
            FROM nedis_meta.population_margins
            """
            basic_stats = self.db.fetch_dataframe(basic_query).iloc[0]
            stats.update(basic_stats.to_dict())
            
            # 시도별 조합 수
            region_combinations_query = """
            SELECT pat_do_cd, COUNT(*) as combinations,
                   SUM(yearly_visits) as total_visits
            FROM nedis_meta.population_margins
            GROUP BY pat_do_cd
            ORDER BY total_visits DESC
            LIMIT 10
            """
            stats['top_regions_combinations'] = self.db.fetch_dataframe(
                region_combinations_query
            ).to_dict('records')
            
        except Exception as e:
            stats['error'] = str(e)
            
        return stats
    
    def _get_hospital_capacity_stats(self) -> Dict[str, Any]:
        """Hospital capacity 통계"""
        stats = {}
        
        try:
            # 기본 통계
            basic_query = """
            SELECT 
                COUNT(*) as total_hospitals,
                COUNT(CASE WHEN daily_capacity_mean IS NOT NULL THEN 1 END) as hospitals_with_stats,
                AVG(daily_capacity_mean) as avg_daily_capacity,
                MIN(daily_capacity_mean) as min_daily_capacity,
                MAX(daily_capacity_mean) as max_daily_capacity
            FROM nedis_meta.hospital_capacity
            """
            basic_stats = self.db.fetch_dataframe(basic_query).iloc[0]
            stats.update(basic_stats.to_dict())
            
            # 병원 종별 분포
            gubun_query = """
            SELECT gubun, COUNT(*) as count,
                   AVG(daily_capacity_mean) as avg_capacity
            FROM nedis_meta.hospital_capacity
            WHERE daily_capacity_mean IS NOT NULL
            GROUP BY gubun
            ORDER BY count DESC
            """
            stats['hospital_types'] = self.db.fetch_dataframe(gubun_query).to_dict('records')
            
        except Exception as e:
            stats['error'] = str(e)
            
        return stats
    
    def _get_data_quality_metrics(self) -> Dict[str, Any]:
        """데이터 품질 지표"""
        metrics = {}
        
        try:
            # Population margins 품질 지표
            margins_quality = self._validate_population_margins()
            metrics['population_margins_issues'] = len(margins_quality)
            metrics['population_margins_details'] = margins_quality
            
            # 결측값 분석 (원본 데이터)
            missing_query = """
            SELECT 
                'pat_do_cd' as field, 
                SUM(CASE WHEN pat_do_cd = '' OR pat_do_cd IS NULL THEN 1 ELSE 0 END) as missing_count,
                COUNT(*) as total_count
            FROM nedis_original.nedis2017
            UNION ALL
            SELECT 
                'pat_age_gr', 
                SUM(CASE WHEN pat_age_gr = '' OR pat_age_gr IS NULL THEN 1 ELSE 0 END),
                COUNT(*)
            FROM nedis_original.nedis2017
            UNION ALL
            SELECT 
                'pat_sex', 
                SUM(CASE WHEN pat_sex NOT IN ('M', 'F') THEN 1 ELSE 0 END),
                COUNT(*)
            FROM nedis_original.nedis2017
            """
            missing_stats = self.db.fetch_dataframe(missing_query)
            missing_stats['missing_percentage'] = (
                missing_stats['missing_count'] / missing_stats['total_count'] * 100
            ).round(2)
            
            metrics['missing_values'] = missing_stats.to_dict('records')
            
        except Exception as e:
            metrics['error'] = str(e)
            
        return metrics