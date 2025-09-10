"""
비균질 포아송 과정(NHPP) 시간 패턴 생성기

연간 볼륨을 일별로 분해하여 현실적인 시간 패턴을 생성합니다.
계절별, 요일별, 공휴일 효과를 반영하여 의료 응급실 방문의
자연스러운 시간적 변동을 모델링합니다.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import calendar
import logging
from tqdm import tqdm
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.database import DatabaseManager
from src.core.config import ConfigManager


class NHPPTemporalGenerator:
    """비균질 포아송 과정 시간 패턴 생성기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        NHPP 시간 패턴 생성기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자 인스턴스
            config: 설정 관리자 인스턴스
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 공휴일 정의 (2017년 기준)
        self.holidays_2017 = config.get('temporal.holidays_2017', [
            '20170101', '20170127', '20170128', '20170129', '20170130',  # 신정, 설날 연휴
            '20170301',  # 3.1절
            '20170503', '20170509',  # 어린이날, 부처님오신날
            '20170506',  # 어린이날 대체공휴일
            '20170815',  # 광복절
            '20171003', '20171004', '20171005', '20171006',  # 추석 연휴
            '20171009',  # 한글날
            '20171225'   # 성탄절
        ])
        
        # 성능 최적화를 위한 설정
        self.batch_size = config.get('database.batch_size', 10000)
        self.chunk_size = 1000  # 메모리 효율적 처리를 위한 청크 크기
        
    def generate_daily_events(self, year: int = 2017) -> bool:
        """
        일별 이벤트 분해 생성
        
        연간 볼륨을 NHPP를 사용하여 365일로 분해하고,
        계절별, 요일별, 공휴일 효과를 반영합니다.
        
        Args:
            year: 생성할 연도
            
        Returns:
            성공 여부
        """
        self.logger.info(f"Starting daily event generation for year {year}")
        
        try:
            # 1. 연간 볼륨 데이터 존재 확인
            if not self.db.table_exists("nedis_synthetic.yearly_volumes"):
                raise Exception("yearly_volumes table not found. Run Phase 1 first.")
            
            yearly_count = self.db.get_table_count("nedis_synthetic.yearly_volumes")
            if yearly_count == 0:
                raise Exception("yearly_volumes table is empty")
                
            self.logger.info(f"Found {yearly_count:,} yearly volume combinations")
            
            # 2. 기존 일별 데이터 삭제
            self.db.execute_query("DELETE FROM nedis_synthetic.daily_volumes")
            
            # 3. 365일 날짜 리스트 생성
            date_list = self._generate_date_list(year)
            self.logger.info(f"Generated {len(date_list)} dates for year {year}")
            
            # 4. 청크 단위로 처리 (메모리 효율성)
            total_processed = 0
            total_daily_records = 0
            
            for chunk_data in self._get_yearly_volume_chunks():
                if len(chunk_data) == 0:
                    continue
                    
                chunk_daily_volumes = []
                
                # 각 연간 볼륨 조합을 일별로 분해
                for _, volume_row in chunk_data.iterrows():
                    daily_volumes = self._decompose_to_daily(volume_row, date_list)
                    chunk_daily_volumes.extend(daily_volumes)
                
                # 배치 저장
                if chunk_daily_volumes:
                    self._save_daily_batch(chunk_daily_volumes)
                    total_daily_records += len(chunk_daily_volumes)
                
                total_processed += len(chunk_data)
                
                if total_processed % (self.chunk_size * 10) == 0:
                    self.logger.info(f"Processed {total_processed:,}/{yearly_count:,} yearly combinations")
            
            # 5. 결과 검증
            daily_table_count = self.db.get_table_count("nedis_synthetic.daily_volumes")
            
            # 총합 검증
            yearly_total_query = "SELECT SUM(synthetic_yearly_count) as total FROM nedis_synthetic.yearly_volumes"
            daily_total_query = "SELECT SUM(synthetic_daily_count) as total FROM nedis_synthetic.daily_volumes"
            
            yearly_total = self.db.fetch_dataframe(yearly_total_query)['total'][0]
            daily_total = self.db.fetch_dataframe(daily_total_query)['total'][0]
            
            self.logger.info(f"Generated {daily_table_count:,} daily volume records")
            self.logger.info(f"Total verification: Yearly={yearly_total:,}, Daily={daily_total:,}")
            
            # 허용 가능한 차이 확인 (1% 이내)
            if yearly_total > 0:
                difference_pct = abs(yearly_total - daily_total) / yearly_total * 100
                if difference_pct > 1.0:
                    self.logger.warning(f"Large discrepancy: {difference_pct:.2f}%")
                else:
                    self.logger.info(f"Total difference: {difference_pct:.3f}% (acceptable)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Daily event generation failed: {e}")
            return False
    
    def _generate_date_list(self, year: int) -> List[str]:
        """연도별 날짜 리스트 생성 (YYYYMMDD 형식)"""
        start_date = datetime(year, 1, 1)
        end_date = datetime(year, 12, 31)
        
        date_list = []
        current_date = start_date
        
        while current_date <= end_date:
            date_str = current_date.strftime('%Y%m%d')
            date_list.append(date_str)
            current_date += timedelta(days=1)
            
        return date_list
    
    def _get_yearly_volume_chunks(self):
        """연간 볼륨 데이터를 청크 단위로 반환"""
        offset = 0
        
        while True:
            # 시간 가중치 정보와 함께 조회
            query = f"""
            SELECT 
                yv.pat_do_cd, yv.pat_age_gr, yv.pat_sex, yv.synthetic_yearly_count,
                pm.seasonal_weight_spring, pm.seasonal_weight_summer,
                pm.seasonal_weight_fall, pm.seasonal_weight_winter,
                pm.weekday_weight, pm.weekend_weight
            FROM nedis_synthetic.yearly_volumes yv
            JOIN nedis_meta.population_margins pm 
                ON yv.pat_do_cd = pm.pat_do_cd 
                AND yv.pat_age_gr = pm.pat_age_gr 
                AND yv.pat_sex = pm.pat_sex
            ORDER BY yv.pat_do_cd, yv.pat_age_gr, yv.pat_sex
            LIMIT {self.chunk_size} OFFSET {offset}
            """
            
            chunk_data = self.db.fetch_dataframe(query)
            
            if len(chunk_data) == 0:
                break
                
            yield chunk_data
            offset += self.chunk_size
    
    def _decompose_to_daily(self, volume_row: pd.Series, date_list: List[str]) -> List[Tuple]:
        """연간 볼륨을 일별로 분해"""
        yearly_count = volume_row['synthetic_yearly_count']
        
        if yearly_count <= 0:
            return []
            
        base_lambda = yearly_count / 365.0
        
        results = []
        daily_counts = []
        lambda_values = []
        
        # 1. 각 날짜별 강도 계산
        for date_str in date_list:
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            
            # 계절 가중치 계산
            seasonal_weight = self._get_seasonal_weight(date_obj, volume_row)
            
            # 요일 가중치 계산
            weekday_weight = self._get_weekday_weight(date_obj, volume_row)
            
            # 공휴일 가중치 (공휴일은 일반적으로 응급실 방문이 증가)
            holiday_weight = 1.2 if date_str in self.holidays_2017 else 1.0
            
            # 최종 강도 계산
            lambda_t = base_lambda * seasonal_weight * weekday_weight * holiday_weight
            lambda_values.append(lambda_t)
            
            # 포아송 샘플링
            daily_count = np.random.poisson(lambda_t)
            daily_counts.append(daily_count)
        
        # 2. Rescaling (연간 총합 맞추기)
        total_generated = sum(daily_counts)
        if total_generated > 0:
            scaling_factor = yearly_count / total_generated
            
            # 스케일링 적용 및 정수화
            scaled_counts = []
            for count in daily_counts:
                scaled_count = count * scaling_factor
                # 확률적 반올림 (fractional part를 확률로 사용)
                integer_part = int(scaled_count)
                fractional_part = scaled_count - integer_part
                
                if np.random.random() < fractional_part:
                    integer_part += 1
                    
                scaled_counts.append(max(0, integer_part))
                
            daily_counts = scaled_counts
        
        # 3. 결과 구성
        for i, date_str in enumerate(date_list):
            if daily_counts[i] > 0:  # 0인 날은 저장하지 않음 (저장 공간 절약)
                results.append((
                    date_str,
                    volume_row['pat_do_cd'],
                    volume_row['pat_age_gr'], 
                    volume_row['pat_sex'],
                    daily_counts[i],
                    lambda_values[i]
                ))
        
        return results
    
    def _get_seasonal_weight(self, date_obj: datetime, volume_row: pd.Series) -> float:
        """계절별 가중치 계산"""
        month = date_obj.month
        
        if month in [3, 4, 5]:  # 봄
            return volume_row['seasonal_weight_spring']
        elif month in [6, 7, 8]:  # 여름
            return volume_row['seasonal_weight_summer']
        elif month in [9, 10, 11]:  # 가을
            return volume_row['seasonal_weight_fall']
        else:  # 겨울 (12, 1, 2월)
            return volume_row['seasonal_weight_winter']
    
    def _get_weekday_weight(self, date_obj: datetime, volume_row: pd.Series) -> float:
        """요일별 가중치 계산"""
        weekday = date_obj.weekday()  # 0=Monday, 6=Sunday
        
        if weekday in [5, 6]:  # 토요일, 일요일
            return volume_row['weekend_weight']
        else:
            return volume_row['weekday_weight']
    
    def _save_daily_batch(self, daily_batch: List[Tuple]):
        """일별 볼륨 배치 저장"""
        if not daily_batch:
            return
            
        try:
            # DataFrame으로 변환
            daily_df = pd.DataFrame(daily_batch, columns=[
                'vst_dt', 'pat_do_cd', 'pat_age_gr', 'pat_sex', 
                'synthetic_daily_count', 'lambda_value'
            ])
            
            # 데이터 타입 최적화
            daily_df['synthetic_daily_count'] = daily_df['synthetic_daily_count'].astype('int32')
            daily_df['lambda_value'] = daily_df['lambda_value'].astype('float32')
            
            # DuckDB에 삽입
            self.db.conn.execute("""
                INSERT INTO nedis_synthetic.daily_volumes 
                SELECT * FROM daily_df
            """)
            
            self.logger.debug(f"Saved daily batch: {len(daily_batch):,} records")
            
        except Exception as e:
            self.logger.error(f"Failed to save daily batch: {e}")
            raise
    
    def get_temporal_summary(self) -> Dict[str, Any]:
        """시간 패턴 생성 결과 요약"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'parameters': {
                'holidays_count': len(self.holidays_2017),
                'chunk_size': self.chunk_size,
                'batch_size': self.batch_size
            }
        }
        
        try:
            if not self.db.table_exists("nedis_synthetic.daily_volumes"):
                summary['error'] = "daily_volumes table not found"
                return summary
            
            # 기본 통계
            basic_stats_query = """
            SELECT 
                COUNT(*) as total_daily_records,
                COUNT(DISTINCT vst_dt) as unique_dates,
                SUM(synthetic_daily_count) as total_visits,
                AVG(synthetic_daily_count) as avg_daily_visits,
                MIN(synthetic_daily_count) as min_daily_visits,
                MAX(synthetic_daily_count) as max_daily_visits,
                AVG(lambda_value) as avg_lambda,
                COUNT(DISTINCT pat_do_cd || '|' || pat_age_gr || '|' || pat_sex) as unique_combinations
            FROM nedis_synthetic.daily_volumes
            """
            
            stats = self.db.fetch_dataframe(basic_stats_query).iloc[0]
            summary['basic_statistics'] = stats.to_dict()
            
            # 날짜별 총 방문수 분포 (상위 10일)
            daily_totals_query = """
            SELECT 
                vst_dt,
                SUM(synthetic_daily_count) as daily_total,
                -- 요일 계산
                CASE DAYOFWEEK(CAST(vst_dt AS DATE))
                    WHEN 1 THEN 'Sunday'
                    WHEN 2 THEN 'Monday'
                    WHEN 3 THEN 'Tuesday'
                    WHEN 4 THEN 'Wednesday'
                    WHEN 5 THEN 'Thursday'
                    WHEN 6 THEN 'Friday'
                    WHEN 7 THEN 'Saturday'
                END as weekday
            FROM nedis_synthetic.daily_volumes
            GROUP BY vst_dt
            ORDER BY daily_total DESC
            LIMIT 10
            """
            
            summary['top_volume_days'] = self.db.fetch_dataframe(daily_totals_query).to_dict('records')
            
            # 요일별 패턴 분석
            weekday_pattern_query = """
            SELECT 
                CASE DAYOFWEEK(CAST(vst_dt AS DATE))
                    WHEN 1 THEN 'Sunday'
                    WHEN 2 THEN 'Monday'
                    WHEN 3 THEN 'Tuesday'
                    WHEN 4 THEN 'Wednesday'
                    WHEN 5 THEN 'Thursday'
                    WHEN 6 THEN 'Friday'
                    WHEN 7 THEN 'Saturday'
                END as weekday,
                DAYOFWEEK(CAST(vst_dt AS DATE)) as weekday_num,
                AVG(synthetic_daily_count) as avg_visits,
                COUNT(DISTINCT vst_dt) as days_count
            FROM nedis_synthetic.daily_volumes
            GROUP BY DAYOFWEEK(CAST(vst_dt AS DATE))
            ORDER BY weekday_num
            """
            
            summary['weekday_patterns'] = self.db.fetch_dataframe(weekday_pattern_query).to_dict('records')
            
            # 월별 패턴 분석
            monthly_pattern_query = """
            SELECT 
                MONTH(CAST(vst_dt AS DATE)) as month,
                AVG(synthetic_daily_count) as avg_daily_visits,
                SUM(synthetic_daily_count) as total_monthly_visits,
                COUNT(DISTINCT vst_dt) as days_in_month
            FROM nedis_synthetic.daily_volumes
            GROUP BY MONTH(CAST(vst_dt AS DATE))
            ORDER BY month
            """
            
            summary['monthly_patterns'] = self.db.fetch_dataframe(monthly_pattern_query).to_dict('records')
            
        except Exception as e:
            summary['error'] = str(e)
            
        return summary
    
    def validate_temporal_patterns(self) -> Dict[str, Any]:
        """시간 패턴 검증"""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'issues': []
        }
        
        try:
            # 1. 날짜 범위 확인
            date_range_query = """
            SELECT 
                MIN(vst_dt) as min_date,
                MAX(vst_dt) as max_date,
                COUNT(DISTINCT vst_dt) as unique_dates
            FROM nedis_synthetic.daily_volumes
            """
            
            date_stats = self.db.fetch_dataframe(date_range_query).iloc[0]
            validation['checks']['date_range'] = date_stats.to_dict()
            
            if date_stats['unique_dates'] != 365:
                validation['issues'].append(f"Expected 365 unique dates, got {date_stats['unique_dates']}")
            
            # 2. 음수 값 확인
            negative_query = """
            SELECT 
                COUNT(CASE WHEN synthetic_daily_count < 0 THEN 1 END) as negative_counts,
                COUNT(CASE WHEN lambda_value < 0 THEN 1 END) as negative_lambdas
            FROM nedis_synthetic.daily_volumes
            """
            
            negative_stats = self.db.fetch_dataframe(negative_query).iloc[0]
            validation['checks']['negative_values'] = negative_stats.to_dict()
            
            if negative_stats['negative_counts'] > 0:
                validation['issues'].append(f"{negative_stats['negative_counts']} negative daily counts")
            if negative_stats['negative_lambdas'] > 0:
                validation['issues'].append(f"{negative_stats['negative_lambdas']} negative lambda values")
            
            # 3. 총합 일치성 확인
            if self.db.table_exists("nedis_synthetic.yearly_volumes"):
                comparison_query = """
                SELECT 
                    (SELECT SUM(synthetic_yearly_count) FROM nedis_synthetic.yearly_volumes) as yearly_total,
                    (SELECT SUM(synthetic_daily_count) FROM nedis_synthetic.daily_volumes) as daily_total
                """
                
                totals = self.db.fetch_dataframe(comparison_query).iloc[0]
                validation['checks']['total_consistency'] = totals.to_dict()
                
                if totals['yearly_total'] > 0:
                    error_rate = abs(totals['yearly_total'] - totals['daily_total']) / totals['yearly_total']
                    validation['checks']['total_error_rate'] = error_rate
                    
                    if error_rate > 0.01:  # 1% 이상 차이
                        validation['issues'].append(f"Large total discrepancy: {error_rate:.2%}")
            
            # 4. 극값 확인 (일별 방문수)
            extreme_query = """
            WITH daily_totals AS (
                SELECT vst_dt, SUM(synthetic_daily_count) as daily_total
                FROM nedis_synthetic.daily_volumes
                GROUP BY vst_dt
            ),
            stats AS (
                SELECT 
                    AVG(daily_total) as avg_daily,
                    STDDEV(daily_total) as std_daily
                FROM daily_totals
            )
            SELECT 
                COUNT(CASE WHEN dt.daily_total > s.avg_daily + 3 * s.std_daily THEN 1 END) as extreme_high_days,
                MAX(dt.daily_total) as max_daily,
                MIN(dt.daily_total) as min_daily,
                s.avg_daily, s.std_daily
            FROM daily_totals dt, stats s
            """
            
            extreme_stats = self.db.fetch_dataframe(extreme_query).iloc[0]
            validation['checks']['extreme_values'] = extreme_stats.to_dict()
            
            if extreme_stats['extreme_high_days'] > 10:  # 10일 이상이 극값이면 문제
                validation['issues'].append(f"{extreme_stats['extreme_high_days']} days with extremely high values")
            
            # 검증 통과 여부
            validation['passed'] = len(validation['issues']) == 0
            
        except Exception as e:
            validation['error'] = str(e)
            validation['passed'] = False
            
        return validation