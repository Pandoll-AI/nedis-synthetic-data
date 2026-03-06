#!/usr/bin/env python3
"""
시간 패턴 할당기 (TemporalPatternAssigner)

생성된 환자들에게 계절성과 주간 패턴을 반영한 날짜/시간을 할당하는 모듈입니다.
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..core.database import DatabaseManager
from ..core.config import ConfigManager
from ..analysis.pattern_analyzer import PatternAnalyzer, PatternAnalysisConfig


@dataclass
class TemporalConfig:
    """시간 패턴 설정"""
    year: int = 2017
    preserve_seasonality: bool = True
    preserve_weekly_pattern: bool = True
    preserve_holiday_effects: bool = True
    time_resolution: str = 'hourly'  # 'hourly' or 'daily'
    enable_conditional_hour_patterns: bool = True
    conditional_context_min_count: int = 30
    conditional_global_mix_weight: float = 0.20
    conditional_smoothing_alpha: float = 0.02


class TemporalPatternAssigner:
    """시간 패턴 할당기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 패턴 분석기 초기화
        self.pattern_analyzer = PatternAnalyzer(
            db_manager, config, PatternAnalysisConfig()
        )
        
        # 동적 시간 패턴 데이터
        self._temporal_patterns = None
        self._holiday_calendar = None
        
    def assign_temporal_patterns(self, patients_df: pd.DataFrame, 
                                temporal_config: TemporalConfig) -> pd.DataFrame:
        """
        환자들에게 시간 패턴 할당
        
        Args:
            patients_df: 날짜 없는 환자 데이터프레임
            temporal_config: 시간 패턴 설정
            
        Returns:
            날짜/시간이 할당된 환자 데이터프레임
        """
        self.logger.info(f"Assigning temporal patterns to {len(patients_df):,} patients")
        
        # 필요한 패턴 데이터 로드
        self._load_temporal_patterns(temporal_config.year)
        
        # 일별 볼륨 계산 (NHPP 기반)
        daily_volumes = self._calculate_daily_volumes(temporal_config)
        
        # 벡터화된 날짜 할당
        result_df = self._assign_dates_vectorized(patients_df, daily_volumes)
        
        # 시간 할당
        if temporal_config.time_resolution == 'hourly':
            result_df = self._assign_times_vectorized(result_df, temporal_config)
        else:
            # 기본 시간 할당 (균등 분포)
            result_df['ptmiintm'] = self._generate_random_times(len(result_df))
        
        self.logger.info(f"Temporal pattern assignment completed")
        return result_df
    
    def _load_temporal_patterns(self, year: int):
        """동적 시간 패턴 로드"""
        self.logger.info(f"Loading dynamic temporal patterns for year {year}")
        
        # 동적 패턴 분석 수행 (캐시 사용)
        if self._temporal_patterns is None:
            self._temporal_patterns = self.pattern_analyzer.analyze_all_patterns()
        
        # 공휴일 달력 로드
        if self._holiday_calendar is None:
            self._holiday_calendar = self._get_korean_holidays(year)
    
    def _calculate_daily_volumes(self, temporal_config: TemporalConfig) -> Dict[str, int]:
        """동적 패턴을 사용한 일별 볼륨 계산"""
        self.logger.info("Calculating daily volumes using dynamic temporal patterns")
        
        year = temporal_config.year
        
        # 동적 패턴에서 시간 패턴 추출
        temporal_data = self._temporal_patterns.get('temporal_patterns', {})
        
        # 요일별 패턴
        weekday_pattern = temporal_data.get('weekday_pattern', {})
        weekday_multipliers = {}
        if weekday_pattern:
            total_prob = sum(data['probability'] for data in weekday_pattern.values())
            for day, data in weekday_pattern.items():
                weekday_multipliers[int(day)] = data['probability'] / (total_prob / 7.0)
        else:
            # 기본 요일별 가중치 (균등 분포)
            weekday_multipliers = {i: 1.0 for i in range(7)}
        
        # 월별 패턴
        monthly_pattern = temporal_data.get('monthly_pattern', {})
        monthly_multipliers = {}
        if monthly_pattern:
            total_prob = sum(data['probability'] for data in monthly_pattern.values())
            for month, data in monthly_pattern.items():
                monthly_multipliers[int(month)] = data['probability'] / (total_prob / 12.0)
        else:
            # 기본 월별 가중치 (균등 분포)
            monthly_multipliers = {i: 1.0 for i in range(1, 13)}
        
        # 기본 통계 계산 (원본 데이터에서)
        # 원본 통계 테이블 선택
        src_table = self.pattern_analyzer.src_table if hasattr(self.pattern_analyzer, 'src_table') else 'nedis_original.emihptmi'
        original_stats = self.db.fetch_dataframe(f"""
            SELECT COUNT(*) as total_count,
                   COUNT(DISTINCT ptmiindt) as unique_dates
            FROM {src_table}
            WHERE ptmiindt IS NOT NULL
        """)
        
        mean_daily_count = original_stats['total_count'].iloc[0] / original_stats['unique_dates'].iloc[0]
        
        # 365일 볼륨 생성
        daily_volumes = {}
        start_date = datetime(year, 1, 1)
        
        for day_offset in range(365):
            current_date = start_date + timedelta(days=day_offset)
            date_str = current_date.strftime('%Y%m%d')
            
            # 기본 볼륨
            base_volume = mean_daily_count
            
            # 요일 효과 적용
            if temporal_config.preserve_weekly_pattern:
                dow = current_date.weekday()  # 0=월요일, 6=일요일
                # PostgreSQL의 DOW (0=일요일)로 변환
                dow_pg = (dow + 1) % 7
                weekday_mult = weekday_multipliers.get(dow_pg, 1.0)
                base_volume *= weekday_mult
            
            # 계절 효과 적용
            if temporal_config.preserve_seasonality:
                month = current_date.month
                seasonal_mult = monthly_multipliers.get(month, 1.0)
                base_volume *= seasonal_mult
            
            # 공휴일 효과 적용
            if temporal_config.preserve_holiday_effects:
                if self._is_holiday(current_date):
                    base_volume *= 1.3  # 공휴일 증가 효과
                elif self._is_holiday_adjacent(current_date):
                    base_volume *= 1.1  # 공휴일 인접 효과
            
            # 랜덤 변동 추가 (포아송 분포)
            daily_volumes[date_str] = max(1, int(np.random.poisson(base_volume)))
        
        # 총 합이 목표와 일치하도록 정규화
        current_total = sum(daily_volumes.values())
        target_total = len(daily_volumes) * mean_daily_count  # 예상 총합
        
        if current_total > 0:
            scale_factor = target_total / current_total
            for date_str in daily_volumes:
                daily_volumes[date_str] = max(1, int(daily_volumes[date_str] * scale_factor))
        
        self.logger.info(f"Generated daily volumes: total = {sum(daily_volumes.values()):,}")
        return daily_volumes
    
    def _assign_dates_vectorized(self, patients_df: pd.DataFrame, 
                                daily_volumes: Dict[str, int]) -> pd.DataFrame:
        """벡터화된 날짜 할당"""
        self.logger.info("Vectorized date assignment")
        
        total_patients = len(patients_df)
        total_volume_slots = sum(daily_volumes.values())
        
        # 환자 수와 볼륨 슬롯 수가 다를 경우 조정
        if total_patients != total_volume_slots:
            self.logger.warning(f"Patient count ({total_patients:,}) != volume slots ({total_volume_slots:,})")
            
            # 비율에 따라 일별 볼륨 조정
            adjustment_factor = total_patients / total_volume_slots
            adjusted_volumes = {}
            
            for date_str, volume in daily_volumes.items():
                adjusted_volume = max(1, int(volume * adjustment_factor))
                adjusted_volumes[date_str] = adjusted_volume
            
            # 남은 환자들을 랜덤하게 분배
            current_total = sum(adjusted_volumes.values())
            remaining = total_patients - current_total
            
            if remaining != 0:
                dates_list = list(adjusted_volumes.keys())
                for _ in range(abs(remaining)):
                    random_date = np.random.choice(dates_list)
                    if remaining > 0:
                        adjusted_volumes[random_date] += 1
                    else:
                        adjusted_volumes[random_date] = max(1, adjusted_volumes[random_date] - 1)
            
            daily_volumes = adjusted_volumes
        
        # 누적 분포 생성
        dates_list = sorted(daily_volumes.keys())
        volumes_list = [daily_volumes[date] for date in dates_list]
        cumulative_volumes = np.cumsum(volumes_list)
        
        # 환자별 날짜 할당
        random_positions = np.random.randint(0, cumulative_volumes[-1], total_patients)
        date_indices = np.searchsorted(cumulative_volumes, random_positions, side='right')
        
        # 데이터프레임에 날짜 할당
        result_df = patients_df.copy()
        result_df['ptmiindt'] = [dates_list[idx] for idx in date_indices]
        
        # 할당 결과 로깅
        assigned_counts = result_df['ptmiindt'].value_counts().sort_index()
        self.logger.info(f"Date assignment completed: {len(assigned_counts)} unique dates")
        
        return result_df
    
    def _assign_times_vectorized(self, patients_df: pd.DataFrame, 
                               temporal_config: TemporalConfig) -> pd.DataFrame:
        """동적 패턴을 사용한 벡터화된 시간 할당"""
        self.logger.info("Vectorized time assignment using dynamic patterns")
        
        result_df = patients_df.copy()
        
        # 동적 패턴에서 시간별 분포 추출
        temporal_data = self._temporal_patterns.get('temporal_patterns', {})
        hourly_pattern = temporal_data.get('hourly_pattern', {})
        conditional_patterns = (
            self._temporal_patterns
            .get('temporal_conditional_patterns', {})
            .get('patterns', {})
        )
        
        if (
            temporal_config.enable_conditional_hour_patterns
            and conditional_patterns
            and (hourly_pattern or conditional_patterns)
        ):
            result_df = self._assign_conditional_hour_patterns(
                result_df=result_df,
                hourly_pattern=hourly_pattern,
                conditional_patterns=conditional_patterns,
                temporal_config=temporal_config,
            )
        elif hourly_pattern:
            # 날짜별로 그룹화하여 처리
            time_assignments = np.empty(len(result_df), dtype='object')
            
            for _, group_indices in result_df.groupby('ptmiindt').groups.items():
                hours = list(hourly_pattern.keys())
                probabilities = [hourly_pattern[h]['probability'] for h in hours]
                
                # 확률 정규화
                probabilities = np.array(probabilities, dtype=float)
                probabilities = probabilities / probabilities.sum()
                
                # 시간 할당
                chosen_hours = np.random.choice(
                    [int(h) for h in hours], 
                    size=len(group_indices), 
                    p=probabilities
                )
                
                # 분과 초는 랜덤하게
                chosen_minutes = np.random.randint(0, 60, size=len(group_indices))
                
                # 시간 문자열 생성
                time_strings = [f"{h:02d}{m:02d}" for h, m in zip(chosen_hours, chosen_minutes)]
                time_assignments[list(group_indices)] = time_strings
            
            result_df['ptmiintm'] = time_assignments
        else:
            # 기본 시간 분포 사용
            result_df['ptmiintm'] = self._generate_random_times(len(result_df))
        
        return result_df

    def _assign_conditional_hour_patterns(
        self,
        result_df: pd.DataFrame,
        hourly_pattern: Dict[str, Any],
        conditional_patterns: Dict[str, Any],
        temporal_config: TemporalConfig,
    ) -> pd.DataFrame:
        """조건부 패턴 기반으로 시간대 할당."""
        if len(result_df) == 0:
            return result_df

        date_obj = pd.to_datetime(result_df['ptmiindt'].astype(str), format='%Y%m%d', errors='coerce')
        result_df = result_df.copy()
        result_df['_month'] = date_obj.dt.month.fillna(0).astype(int)
        # Convert pandas dayofweek (0=Mon,6=Sun) to DuckDB/PostgreSQL DOW (0=Sun,6=Sat)
        # to match the DOW convention used in conditional_patterns from SQL EXTRACT(DOW)
        result_df['_dow'] = ((date_obj.dt.dayofweek.fillna(0).astype(int) + 1) % 7)
        result_df['_age'] = result_df.get('ptmibrtd', pd.Series(['unknown'] * len(result_df), index=result_df.index)).fillna('unknown').astype(str)
        result_df['_sex'] = result_df.get('ptmisexx', pd.Series(['unknown'] * len(result_df), index=result_df.index)).fillna('unknown').astype(str)
        result_df['_ktas'] = result_df.get(
            'ptmikts1',
            result_df.get('ptmikpr1', pd.Series(['3'] * len(result_df), index=result_df.index)),
        ).fillna('unknown').astype(str)

        global_probs = self._hour_distribution_to_prob_vector(hourly_pattern)
        if global_probs is None:
            global_probs = np.ones(24, dtype=float) / 24.0
            if not conditional_patterns:
                result_df['ptmiintm'] = self._generate_random_times(len(result_df))
                return result_df.drop(columns=['_month', '_dow', '_age', '_sex', '_ktas'])

        result_df['ptmiintm'] = ''
        result_df['_hour_profile_source'] = ''

        for _, group_indices in result_df.groupby(['_month', '_dow', '_age', '_sex', '_ktas']).groups.items():
            idx = np.array(list(group_indices), dtype=int)
            if len(idx) == 0:
                continue

            first = idx[0]
            distribution, source = self._resolve_hour_distribution(
                month=int(result_df.iloc[first]['_month']),
                dow=int(result_df.iloc[first]['_dow']),
                age_group=str(result_df.iloc[first]['_age']),
                sex=str(result_df.iloc[first]['_sex']),
                ktas=str(result_df.iloc[first]['_ktas']),
                conditional_patterns=conditional_patterns,
                global_probs=global_probs,
                temporal_config=temporal_config,
            )
            result_df.loc[result_df.index[idx], '_hour_profile_source'] = source
            chosen_hours = np.random.choice(np.arange(24), size=len(idx), p=distribution)
            chosen_minutes = np.random.randint(0, 60, size=len(idx))
            result_df.loc[result_df.index[idx], 'ptmiintm'] = [
                f"{int(h):02d}{int(m):02d}" for h, m in zip(chosen_hours, chosen_minutes)
            ]

        return result_df.drop(columns=['_month', '_dow', '_age', '_sex', '_ktas', '_hour_profile_source'])

    def _resolve_hour_distribution(
        self,
        month: int,
        dow: int,
        age_group: str,
        sex: str,
        ktas: str,
        conditional_patterns: Dict[str, Any],
        global_probs: np.ndarray,
        temporal_config: TemporalConfig,
    ) -> Tuple[np.ndarray, str]:
        """조건별 시간 분포를 계층적으로 결합하고 스무딩."""
        candidate_specs = [
            ('month_dow_hour', f"{month}|{dow}", 0.35),
            ('month_hour', f"{month}", 0.25),
            ('dow_hour', f"{dow}", 0.15),
            ('age_sex_ktas_hour', f"{age_group}|{sex}|{ktas}", 0.10),
            ('ktas_age_hour', f"{ktas}|{age_group}", 0.08),
            ('ktas_hour', f"{ktas}", 0.07),
            ('age_hour', f"{age_group}", 0.05),
            ('age_sex_hour', f"{age_group}|{sex}", 0.05),
        ]

        weighted = np.zeros(24, dtype=float)
        used_weight = 0.0
        min_count = max(1, int(temporal_config.conditional_context_min_count))

        for level, key, weight in candidate_specs:
            level_patterns = conditional_patterns.get(level, {})
            raw_entry = level_patterns.get(key)
            if not isinstance(raw_entry, dict):
                continue
            if int(raw_entry.get('total_count', 0)) < min_count:
                continue
            probs = self._hour_distribution_to_prob_vector(raw_entry.get('patterns', {}))
            if probs is None:
                continue
            weighted += probs * weight
            used_weight += weight

        if used_weight <= 0.0:
            source = "fallback_global"
            return self._smooth_hour_distribution(global_probs, temporal_config.conditional_smoothing_alpha), source

        weighted = weighted / used_weight
        blend_weight = float(temporal_config.conditional_global_mix_weight)
        blend_weight = max(0.0, min(1.0, blend_weight))
        blended = (1.0 - blend_weight) * weighted + blend_weight * global_probs
        source = "conditional_blend"
        return self._smooth_hour_distribution(blended, temporal_config.conditional_smoothing_alpha), source

    @staticmethod
    def _hour_distribution_to_prob_vector(pattern: Dict[str, Any]) -> Optional[np.ndarray]:
        """패턴 맵을 24칸 확률 벡터로 변환."""
        if not pattern:
            return None

        probs = np.zeros(24, dtype=float)
        total = 0.0
        for hour, payload in pattern.items():
            try:
                hour_int = int(hour)
            except (TypeError, ValueError):
                continue
            if hour_int < 0 or hour_int > 23:
                continue
            value = payload.get('probability') if isinstance(payload, dict) else payload
            try:
                p = float(value)
            except (TypeError, ValueError):
                continue
            probs[hour_int] = max(0.0, p)
            total += probs[hour_int]
        if total <= 0.0:
            return None
        return probs / total

    @staticmethod
    def _smooth_hour_distribution(probabilities: np.ndarray, alpha: float) -> np.ndarray:
        """분포를 스무딩."""
        alpha = max(0.0, min(1.0, float(alpha)))
        if alpha <= 0.0:
            return probabilities / max(float(probabilities.sum()), 1e-12)
        base = np.ones(24, dtype=float) / 24.0
        mixed = (1.0 - alpha) * probabilities + alpha * base
        return mixed / max(float(mixed.sum()), 1e-12)
    
    def _generate_random_times(self, count: int) -> List[str]:
        """랜덤 시간 생성 (설정 기반 폴백 분포)"""
        fallback_weights = self.config.get('temporal.fallback_hour_weights')
        if not fallback_weights or not isinstance(fallback_weights, list) or len(fallback_weights) != 24:
            # 균등 분포 폴백
            hour_probs = np.ones(24) / 24.0
        else:
            hour_weights = np.array(fallback_weights, dtype=float)
            hour_probs = hour_weights / hour_weights.sum()

        hours = np.random.choice(24, size=count, p=hour_probs)
        minutes = np.random.randint(0, 60, size=count)
        return [f"{h:02d}{m:02d}" for h, m in zip(hours, minutes)]
    
    def _get_korean_holidays(self, year: int) -> List[datetime]:
        """한국 공휴일 목록 생성"""
        holidays = [
            # 고정 공휴일
            datetime(year, 1, 1),   # 신정
            datetime(year, 3, 1),   # 삼일절
            datetime(year, 5, 5),   # 어린이날
            datetime(year, 6, 6),   # 현충일
            datetime(year, 8, 15),  # 광복절
            datetime(year, 10, 3),  # 개천절
            datetime(year, 10, 9),  # 한글날
            datetime(year, 12, 25), # 크리스마스
        ]
        
        # 2017년 특정 공휴일들 (음력 기반이라 매년 다름)
        if year == 2017:
            holidays.extend([
                datetime(2017, 1, 27),  # 설날 전날
                datetime(2017, 1, 28),  # 설날
                datetime(2017, 1, 30),  # 설날 연휴
                datetime(2017, 5, 3),   # 부처님오신날
                datetime(2017, 5, 9),   # 대통령 선거일
                datetime(2017, 10, 2),  # 임시공휴일
                datetime(2017, 10, 4),  # 추석 연휴
                datetime(2017, 10, 5),  # 추석
                datetime(2017, 10, 6),  # 추석 연휴
            ])
        
        return holidays
    
    def _is_holiday(self, date: datetime) -> bool:
        """공휴일 여부 확인"""
        return date in self._holiday_calendar
    
    def _is_holiday_adjacent(self, date: datetime) -> bool:
        """공휴일 인접일 여부 확인 (전후 1일)"""
        prev_day = date - timedelta(days=1)
        next_day = date + timedelta(days=1)
        return prev_day in self._holiday_calendar or next_day in self._holiday_calendar
    
    def validate_temporal_assignment(self, result_df: pd.DataFrame, 
                                   temporal_config: TemporalConfig) -> Dict[str, Any]:
        """시간 할당 결과 검증"""
        self.logger.info("Validating temporal assignment results")
        
        validation_results = {}
        
        # 날짜 범위 검증
        min_date = result_df['ptmiindt'].min()
        max_date = result_df['ptmiindt'].max()
        expected_start = f"{temporal_config.year}0101"
        expected_end = f"{temporal_config.year}1231"
        
        validation_results['date_range'] = {
            'min_date': min_date,
            'max_date': max_date,
            'expected_start': expected_start,
            'expected_end': expected_end,
            'range_valid': min_date >= expected_start and max_date <= expected_end
        }
        
        # 요일별 분포 검증
        result_df['date_obj'] = pd.to_datetime(result_df['ptmiindt'], format='%Y%m%d')
        result_df['day_of_week'] = result_df['date_obj'].dt.dayofweek
        
        dow_distribution = result_df['day_of_week'].value_counts(normalize=True).sort_index()
        validation_results['weekday_distribution'] = dow_distribution.to_dict()
        
        # 월별 분포 검증
        result_df['month'] = result_df['date_obj'].dt.month
        monthly_distribution = result_df['month'].value_counts(normalize=True).sort_index()
        validation_results['monthly_distribution'] = monthly_distribution.to_dict()
        
        # 시간 분포 검증 (시간이 할당된 경우)
        if 'ptmiintm' in result_df.columns:
            result_df['hour'] = result_df['ptmiintm'].str[:2].astype(int)
            hourly_distribution = result_df['hour'].value_counts(normalize=True).sort_index()
            validation_results['hourly_distribution'] = hourly_distribution.to_dict()
            if '_hour_profile_source' in result_df.columns:
                validation_results['conditional_hour_profile_usage'] = (
                    result_df['_hour_profile_source'].value_counts().to_dict()
                )
                result_df = result_df.drop(columns=['_hour_profile_source'])
        
        # 총 일수와 평균 일별 방문자 수
        unique_dates = result_df['ptmiindt'].nunique()
        avg_daily_visits = len(result_df) / unique_dates
        
        validation_results['summary'] = {
            'total_patients': len(result_df),
            'unique_dates': unique_dates,
            'avg_daily_visits': avg_daily_visits,
            'date_coverage_days': unique_dates
        }
        
        self.logger.info(f"Temporal validation completed: {unique_dates} unique dates, "
                        f"avg {avg_daily_visits:.1f} visits/day")
        
        return validation_results
