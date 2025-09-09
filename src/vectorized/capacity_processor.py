#!/usr/bin/env python3
"""
병원 용량 제약 후처리기 (CapacityConstraintPostProcessor)

생성된 환자들에게 할당된 병원의 용량 제약을 적용하여 초과 환자를 재할당하는 모듈입니다.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


@dataclass
class CapacityConfig:
    """용량 제약 설정"""
    base_capacity_multiplier: float = 1.0
    weekend_capacity_multiplier: float = 0.8
    holiday_capacity_multiplier: float = 0.7
    safety_margin: float = 1.2
    overflow_redistribution_method: str = 'same_region_first'


class CapacityConstraintPostProcessor:
    """병원 용량 제약 후처리기"""
    
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
        
        # 병원 데이터 캐시
        self._hospital_capacity_cache = None
        self._hospital_choice_probs_cache = None
        
    def apply_capacity_constraints(self, patients_df: pd.DataFrame, 
                                  capacity_config: CapacityConfig) -> pd.DataFrame:
        """
        병원 용량 제약 적용
        
        Args:
            patients_df: 날짜와 초기 병원이 할당된 환자 데이터프레임
            capacity_config: 용량 제약 설정
            
        Returns:
            용량 제약이 적용된 환자 데이터프레임
        """
        self.logger.info(f"Applying capacity constraints to {len(patients_df):,} patients")
        
        # 필요한 참조 데이터 로드
        self._load_capacity_reference_data()
        
        # 용량 제약 계산
        daily_capacity_limits = self._calculate_dynamic_capacity_limits(capacity_config)
        
        # 현재 병원별 부하 계산
        current_loads = self._calculate_current_loads(patients_df)
        
        # Overflow 감지 및 재할당
        result_df = self._redistribute_overflow_patients(
            patients_df, current_loads, daily_capacity_limits, capacity_config
        )
        
        # 결과 검증
        self._validate_capacity_constraints(result_df, daily_capacity_limits)
        
        self.logger.info("Capacity constraint processing completed")
        return result_df
    
    def _load_capacity_reference_data(self):
        """용량 관련 참조 데이터 로드"""
        self.logger.info("Loading capacity reference data")
        
        # 병원 용량 정보
        if self._hospital_capacity_cache is None:
            self._hospital_capacity_cache = self.db.fetch_dataframe("""
                SELECT 
                    emorg_cd,
                    hospname,
                    adr,
                    daily_capacity_mean as capacity_beds,
                    COALESCE(ktas1_capacity + ktas2_capacity, daily_capacity_mean * 0.1) as effective_er_capacity
                FROM nedis_meta.hospital_capacity
            """)
        
        # 병원 선택 확률 (재할당 시 사용)
        if self._hospital_choice_probs_cache is None:
            self._hospital_choice_probs_cache = self.db.fetch_dataframe("""
                SELECT 
                    pat_do_cd, pat_age_gr, pat_sex, emorg_cd,
                    probability, rank
                FROM nedis_meta.hospital_choice_prob
                WHERE rank <= 3  -- Top 3 선호 병원만
            """)
    
    def _calculate_dynamic_capacity_limits(self, capacity_config: CapacityConfig) -> pd.DataFrame:
        """동적 용량 한계 계산"""
        self.logger.info("Calculating dynamic capacity limits")
        
        hospital_capacity = self._hospital_capacity_cache.copy()
        
        # 기본 용량에 안전 마진 적용
        hospital_capacity['base_daily_capacity'] = (
            hospital_capacity['effective_er_capacity'] * 
            capacity_config.base_capacity_multiplier * 
            capacity_config.safety_margin
        ).astype(int)
        
        # 날짜별 용량 조정을 위한 확장 (365일)
        dates = [f"2017{month:02d}{day:02d}" 
                for month in range(1, 13) 
                for day in range(1, 32)
                if self._is_valid_date(2017, month, day)][:365]
        
        # 날짜-병원 조합 생성
        capacity_limits = []
        
        for date_str in dates:
            date_obj = pd.to_datetime(date_str, format='%Y%m%d')
            is_weekend = date_obj.weekday() >= 5
            is_holiday = self._is_holiday_date(date_obj)
            
            for _, hospital in hospital_capacity.iterrows():
                daily_capacity = hospital['base_daily_capacity']
                
                # 주말 조정
                if is_weekend:
                    daily_capacity *= capacity_config.weekend_capacity_multiplier
                
                # 공휴일 조정
                if is_holiday:
                    daily_capacity *= capacity_config.holiday_capacity_multiplier
                
                capacity_limits.append({
                    'vst_dt': date_str,
                    'emorg_cd': hospital['emorg_cd'],
                    'daily_capacity_limit': max(1, int(daily_capacity)),
                    'base_capacity': hospital['base_daily_capacity'],
                    'is_weekend': is_weekend,
                    'is_holiday': is_holiday
                })
        
        capacity_df = pd.DataFrame(capacity_limits)
        self.logger.info(f"Calculated capacity limits for {len(capacity_df):,} date-hospital combinations")
        
        return capacity_df
    
    def _calculate_current_loads(self, patients_df: pd.DataFrame) -> pd.DataFrame:
        """현재 병원별 부하 계산"""
        current_loads = patients_df.groupby(['vst_dt', 'initial_hospital']).size().reset_index()
        current_loads.columns = ['vst_dt', 'emorg_cd', 'current_load']
        
        return current_loads
    
    def _redistribute_overflow_patients(self, patients_df: pd.DataFrame, 
                                      current_loads: pd.DataFrame,
                                      capacity_limits: pd.DataFrame,
                                      capacity_config: CapacityConfig) -> pd.DataFrame:
        """Overflow 환자 재할당"""
        self.logger.info("Redistributing overflow patients")
        
        result_df = patients_df.copy()
        result_df['emorg_cd'] = result_df['initial_hospital']  # 초기값 설정
        result_df['overflow_flag'] = False
        result_df['redistribution_method'] = 'initial'
        
        # 용량 제한과 현재 부하 비교
        load_vs_capacity = pd.merge(
            current_loads,
            capacity_limits,
            on=['vst_dt', 'emorg_cd'],
            how='left'
        )
        
        # Overflow 발생 케이스 식별
        overflow_cases = load_vs_capacity[
            load_vs_capacity['current_load'] > load_vs_capacity['daily_capacity_limit']
        ].copy()
        
        if len(overflow_cases) == 0:
            self.logger.info("No overflow cases detected")
            return result_df
        
        self.logger.info(f"Found {len(overflow_cases)} overflow cases")
        
        # 날짜-병원별로 처리
        total_redistributed = 0
        
        for _, overflow_case in overflow_cases.iterrows():
            date_str = overflow_case['vst_dt']
            hospital_cd = overflow_case['emorg_cd']
            excess_count = int(overflow_case['current_load'] - overflow_case['daily_capacity_limit'])
            
            if excess_count <= 0:
                continue
            
            # 해당 날짜-병원의 환자들 식별
            overflow_patients_mask = (
                (result_df['vst_dt'] == date_str) & 
                (result_df['emorg_cd'] == hospital_cd)
            )
            overflow_patient_indices = result_df[overflow_patients_mask].index.tolist()
            
            # 재할당할 환자들 랜덤 선택
            if len(overflow_patient_indices) >= excess_count:
                patients_to_redistribute = np.random.choice(
                    overflow_patient_indices, 
                    size=excess_count, 
                    replace=False
                )
                
                # 재할당 실행
                successful_redistributions = self._execute_patient_redistribution(
                    result_df, patients_to_redistribute, date_str, 
                    capacity_limits, capacity_config
                )
                
                total_redistributed += successful_redistributions
        
        self.logger.info(f"Successfully redistributed {total_redistributed} patients")
        return result_df
    
    def _execute_patient_redistribution(self, result_df: pd.DataFrame,
                                      patient_indices: np.ndarray,
                                      date_str: str,
                                      capacity_limits: pd.DataFrame,
                                      capacity_config: CapacityConfig) -> int:
        """개별 환자들의 재할당 실행"""
        successful_count = 0
        
        # 해당 날짜의 병원별 여유 용량 계산
        date_capacities = capacity_limits[capacity_limits['vst_dt'] == date_str].copy()
        
        # 현재 날짜의 병원별 부하 재계산
        current_date_loads = result_df[result_df['vst_dt'] == date_str].groupby('emorg_cd').size()
        
        for hospital_cd, capacity_limit in zip(date_capacities['emorg_cd'], date_capacities['daily_capacity_limit']):
            current_load = current_date_loads.get(hospital_cd, 0)
            date_capacities.loc[date_capacities['emorg_cd'] == hospital_cd, 'available_capacity'] = \
                max(0, capacity_limit - current_load)
        
        # 여유 용량이 있는 병원들만 선택
        available_hospitals = date_capacities[date_capacities['available_capacity'] > 0].copy()
        
        if len(available_hospitals) == 0:
            # Only log every 1000th case to avoid spam
            if hasattr(self, '_warning_count'):
                self._warning_count += 1
            else:
                self._warning_count = 1
            
            if self._warning_count % 1000 == 0:
                self.logger.warning(f"No available hospitals for redistribution (count: {self._warning_count})")
            return 0
        
        for patient_idx in patient_indices:
            if len(available_hospitals[available_hospitals['available_capacity'] > 0]) == 0:
                break
            
            patient_data = result_df.loc[patient_idx]
            
            # 재할당 병원 선택 (방법에 따라)
            new_hospital = self._select_redistribution_hospital(
                patient_data, available_hospitals, capacity_config
            )
            
            if new_hospital is not None:
                # 재할당 수행
                result_df.loc[patient_idx, 'emorg_cd'] = new_hospital
                result_df.loc[patient_idx, 'overflow_flag'] = True
                result_df.loc[patient_idx, 'redistribution_method'] = capacity_config.overflow_redistribution_method
                
                # 해당 병원의 여유 용량 감소
                hospital_mask = available_hospitals['emorg_cd'] == new_hospital
                available_hospitals.loc[hospital_mask, 'available_capacity'] -= 1
                
                successful_count += 1
        
        return successful_count
    
    def _select_redistribution_hospital(self, patient_data: pd.Series,
                                      available_hospitals: pd.DataFrame,
                                      capacity_config: CapacityConfig) -> Optional[str]:
        """재할당 병원 선택"""
        
        method = capacity_config.overflow_redistribution_method
        
        if method == 'random_available':
            # 여유 용량이 있는 병원 중 랜덤 선택
            candidates = available_hospitals[available_hospitals['available_capacity'] > 0]
            if len(candidates) > 0:
                return np.random.choice(candidates['emorg_cd'].values)
        
        elif method == 'same_region_first':
            # 동일 지역 내 병원 우선 재할당
            patient_region = str(patient_data['pat_do_cd'])
            # 병원 지역코드 파생 (pat_do_cd가 있으면 우선 사용, 없으면 adr 사용)
            hospital_df = self._hospital_capacity_cache.copy()
            if 'pat_do_cd' in hospital_df.columns:
                hospital_df['hospital_region'] = hospital_df['pat_do_cd'].astype(str)
            else:
                hospital_df['hospital_region'] = hospital_df['adr'].astype(str)
            hosp_regions = hospital_df[['emorg_cd', 'hospital_region']]

            candidates = available_hospitals[available_hospitals['available_capacity'] > 0].copy()
            candidates = pd.merge(candidates, hosp_regions, on='emorg_cd', how='left')
            same_region = candidates[candidates['hospital_region'] == patient_region]
            if len(same_region) > 0:
                return np.random.choice(same_region['emorg_cd'].values)
            else:
                if len(candidates) > 0:
                    return np.random.choice(candidates['emorg_cd'].values)
        
        elif method == 'second_choice_probability':
            # 원래 선호도 기반 재할당
            patient_profile = (
                patient_data['pat_do_cd'],
                patient_data['pat_age_gr'], 
                patient_data['pat_sex']
            )
            
            # 해당 환자 프로필의 병원 선택 확률
            choice_probs = self._hospital_choice_probs_cache[
                (self._hospital_choice_probs_cache['pat_do_cd'] == patient_profile[0]) &
                (self._hospital_choice_probs_cache['pat_age_gr'] == patient_profile[1]) &
                (self._hospital_choice_probs_cache['pat_sex'] == patient_profile[2]) &
                (self._hospital_choice_probs_cache['rank'] >= 2)  # 2nd choice 이상
            ].copy()
            
            if len(choice_probs) > 0:
                # 사용 가능한 병원과 교집합
                available_choices = pd.merge(
                    available_hospitals[available_hospitals['available_capacity'] > 0],
                    choice_probs,
                    left_on='emorg_cd',
                    right_on='emorg_cd',
                    how='inner'
                )
                
                if len(available_choices) > 0:
                    # 확률에 따라 선택
                    probs = available_choices['probability'].values
                    probs = probs / probs.sum()  # 정규화
                    
                    chosen_hospital = np.random.choice(
                        available_choices['emorg_cd'].values,
                        p=probs
                    )
                    return chosen_hospital
        
        # 모든 방법이 실패한 경우 None 반환
        return None
    
    def _validate_capacity_constraints(self, result_df: pd.DataFrame,
                                     capacity_limits: pd.DataFrame):
        """용량 제약 적용 결과 검증"""
        self.logger.info("Validating capacity constraint results")
        
        # 최종 병원별 부하 계산
        final_loads = result_df.groupby(['vst_dt', 'emorg_cd']).size().reset_index()
        final_loads.columns = ['vst_dt', 'emorg_cd', 'final_load']
        
        # 용량 제한과 비교
        load_vs_limit = pd.merge(
            final_loads,
            capacity_limits,
            on=['vst_dt', 'emorg_cd'],
            how='left'
        )
        
        # 여전히 초과하는 경우들 확인
        remaining_violations = load_vs_limit[
            load_vs_limit['final_load'] > load_vs_limit['daily_capacity_limit']
        ]
        
        if len(remaining_violations) > 0:
            self.logger.warning(f"Still {len(remaining_violations)} capacity violations remain")
            
            # 상세 정보 로깅
            total_excess = (remaining_violations['final_load'] - 
                           remaining_violations['daily_capacity_limit']).sum()
            self.logger.warning(f"Total excess patients: {total_excess}")
        else:
            self.logger.info("All capacity constraints successfully satisfied")
        
        # 재할당 통계
        redistribution_stats = result_df['overflow_flag'].value_counts()
        if True in redistribution_stats:
            redistributed_count = redistribution_stats[True]
            redistribution_rate = redistributed_count / len(result_df) * 100
            self.logger.info(f"Redistributed {redistributed_count:,} patients ({redistribution_rate:.1f}%)")
    
    def _is_valid_date(self, year: int, month: int, day: int) -> bool:
        """유효한 날짜인지 확인"""
        try:
            pd.Timestamp(year=year, month=month, day=day)
            return True
        except ValueError:
            return False
    
    def _is_holiday_date(self, date_obj: pd.Timestamp) -> bool:
        """공휴일 여부 확인 (간단한 휴리스틱)"""
        # 2017년 주요 공휴일들
        holidays_2017 = [
            (1, 1),   # 신정
            (1, 27), (1, 28), (1, 30),  # 설날 연휴
            (3, 1),   # 삼일절
            (5, 3),   # 부처님오신날
            (5, 5),   # 어린이날
            (5, 9),   # 대통령선거
            (6, 6),   # 현충일
            (8, 15),  # 광복절
            (10, 2), (10, 3), (10, 4), (10, 5), (10, 6), (10, 9),  # 추석+개천절+한글날
            (12, 25)  # 크리스마스
        ]
        
        return (date_obj.month, date_obj.day) in holidays_2017
    
    def generate_capacity_report(self, result_df: pd.DataFrame) -> Dict[str, Any]:
        """용량 제약 처리 결과 보고서 생성"""
        self.logger.info("Generating capacity constraint report")
        
        report = {}
        
        # 전체 통계
        report['total_patients'] = len(result_df)
        report['redistributed_patients'] = (result_df['overflow_flag'] == True).sum()
        report['redistribution_rate'] = report['redistributed_patients'] / report['total_patients'] * 100
        
        # 재할당 방법별 통계
        redistribution_methods = result_df[result_df['overflow_flag'] == True]['redistribution_method'].value_counts()
        report['redistribution_methods'] = redistribution_methods.to_dict()
        
        # 병원별 최종 부하 분포
        final_hospital_loads = result_df.groupby(['vst_dt', 'emorg_cd']).size()
        report['avg_daily_load_per_hospital'] = final_hospital_loads.mean()
        report['max_daily_load_per_hospital'] = final_hospital_loads.max()
        report['min_daily_load_per_hospital'] = final_hospital_loads.min()
        
        # 날짜별 재할당률
        daily_redistribution = result_df.groupby('vst_dt')['overflow_flag'].mean() * 100
        report['avg_daily_redistribution_rate'] = daily_redistribution.mean()
        report['max_daily_redistribution_rate'] = daily_redistribution.max()
        
        self.logger.info(f"Capacity report: {report['redistribution_rate']:.1f}% redistribution rate")
        
        return report
