"""
Duration Generator

응급실 체류시간과 입원 기간 생성기입니다.
KTAS별 의학적 근거에 기반한 현실적인 체류시간과 입원 기간을 모델링합니다.

주요 기능:
- KTAS별 응급실 체류시간 생성
- 치료결과별 입원 기간 생성  
- 시간 일관성 검증 및 조정
- 의학적 타당성 확보
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
from scipy import stats

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class DurationGenerator:
    """체류시간 및 입원 기간 생성기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        체류시간 생성기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # KTAS별 응급실 체류시간 파라미터 (분 단위)
        self.er_duration_params = {
            '1': {  # 중증 - 긴 체류시간
                'mean': 280,      # 평균 4시간 40분
                'std': 150,       # 표준편차 2.5시간
                'min': 60,        # 최소 1시간
                'max': 720,       # 최대 12시간
                'lognorm_sigma': 0.6,  # 로그정규분포 형태 파라미터
                'long_stay_prob': 0.25  # 장기 체류 확률
            },
            '2': {  # 응급
                'mean': 220,      # 평균 3시간 40분
                'std': 120,       # 표준편차 2시간
                'min': 45,        # 최소 45분
                'max': 600,       # 최대 10시간
                'lognorm_sigma': 0.5,
                'long_stay_prob': 0.15
            },
            '3': {  # 긴급
                'mean': 140,      # 평균 2시간 20분
                'std': 80,        # 표준편차 1시간 20분
                'min': 30,        # 최소 30분
                'max': 480,       # 최대 8시간
                'lognorm_sigma': 0.4,
                'long_stay_prob': 0.10
            },
            '4': {  # 준응급
                'mean': 100,      # 평균 1시간 40분
                'std': 60,        # 표준편차 1시간
                'min': 20,        # 최소 20분
                'max': 360,       # 최대 6시간
                'lognorm_sigma': 0.35,
                'long_stay_prob': 0.08
            },
            '5': {  # 비응급
                'mean': 70,       # 평균 1시간 10분
                'std': 40,        # 표준편차 40분
                'min': 15,        # 최소 15분
                'max': 240,       # 최대 4시간
                'lognorm_sigma': 0.3,
                'long_stay_prob': 0.05
            }
        }
        
        # 입원 기간 파라미터 (일 단위)
        self.admission_duration_params = {
            '31': {  # 병실 입원
                'mean': 7.2,          # 평균 7.2일
                'std': 8.5,           # 표준편차 8.5일
                'min': 1,             # 최소 1일
                'max': 60,            # 최대 60일
                'zero_inflated_prob': 0.05,  # 당일 퇴원 확률
                'long_stay_prob': 0.15       # 장기 입원 확률 (14일+)
            },
            '32': {  # 중환자실 입원
                'mean': 14.5,         # 평균 14.5일
                'std': 18.0,          # 표준편차 18일
                'min': 1,             # 최소 1일
                'max': 90,            # 최대 90일
                'zero_inflated_prob': 0.02,  # 당일 퇴원 확률 낮음
                'long_stay_prob': 0.35       # 장기 입원 확률 높음
            }
        }
        
        # 입원 결과별 확률 분포
        self.admission_outcome_probs = {
            '31': {  # 병실 입원
                '1': 0.75,  # 완쾌
                '2': 0.18,  # 호전
                '3': 0.05,  # 미호전
                '4': 0.02   # 사망
            },
            '32': {  # 중환자실 입원
                '1': 0.45,  # 완쾌
                '2': 0.25,  # 호전
                '3': 0.15,  # 미호전
                '4': 0.15   # 사망 (높은 확률)
            }
        }
        
        # 연령별 체류시간 조정 계수
        self.age_duration_factors = {
            '01': 1.2,  # 영아 - 더 긴 체류
            '09': 1.1,  # 유아
            '10': 0.9,  # 10대 - 짧은 체류
            '20': 0.9,  # 20대
            '30': 1.0,  # 30대 - 기준
            '40': 1.0,  # 40대
            '50': 1.1,  # 50대
            '60': 1.2,  # 60대
            '70': 1.3,  # 70대 - 긴 체류
            '80': 1.4,  # 80대
            '90': 1.5   # 90대 - 가장 긴 체류
        }
    
    def generate_all_durations(self, date_str: str) -> Dict[str, Any]:
        """
        특정 날짜의 모든 환자에 대해 체류시간 생성
        
        Args:
            date_str: 대상 날짜 ('YYYYMMDD' 형식)
            
        Returns:
            체류시간 생성 결과 딕셔너리
        """
        
        self.logger.info(f"Generating durations for date: {date_str}")
        
        try:
            # 해당 날짜 임상 레코드 조회
            clinical_records = self.db.fetch_dataframe("""
                SELECT 
                    index_key, vst_dt, vst_tm, pat_age_gr, 
                    ktas_fstu, emtrt_rust, otrm_dt, otrm_tm
                FROM nedis_synthetic.clinical_records
                WHERE vst_dt = ?
                ORDER BY index_key
            """, [date_str])
            
            if len(clinical_records) == 0:
                self.logger.warning(f"No clinical records found for date: {date_str}")
                return {'success': False, 'reason': 'No clinical records'}
            
            total_patients = len(clinical_records)
            self.logger.info(f"Generating durations for {total_patients} patients")
            
            er_duration_updates = []
            admission_duration_updates = []
            
            for _, patient in clinical_records.iterrows():
                # ER 체류시간 재계산 (기존 otrm_dt, otrm_tm을 덮어씀)
                er_duration_update = self._generate_er_duration(patient)
                if er_duration_update:
                    er_duration_updates.append(er_duration_update)
                
                # 입원 환자의 경우 입원 기간 생성
                if patient['emtrt_rust'] in ['31', '32']:
                    admission_update = self._generate_admission_duration(patient)
                    if admission_update:
                        admission_duration_updates.append(admission_update)
            
            # 배치 업데이트
            self._batch_update_er_durations(er_duration_updates)
            if admission_duration_updates:
                self._batch_update_admission_durations(admission_duration_updates)
            
            # 결과 요약
            result = {
                'success': True,
                'date': date_str,
                'patients_processed': total_patients,
                'er_durations_updated': len(er_duration_updates),
                'admission_durations_generated': len(admission_duration_updates),
                'admission_patients': sum(1 for p in clinical_records.itertuples() 
                                        if p.emtrt_rust in ['31', '32']),
                'duration_summary': self._get_duration_summary(er_duration_updates, admission_duration_updates)
            }
            
            self.logger.info(
                f"Duration generation completed: "
                f"{len(er_duration_updates)} ER durations, {len(admission_duration_updates)} admission durations"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate durations: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_er_duration(self, patient: pd.Series) -> Optional[Dict[str, Any]]:
        """
        개별 환자의 ER 체류시간 생성 및 퇴실시간 계산
        
        Args:
            patient: 환자 임상 데이터
            
        Returns:
            ER 체류시간 업데이트 정보 or None
        """
        
        try:
            index_key = patient['index_key']
            vst_dt = patient['vst_dt']
            vst_tm = patient['vst_tm']
            pat_age_gr = patient['pat_age_gr']
            ktas_level = patient['ktas_fstu']
            emtrt_rust = patient['emtrt_rust']
            
            # KTAS별 체류시간 파라미터 조회
            duration_params = self.er_duration_params.get(ktas_level, self.er_duration_params['3'])
            
            # 연령 조정 계수 적용
            age_factor = self.age_duration_factors.get(pat_age_gr, 1.0)
            adjusted_mean = duration_params['mean'] * age_factor
            
            # 치료결과에 따른 추가 조정
            result_factor = self._get_treatment_result_duration_factor(emtrt_rust)
            adjusted_mean *= result_factor
            
            # 체류시간 생성 (로그정규분포 사용)
            duration_minutes = self._generate_lognormal_duration(
                adjusted_mean, 
                duration_params['std'], 
                duration_params['lognorm_sigma'],
                duration_params['min'],
                duration_params['max'],
                duration_params['long_stay_prob']
            )
            
            # 퇴실시간 계산
            otrm_dt, otrm_tm = self._calculate_discharge_time(vst_dt, vst_tm, duration_minutes)
            
            return {
                'index_key': index_key,
                'duration_minutes': duration_minutes,
                'otrm_dt': otrm_dt,
                'otrm_tm': otrm_tm
            }
            
        except Exception as e:
            self.logger.warning(f"ER duration generation failed for patient {patient['index_key']}: {e}")
            return None
    
    def _generate_admission_duration(self, patient: pd.Series) -> Optional[Dict[str, Any]]:
        """
        입원 환자의 입원 기간 생성
        
        Args:
            patient: 환자 임상 데이터
            
        Returns:
            입원 기간 정보 or None
        """
        
        try:
            index_key = patient['index_key']
            emtrt_rust = patient['emtrt_rust']
            pat_age_gr = patient['pat_age_gr']
            
            # 입원 시작시간 = ER 퇴실시간 + 30-90분
            er_discharge_time = datetime.strptime(
                f"{patient['otrm_dt']}{patient['otrm_tm']}", "%Y%m%d%H%M"
            )
            
            admission_delay = np.random.randint(30, 91)  # 30-90분 지연
            admission_start = er_discharge_time + timedelta(minutes=admission_delay)
            
            inpat_dt = admission_start.strftime("%Y%m%d")
            inpat_tm = admission_start.strftime("%H%M")
            
            # 입원 기간 생성
            admission_params = self.admission_duration_params.get(emtrt_rust, self.admission_duration_params['31'])
            
            # 연령별 조정 (고령자는 입원 기간 증가)
            age_factor = self.age_duration_factors.get(pat_age_gr, 1.0)
            if pat_age_gr in ['70', '80', '90']:
                age_factor *= 1.3  # 고령자 입원 기간 30% 증가
            
            adjusted_mean = admission_params['mean'] * age_factor
            
            # Zero-inflated negative binomial distribution 사용
            admission_days = self._generate_admission_duration_days(
                adjusted_mean,
                admission_params['std'],
                admission_params['zero_inflated_prob'],
                admission_params['long_stay_prob'],
                admission_params['min'],
                admission_params['max']
            )
            
            # 입원 결과 생성
            inpat_rust = self._generate_admission_outcome(emtrt_rust, admission_days, pat_age_gr)
            
            return {
                'index_key': index_key,
                'inpat_dt': inpat_dt,
                'inpat_tm': inpat_tm,
                'admission_days': admission_days,
                'inpat_rust': inpat_rust
            }
            
        except Exception as e:
            self.logger.warning(f"Admission duration generation failed for patient {patient['index_key']}: {e}")
            return None
    
    def _generate_lognormal_duration(self, mean: float, std: float, sigma: float,
                                   min_val: float, max_val: float, long_stay_prob: float) -> int:
        """
        로그정규분포 기반 체류시간 생성
        
        Args:
            mean: 목표 평균
            std: 표준편차
            sigma: 로그정규분포 형태 파라미터
            min_val: 최소값
            max_val: 최대값
            long_stay_prob: 장기 체류 확률
            
        Returns:
            체류시간 (분)
        """
        
        # 80% 정상 분포, 20% 장기 체류로 혼합
        if np.random.random() < long_stay_prob:
            # 장기 체류 (평균의 2-4배)
            long_mean = mean * np.random.uniform(2.0, 4.0)
            duration = np.random.lognormal(np.log(long_mean), sigma * 1.5)
        else:
            # 정상 체류
            duration = np.random.lognormal(np.log(mean), sigma)
        
        # 범위 제한
        duration = max(min_val, min(duration, max_val))
        
        return int(round(duration))
    
    def _generate_admission_duration_days(self, mean: float, std: float, 
                                        zero_inflated_prob: float, long_stay_prob: float,
                                        min_val: int, max_val: int) -> int:
        """
        Zero-inflated negative binomial 기반 입원 기간 생성
        
        Args:
            mean: 평균 입원 기간
            std: 표준편차
            zero_inflated_prob: 당일 퇴원 확률
            long_stay_prob: 장기 입원 확률
            min_val: 최소 입원 기간
            max_val: 최대 입원 기간
            
        Returns:
            입원 기간 (일)
        """
        
        # 당일 퇴원 (24시간 이내)
        if np.random.random() < zero_inflated_prob:
            return 1
        
        # 장기 입원 (평균의 3배 이상)
        if np.random.random() < long_stay_prob:
            long_mean = mean * np.random.uniform(3.0, 6.0)
            days = np.random.exponential(long_mean / 2)  # 지수분포로 긴 꼬리
        else:
            # 일반적인 입원 기간 - Gamma 분포 사용
            # Gamma 분포의 형태 파라미터 계산
            alpha = (mean / std) ** 2
            beta = mean / alpha
            days = np.random.gamma(alpha, beta)
        
        # 범위 제한 및 정수화
        days = max(min_val, min(int(round(days)), max_val))
        
        return days
    
    def _generate_admission_outcome(self, emtrt_rust: str, admission_days: int, pat_age_gr: str) -> str:
        """
        입원 기간과 환자 특성 기반 입원 결과 생성
        
        Args:
            emtrt_rust: 응급치료결과 (31: 병실입원, 32: 중환자실입원)
            admission_days: 입원 기간
            pat_age_gr: 연령군
            
        Returns:
            입원 결과 코드 ('1': 완쾌, '2': 호전, '3': 미호전, '4': 사망)
        """
        
        base_probs = self.admission_outcome_probs.get(emtrt_rust, self.admission_outcome_probs['31'])
        adjusted_probs = base_probs.copy()
        
        # 입원 기간에 따른 조정
        if admission_days >= 30:  # 장기 입원
            # 사망 확률 증가, 완쾌 확률 감소
            adjusted_probs['4'] *= 2.0  # 사망 확률 2배
            adjusted_probs['1'] *= 0.5  # 완쾌 확률 절반
            adjusted_probs['3'] *= 1.5  # 미호전 확률 증가
            
        elif admission_days <= 2:  # 단기 입원
            # 완쾌 확률 증가, 사망 확률 감소
            adjusted_probs['1'] *= 1.5  # 완쾌 확률 1.5배
            adjusted_probs['4'] *= 0.3  # 사망 확률 감소
        
        # 연령에 따른 조정
        if pat_age_gr in ['80', '90']:  # 고령자
            adjusted_probs['4'] *= 2.0  # 사망 확률 2배
            adjusted_probs['1'] *= 0.6  # 완쾌 확률 감소
        elif pat_age_gr in ['01', '09', '10']:  # 소아/청소년
            adjusted_probs['1'] *= 1.3  # 완쾌 확률 증가
            adjusted_probs['4'] *= 0.2  # 사망 확률 매우 낮음
        
        # 확률 정규화
        total_prob = sum(adjusted_probs.values())
        for key in adjusted_probs:
            adjusted_probs[key] /= total_prob
        
        # 확률적 선택
        outcomes = list(adjusted_probs.keys())
        probabilities = list(adjusted_probs.values())
        
        return np.random.choice(outcomes, p=probabilities)
    
    def _get_treatment_result_duration_factor(self, emtrt_rust: str) -> float:
        """
        치료결과에 따른 체류시간 조정 계수
        
        Args:
            emtrt_rust: 응급치료결과
            
        Returns:
            체류시간 조정 계수
        """
        
        duration_factors = {
            '11': 0.8,   # 귀가 - 짧은 체류
            '31': 1.2,   # 병실입원 - 긴 체류 (입원 준비)
            '32': 1.5,   # 중환자실입원 - 매우 긴 체류 (중환자 처치)
            '41': 2.0,   # 사망 - 가장 긴 체류 (응급처치 시도)
            '14': 0.5,   # 자의퇴실 - 매우 짧은 체류
            '43': 0.7    # 타병원전원 - 짧은 체류
        }
        
        return duration_factors.get(emtrt_rust, 1.0)
    
    def _calculate_discharge_time(self, vst_dt: str, vst_tm: str, duration_minutes: int) -> Tuple[str, str]:
        """
        방문시간과 체류시간으로 퇴실시간 계산
        
        Args:
            vst_dt: 방문 날짜 ('YYYYMMDD')
            vst_tm: 방문 시간 ('HHMM')
            duration_minutes: 체류시간 (분)
            
        Returns:
            (퇴실 날짜, 퇴실 시간) 튜플
        """
        
        try:
            # 방문시간을 datetime으로 변환
            arrival_time = datetime.strptime(f"{vst_dt}{vst_tm}", "%Y%m%d%H%M")
            
            # 체류시간 추가
            discharge_time = arrival_time + timedelta(minutes=duration_minutes)
            
            # 날짜와 시간 분리
            otrm_dt = discharge_time.strftime("%Y%m%d")
            otrm_tm = discharge_time.strftime("%H%M")
            
            return otrm_dt, otrm_tm
            
        except Exception as e:
            self.logger.error(f"Discharge time calculation failed: {e}")
            # 실패 시 기본값 반환 (방문일 + 2시간)
            fallback_time = datetime.strptime(f"{vst_dt}{vst_tm}", "%Y%m%d%H%M") + timedelta(hours=2)
            return fallback_time.strftime("%Y%m%d"), fallback_time.strftime("%H%M")
    
    def _batch_update_er_durations(self, duration_updates: List[Dict[str, Any]]):
        """배치 ER 체류시간 업데이트"""
        
        if not duration_updates:
            return
        
        self.logger.info(f"Batch updating ER durations for {len(duration_updates)} patients")
        
        try:
            update_sql = """
                UPDATE nedis_synthetic.clinical_records
                SET otrm_dt = ?, otrm_tm = ?
                WHERE index_key = ?
            """
            
            batch_data = []
            for update in duration_updates:
                batch_data.append((
                    update['otrm_dt'],
                    update['otrm_tm'],
                    update['index_key']
                ))
            
            # 청크 단위로 업데이트
            chunk_size = 1000
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                for row in chunk:
                    self.db.execute_query(update_sql, row)
            
            self.logger.info("ER durations batch update completed")
            
        except Exception as e:
            self.logger.error(f"ER durations batch update failed: {e}")
            raise
    
    def _batch_update_admission_durations(self, admission_updates: List[Dict[str, Any]]):
        """배치 입원 기간 업데이트"""
        
        if not admission_updates:
            return
        
        self.logger.info(f"Batch updating admission durations for {len(admission_updates)} patients")
        
        try:
            update_sql = """
                UPDATE nedis_synthetic.clinical_records
                SET inpat_dt = ?, inpat_tm = ?, inpat_rust = ?
                WHERE index_key = ?
            """
            
            batch_data = []
            for update in admission_updates:
                batch_data.append((
                    update['inpat_dt'],
                    update['inpat_tm'],
                    update['inpat_rust'],
                    update['index_key']
                ))
            
            # 청크 단위로 업데이트
            chunk_size = 1000
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                for row in chunk:
                    self.db.execute_query(update_sql, row)
            
            self.logger.info("Admission durations batch update completed")
            
        except Exception as e:
            self.logger.error(f"Admission durations batch update failed: {e}")
            raise
    
    def _get_duration_summary(self, er_updates: List[Dict[str, Any]], 
                            admission_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """체류시간 생성 결과 요약"""
        
        summary = {
            'er_duration_summary': {},
            'admission_duration_summary': {}
        }
        
        # ER 체류시간 요약
        if er_updates:
            durations = [update['duration_minutes'] for update in er_updates]
            durations_hours = [d / 60.0 for d in durations]
            
            summary['er_duration_summary'] = {
                'total_patients': len(durations),
                'mean_minutes': float(np.mean(durations)),
                'mean_hours': float(np.mean(durations_hours)),
                'std_hours': float(np.std(durations_hours)),
                'median_hours': float(np.median(durations_hours)),
                'min_hours': float(min(durations_hours)),
                'max_hours': float(max(durations_hours)),
                'percentiles': {
                    '25th': float(np.percentile(durations_hours, 25)),
                    '75th': float(np.percentile(durations_hours, 75)),
                    '90th': float(np.percentile(durations_hours, 90))
                },
                'long_stay_count': sum(1 for d in durations_hours if d >= 6),  # 6시간 이상
                'long_stay_rate': sum(1 for d in durations_hours if d >= 6) / len(durations_hours)
            }
        
        # 입원 기간 요약
        if admission_updates:
            admission_days = [update['admission_days'] for update in admission_updates]
            outcomes = [update['inpat_rust'] for update in admission_updates]
            
            summary['admission_duration_summary'] = {
                'total_patients': len(admission_days),
                'mean_days': float(np.mean(admission_days)),
                'std_days': float(np.std(admission_days)),
                'median_days': float(np.median(admission_days)),
                'min_days': int(min(admission_days)),
                'max_days': int(max(admission_days)),
                'percentiles': {
                    '25th': float(np.percentile(admission_days, 25)),
                    '75th': float(np.percentile(admission_days, 75)),
                    '90th': float(np.percentile(admission_days, 90))
                },
                'outcome_distribution': {
                    outcome: outcomes.count(outcome) for outcome in ['1', '2', '3', '4']
                },
                'short_stay_count': sum(1 for d in admission_days if d <= 3),  # 3일 이하
                'long_stay_count': sum(1 for d in admission_days if d >= 14),  # 14일 이상
                'short_stay_rate': sum(1 for d in admission_days if d <= 3) / len(admission_days),
                'long_stay_rate': sum(1 for d in admission_days if d >= 14) / len(admission_days)
            }
        
        return summary
    
    def generate_batch_durations(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        날짜 범위에 대한 배치 체류시간 생성
        
        Args:
            start_date: 시작 날짜 ('YYYYMMDD')
            end_date: 종료 날짜 ('YYYYMMDD')
            
        Returns:
            배치 체류시간 생성 결과
        """
        
        self.logger.info(f"Starting batch duration generation: {start_date} to {end_date}")
        
        try:
            # 날짜 범위 생성
            date_range = pd.date_range(
                start=pd.to_datetime(start_date, format='%Y%m%d'),
                end=pd.to_datetime(end_date, format='%Y%m%d'),
                freq='D'
            )
            
            batch_results = []
            total_patients = 0
            total_admissions = 0
            successful_dates = 0
            failed_dates = 0
            
            for date in date_range:
                date_str = date.strftime('%Y%m%d')
                
                try:
                    result = self.generate_all_durations(date_str)
                    
                    if result['success']:
                        successful_dates += 1
                        total_patients += result['patients_processed']
                        total_admissions += result['admission_durations_generated']
                    else:
                        failed_dates += 1
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Duration generation failed for date {date_str}: {e}")
                    failed_dates += 1
                    batch_results.append({
                        'success': False,
                        'date': date_str,
                        'error': str(e)
                    })
            
            # 배치 결과 요약
            summary = {
                'success': True,
                'total_dates': len(date_range),
                'successful_dates': successful_dates,
                'failed_dates': failed_dates,
                'success_rate': successful_dates / len(date_range),
                'total_patients_processed': total_patients,
                'total_admission_durations': total_admissions,
                'avg_patients_per_day': total_patients / successful_dates if successful_dates > 0 else 0,
                'avg_admissions_per_day': total_admissions / successful_dates if successful_dates > 0 else 0,
                'results': batch_results
            }
            
            self.logger.info(
                f"Batch duration generation completed: "
                f"{total_patients} patients processed, {total_admissions} admission durations generated"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Batch duration generation failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def validate_temporal_consistency(self, date_str: str) -> Dict[str, Any]:
        """
        시간적 일관성 검증
        
        Args:
            date_str: 검증할 날짜
            
        Returns:
            시간적 일관성 검증 결과
        """
        
        self.logger.info(f"Validating temporal consistency for date: {date_str}")
        
        try:
            # 시간 데이터 조회
            time_data = self.db.fetch_dataframe("""
                SELECT 
                    index_key, vst_dt, vst_tm, otrm_dt, otrm_tm,
                    inpat_dt, inpat_tm, inpat_rust, emtrt_rust, ktas_fstu
                FROM nedis_synthetic.clinical_records
                WHERE vst_dt = ?
            """, [date_str])
            
            if len(time_data) == 0:
                return {'valid': False, 'reason': 'No data'}
            
            validation_results = {
                'total_records': len(time_data),
                'violations': [],
                'summary': {}
            }
            
            for _, record in time_data.iterrows():
                # 방문시간 < 퇴실시간 검증
                vst_time = datetime.strptime(f"{record['vst_dt']}{record['vst_tm']}", "%Y%m%d%H%M")
                otrm_time = datetime.strptime(f"{record['otrm_dt']}{record['otrm_tm']}", "%Y%m%d%H%M")
                
                if vst_time >= otrm_time:
                    validation_results['violations'].append({
                        'index_key': record['index_key'],
                        'violation_type': 'discharge_before_arrival',
                        'message': f"Discharge time {record['otrm_dt']}{record['otrm_tm']} before arrival {record['vst_dt']}{record['vst_tm']}"
                    })
                
                # 체류시간 타당성 검증
                er_duration_hours = (otrm_time - vst_time).total_seconds() / 3600
                if er_duration_hours > 24:  # 24시간 초과
                    validation_results['violations'].append({
                        'index_key': record['index_key'],
                        'violation_type': 'excessive_er_stay',
                        'duration_hours': er_duration_hours,
                        'message': f"ER stay {er_duration_hours:.1f} hours exceeds 24 hours"
                    })
                
                # 입원 시간 일관성 검증
                if record['emtrt_rust'] in ['31', '32'] and pd.notna(record['inpat_dt']):
                    inpat_time = datetime.strptime(f"{record['inpat_dt']}{record['inpat_tm']}", "%Y%m%d%H%M")
                    
                    if inpat_time < otrm_time:
                        validation_results['violations'].append({
                            'index_key': record['index_key'],
                            'violation_type': 'admission_before_discharge',
                            'message': f"Admission time {record['inpat_dt']}{record['inpat_tm']} before ER discharge"
                        })
            
            # 요약 통계
            validation_results['summary'] = {
                'violation_count': len(validation_results['violations']),
                'violation_rate': len(validation_results['violations']) / len(time_data),
                'valid_records': len(time_data) - len(validation_results['violations']),
                'validity_rate': 1 - (len(validation_results['violations']) / len(time_data))
            }
            
            self.logger.info(
                f"Temporal validation completed: "
                f"{validation_results['summary']['violation_count']} violations "
                f"({validation_results['summary']['violation_rate']:.1%} rate)"
            )
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Temporal consistency validation failed: {e}")
            return {'valid': False, 'error': str(e)}