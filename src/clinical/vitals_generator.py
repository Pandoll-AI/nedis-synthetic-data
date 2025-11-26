"""
Vital Signs Generator

KTAS 등급과 연령대 기반 생체징후 생성기입니다.
의학적 정상 범위와 KTAS별 이상 패턴을 고려하여 현실적인 생체징후를 생성합니다.

생성 항목:
- vst_sbp: 수축기혈압 (mmHg)
- vst_dbp: 이완기혈압 (mmHg)  
- vst_per_pu: 맥박수 (회/분)
- vst_per_br: 호흡수 (회/분)
- vst_bdht: 체온 (°C)
- vst_oxy: 산소포화도 (%)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class VitalSignsGenerator:
    """생체징후 생성기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        생체징후 생성기 초기화
        
        Args:
            db_manager: 데이터베이터 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # KTAS별 생체징후 측정 확률
        self.ktas_measurement_probability = {
            '1': 1.00,  # 중증 - 100% 측정
            '2': 0.95,  # 응급 - 95% 측정
            '3': 0.85,  # 긴급 - 85% 측정
            '4': 0.70,  # 준응급 - 70% 측정
            '5': 0.50   # 비응급 - 50% 측정
        }
        
        # 상관관계 행렬 초기화
        self.correlation_matrix = None
        self._learn_correlation_matrix()
        
        # 연령대별 정상 범위 (평균값 기준)
        self.age_normal_ranges = {
            # 수축기혈압 (mmHg)
            'sbp': {
                '01': {'mean': 85, 'std': 10, 'min': 60, 'max': 110},   # 영아
                '09': {'mean': 95, 'std': 10, 'min': 70, 'max': 120},   # 유아
                '10': {'mean': 110, 'std': 10, 'min': 90, 'max': 135},  # 10대
                '20': {'mean': 120, 'std': 12, 'min': 100, 'max': 140}, # 20대
                '30': {'mean': 120, 'std': 12, 'min': 100, 'max': 140}, # 30대
                '40': {'mean': 125, 'std': 15, 'min': 100, 'max': 150}, # 40대
                '50': {'mean': 130, 'std': 15, 'min': 105, 'max': 160}, # 50대
                '60': {'mean': 135, 'std': 18, 'min': 105, 'max': 170}, # 60대
                '70': {'mean': 140, 'std': 20, 'min': 110, 'max': 180}, # 70대
                '80': {'mean': 145, 'std': 20, 'min': 110, 'max': 190}, # 80대
                '90': {'mean': 150, 'std': 25, 'min': 110, 'max': 200}  # 90대
            },
            
            # 이완기혈압 (mmHg) - 수축기의 약 60-70%
            'dbp': {
                '01': {'mean': 50, 'std': 8, 'min': 35, 'max': 70},
                '09': {'mean': 55, 'std': 8, 'min': 40, 'max': 75},
                '10': {'mean': 65, 'std': 8, 'min': 50, 'max': 85},
                '20': {'mean': 75, 'std': 10, 'min': 60, 'max': 90},
                '30': {'mean': 75, 'std': 10, 'min': 60, 'max': 90},
                '40': {'mean': 80, 'std': 12, 'min': 60, 'max': 95},
                '50': {'mean': 82, 'std': 12, 'min': 65, 'max': 100},
                '60': {'mean': 85, 'std': 15, 'min': 65, 'max': 105},
                '70': {'mean': 85, 'std': 15, 'min': 70, 'max': 110},
                '80': {'mean': 85, 'std': 15, 'min': 70, 'max': 115},
                '90': {'mean': 85, 'std': 18, 'min': 70, 'max': 120}
            },
            
            # 맥박수 (회/분)
            'pulse': {
                '01': {'mean': 130, 'std': 20, 'min': 100, 'max': 180},  # 영아 빠름
                '09': {'mean': 110, 'std': 15, 'min': 80, 'max': 150},   # 유아
                '10': {'mean': 85, 'std': 15, 'min': 60, 'max': 120},    # 10대
                '20': {'mean': 75, 'std': 12, 'min': 55, 'max': 110},    # 20대
                '30': {'mean': 75, 'std': 12, 'min': 55, 'max': 110},    # 30대
                '40': {'mean': 78, 'std': 12, 'min': 55, 'max': 110},    # 40대
                '50': {'mean': 80, 'std': 12, 'min': 55, 'max': 115},    # 50대
                '60': {'mean': 82, 'std': 15, 'min': 60, 'max': 120},    # 60대
                '70': {'mean': 85, 'std': 15, 'min': 60, 'max': 125},    # 70대
                '80': {'mean': 88, 'std': 18, 'min': 65, 'max': 130},    # 80대
                '90': {'mean': 90, 'std': 20, 'min': 65, 'max': 135}     # 90대
            },
            
            # 호흡수 (회/분)
            'respiration': {
                '01': {'mean': 35, 'std': 8, 'min': 25, 'max': 50},     # 영아 빠름
                '09': {'mean': 25, 'std': 5, 'min': 18, 'max': 35},     # 유아
                '10': {'mean': 18, 'std': 3, 'min': 14, 'max': 25},     # 10대
                '20': {'mean': 16, 'std': 3, 'min': 12, 'max': 22},     # 20대
                '30': {'mean': 16, 'std': 3, 'min': 12, 'max': 22},     # 30대
                '40': {'mean': 17, 'std': 3, 'min': 12, 'max': 24},     # 40대
                '50': {'mean': 17, 'std': 3, 'min': 13, 'max': 24},     # 50대
                '60': {'mean': 18, 'std': 4, 'min': 13, 'max': 26},     # 60대
                '70': {'mean': 19, 'std': 4, 'min': 14, 'max': 28},     # 70대
                '80': {'mean': 20, 'std': 5, 'min': 14, 'max': 30},     # 80대
                '90': {'mean': 22, 'std': 5, 'min': 15, 'max': 32}      # 90대
            },
            
            # 체온 (°C)
            'temperature': {
                'normal': {'mean': 36.5, 'std': 0.4, 'min': 35.5, 'max': 37.5}
            },
            
            # 산소포화도 (%)
            'oxygen_saturation': {
                'normal': {'mean': 98, 'std': 1.5, 'min': 95, 'max': 100}
            }
        }
        
        # KTAS별 이상 생체징후 확률 및 패턴
        self.ktas_abnormal_patterns = {
            '1': {  # 중증 - 높은 이상 확률
                'hypotension_prob': 0.40,    # 저혈압
                'hypertension_prob': 0.25,   # 고혈압
                'tachycardia_prob': 0.45,    # 빈맥
                'bradycardia_prob': 0.15,    # 서맥
                'tachypnea_prob': 0.50,      # 빈호흡
                'fever_prob': 0.35,          # 발열
                'hypothermia_prob': 0.15,    # 체온저하
                'hypoxia_prob': 0.40         # 저산소증
            },
            '2': {  # 응급
                'hypotension_prob': 0.20,
                'hypertension_prob': 0.30,
                'tachycardia_prob': 0.35,
                'bradycardia_prob': 0.10,
                'tachypnea_prob': 0.35,
                'fever_prob': 0.25,
                'hypothermia_prob': 0.05,
                'hypoxia_prob': 0.20
            },
            '3': {  # 긴급
                'hypotension_prob': 0.10,
                'hypertension_prob': 0.25,
                'tachycardia_prob': 0.25,
                'bradycardia_prob': 0.08,
                'tachypnea_prob': 0.20,
                'fever_prob': 0.20,
                'hypothermia_prob': 0.02,
                'hypoxia_prob': 0.10
            },
            '4': {  # 준응급
                'hypotension_prob': 0.05,
                'hypertension_prob': 0.15,
                'tachycardia_prob': 0.15,
                'bradycardia_prob': 0.05,
                'tachypnea_prob': 0.10,
                'fever_prob': 0.15,
                'hypothermia_prob': 0.01,
                'hypoxia_prob': 0.05
            },
            '5': {  # 비응급 - 대부분 정상
                'hypotension_prob': 0.02,
                'hypertension_prob': 0.08,
                'tachycardia_prob': 0.08,
                'bradycardia_prob': 0.03,
                'tachypnea_prob': 0.05,
                'fever_prob': 0.10,
                'hypothermia_prob': 0.01,
                'hypoxia_prob': 0.02
            }
        }
    
    def _learn_correlation_matrix(self):
        """실제 데이터에서 생체징후 간 상관관계 학습"""
        self.logger.info("Learning vital signs correlation matrix from real data")
        
        try:
            # 원본 데이터 테이블 확인
            source_table = self.config.get('original.source_table', 'nedis_original.nedis2017')
            
            # 생체징후 데이터 조회 (유효한 값만)
            query = f"""
                SELECT 
                    vst_sbp, vst_dbp, vst_per_pu, vst_per_br, vst_bdht, vst_oxy
                FROM {source_table}
                WHERE vst_sbp > 0 AND vst_dbp > 0 AND vst_per_pu > 0 
                  AND vst_per_br > 0 AND vst_bdht > 0 AND vst_oxy > 0
                LIMIT 50000
            """
            
            df = self.db.fetch_dataframe(query)
            
            if len(df) > 100:
                # 상관관계 행렬 계산
                self.correlation_matrix = df.corr()
                self.logger.info("Successfully learned correlation matrix")
                self.logger.debug(f"Correlation matrix:\n{self.correlation_matrix}")
            else:
                self.logger.warning("Not enough data to learn correlations, using default")
                self._set_default_correlations()
                
        except Exception as e:
            self.logger.warning(f"Failed to learn correlation matrix: {e}")
            self._set_default_correlations()
            
    def _set_default_correlations(self):
        """기본 상관관계 설정 (의학적 지식 기반)"""
        # 기본적으로 독립적(Identity)이지만, SBP-DBP, PR-BT 등은 상관관계 설정
        cols = ['vst_sbp', 'vst_dbp', 'vst_per_pu', 'vst_per_br', 'vst_bdht', 'vst_oxy']
        n = len(cols)
        corr = np.eye(n)
        df_corr = pd.DataFrame(corr, index=cols, columns=cols)
        
        # SBP - DBP: 강한 양의 상관관계
        df_corr.loc['vst_sbp', 'vst_dbp'] = 0.7
        df_corr.loc['vst_dbp', 'vst_sbp'] = 0.7
        
        # PR - BT: 양의 상관관계 (발열 시 맥박 증가)
        df_corr.loc['vst_per_pu', 'vst_bdht'] = 0.4
        df_corr.loc['vst_bdht', 'vst_per_pu'] = 0.4
        
        # PR - SBP: 약한 양의 상관관계
        df_corr.loc['vst_per_pu', 'vst_sbp'] = 0.2
        df_corr.loc['vst_sbp', 'vst_per_pu'] = 0.2
        
        self.correlation_matrix = df_corr

    def generate_all_vital_signs(self, date_str: str) -> Dict[str, Any]:
        """
        특정 날짜의 모든 환자 생체징후 생성
        
        Args:
            date_str: 대상 날짜 ('YYYYMMDD' 형식)
            
        Returns:
            생체징후 생성 결과 딕셔너리
        """
        
        self.logger.info(f"Generating vital signs for date: {date_str}")
        
        try:
            # 해당 날짜 임상 레코드 조회
            clinical_records = self.db.fetch_dataframe("""
                SELECT 
                    index_key, pat_age_gr, pat_sex, ktas_fstu
                FROM nedis_synthetic.clinical_records
                WHERE vst_dt = ?
                ORDER BY index_key
            """, [date_str])
            
            if len(clinical_records) == 0:
                self.logger.warning(f"No clinical records found for date: {date_str}")
                return {'success': False, 'reason': 'No clinical records'}
            
            total_patients = len(clinical_records)
            self.logger.info(f"Generating vital signs for {total_patients} patients")
            
            vital_updates = []
            
            for _, patient in clinical_records.iterrows():
                vitals = self._generate_patient_vitals(patient)
                if vitals:
                    vital_updates.append(vitals)
            
            # 배치 업데이트
            self._batch_update_vitals(vital_updates)
            
            # 결과 요약
            result = {
                'success': True,
                'date': date_str,
                'patients_processed': total_patients,
                'vitals_generated': len(vital_updates),
                'vitals_summary': self._get_vitals_summary(vital_updates)
            }
            
            self.logger.info(f"Vital signs generation completed: {len(vital_updates)} patients updated")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate vital signs: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_patient_vitals(self, patient: pd.Series) -> Optional[Dict[str, Any]]:
        """
        개별 환자의 생체징후 생성
        
        Args:
            patient: 환자 임상 데이터
            
        Returns:
            생성된 생체징후 딕셔너리 or None
        """
        
        index_key = patient['index_key']
        pat_age_gr = patient['pat_age_gr']
        pat_sex = patient['pat_sex']
        ktas_level = patient['ktas_fstu']
        
        try:
            # 1. 상관관계가 고려된 전체 생체징후 생성
            correlated_vitals = self._generate_correlated_vitals(pat_age_gr, ktas_level)
            
            if not correlated_vitals:
                # 실패 시 기존 방식(독립 생성)으로 폴백하거나 None 반환
                # 여기서는 간단히 None 반환 (로그는 _generate_correlated_vitals에서 남김)
                return None

            # 2. KTAS별 측정 확률 체크 및 적용
            measurement_prob = self.ktas_measurement_probability.get(ktas_level, 0.7)
            
            vitals = {
                'index_key': index_key,
                'vst_sbp': -1,
                'vst_dbp': -1,
                'vst_per_pu': -1,
                'vst_per_br': -1,
                'vst_bdht': -1.0,
                'vst_oxy': -1
            }
            
            # 혈압
            if np.random.random() < measurement_prob:
                vitals['vst_sbp'] = int(round(correlated_vitals['vst_sbp']))
                vitals['vst_dbp'] = int(round(correlated_vitals['vst_dbp']))
                
            # 맥박
            if np.random.random() < measurement_prob:
                vitals['vst_per_pu'] = int(round(correlated_vitals['vst_per_pu']))
                
            # 호흡
            if np.random.random() < measurement_prob * 0.9:
                vitals['vst_per_br'] = int(round(correlated_vitals['vst_per_br']))
                
            # 체온
            if np.random.random() < measurement_prob * 0.8:
                vitals['vst_bdht'] = round(correlated_vitals['vst_bdht'], 1)
                
            # 산소포화도
            if np.random.random() < measurement_prob * 0.7:
                vitals['vst_oxy'] = int(round(correlated_vitals['vst_oxy']))
            
            return vitals
            
        except Exception as e:
            self.logger.warning(f"Vital signs generation failed for patient {index_key}: {e}")
            return None
            
    def _generate_correlated_vitals(self, pat_age_gr: str, ktas_level: str) -> Dict[str, float]:
        """
        상관관계가 고려된 생체징후 생성 (Gaussian Copula)
        1. 다변량 정규분포에서 샘플링 (Z-score)
        2. 각 변수의 주변분포(Marginal Distribution)로 변환
        """
        try:
            # 1. 다변량 정규분포 샘플링
            if self.correlation_matrix is None:
                self._set_default_correlations()
                
            mean = np.zeros(len(self.correlation_matrix))
            cov = self.correlation_matrix.values
            
            # Z-score 샘플링
            z_scores = np.random.multivariate_normal(mean, cov)
            
            # 2. 각 변수별 값 변환 (Z-score -> Percentile -> Value)
            vitals = {}
            cols = self.correlation_matrix.columns
            
            for i, col in enumerate(cols):
                z = z_scores[i]
                percentile = stats.norm.cdf(z)
                
                # 각 변수의 주변분포 파라미터 가져오기
                if col == 'vst_sbp':
                    params = self.age_normal_ranges['sbp'].get(pat_age_gr, self.age_normal_ranges['sbp']['30'])
                    val = stats.norm.ppf(percentile, loc=params['mean'], scale=params['std'])
                elif col == 'vst_dbp':
                    # DBP는 SBP와 강한 상관관계가 이미 반영됨
                    params = self.age_normal_ranges['dbp'].get(pat_age_gr, self.age_normal_ranges['dbp']['30'])
                    val = stats.norm.ppf(percentile, loc=params['mean'], scale=params['std'])
                elif col == 'vst_per_pu':
                    params = self.age_normal_ranges['pulse'].get(pat_age_gr, self.age_normal_ranges['pulse']['30'])
                    val = stats.norm.ppf(percentile, loc=params['mean'], scale=params['std'])
                elif col == 'vst_per_br':
                    params = self.age_normal_ranges['respiration'].get(pat_age_gr, self.age_normal_ranges['respiration']['30'])
                    val = stats.norm.ppf(percentile, loc=params['mean'], scale=params['std'])
                elif col == 'vst_bdht':
                    params = self.age_normal_ranges['temperature']['normal']
                    val = stats.norm.ppf(percentile, loc=params['mean'], scale=params['std'])
                elif col == 'vst_oxy':
                    params = self.age_normal_ranges['oxygen_saturation']['normal']
                    # SpO2는 상한이 있으므로 절단된 정규분포 느낌으로 처리하거나, 단순 매핑 후 클리핑
                    val = stats.norm.ppf(percentile, loc=params['mean'], scale=params['std'])
                else:
                    val = 0
                
                vitals[col] = val
            
            # 3. KTAS별 이상 패턴 적용 (상관관계 유지하면서 Shift)
            # 중증 환자의 경우 전체적으로 분포를 이동시킴
            abnormal_patterns = self.ktas_abnormal_patterns.get(ktas_level, self.ktas_abnormal_patterns['3'])
            
            # 쇼크 (저혈압 + 빈맥)
            if np.random.random() < abnormal_patterns.get('hypotension_prob', 0) * 0.5: # 동시 발생 확률
                vitals['vst_sbp'] -= 30
                vitals['vst_dbp'] -= 20
                vitals['vst_per_pu'] += 30
            
            # 발열 (고체온 + 빈맥 + 빈호흡)
            if np.random.random() < abnormal_patterns.get('fever_prob', 0):
                vitals['vst_bdht'] += 2.0
                vitals['vst_per_pu'] += 20
                vitals['vst_per_br'] += 4
                
            return vitals
            
        except Exception as e:
            self.logger.warning(f"Correlated vitals generation failed: {e}")
            return None
    
    def _generate_blood_pressure(self, pat_age_gr: str, ktas_level: str) -> Dict[str, int]:
        """혈압 생성 (수축기/이완기)"""
        
        sbp_params = self.age_normal_ranges['sbp'].get(pat_age_gr, self.age_normal_ranges['sbp']['30'])
        dbp_params = self.age_normal_ranges['dbp'].get(pat_age_gr, self.age_normal_ranges['dbp']['30'])
        
        abnormal_patterns = self.ktas_abnormal_patterns.get(ktas_level, self.ktas_abnormal_patterns['3'])
        
        # 정상 혈압 생성
        sbp = np.random.normal(sbp_params['mean'], sbp_params['std'])
        
        # 이상 패턴 적용
        if np.random.random() < abnormal_patterns['hypotension_prob']:
            # 저혈압 (쇼크 상태)
            sbp = np.random.normal(75, 15)  # 평균 75mmHg
            sbp = max(40, min(sbp, 90))  # 40-90 범위
        elif np.random.random() < abnormal_patterns['hypertension_prob']:
            # 고혈압
            sbp = np.random.normal(180, 25)  # 평균 180mmHg
            sbp = max(160, min(sbp, 250))  # 160-250 범위
        else:
            # 정상 범위로 제한
            sbp = max(sbp_params['min'], min(sbp, sbp_params['max']))
        
        # 이완기혈압 계산 (수축기의 60-75% + 변동)
        dbp_ratio = np.random.uniform(0.60, 0.75)
        dbp = sbp * dbp_ratio + np.random.normal(0, 5)
        
        # 이완기혈압 범위 제한
        dbp = max(30, min(dbp, sbp - 20))  # 수축기보다 최소 20 낮게
        dbp = max(dbp_params['min'], min(dbp, dbp_params['max']))
        
        return {
            'vst_sbp': int(round(sbp)),
            'vst_dbp': int(round(dbp))
        }
    
    def _generate_pulse_rate(self, pat_age_gr: str, ktas_level: str) -> int:
        """맥박수 생성"""
        
        pulse_params = self.age_normal_ranges['pulse'].get(pat_age_gr, self.age_normal_ranges['pulse']['30'])
        abnormal_patterns = self.ktas_abnormal_patterns.get(ktas_level, self.ktas_abnormal_patterns['3'])
        
        # 정상 맥박 생성
        pulse = np.random.normal(pulse_params['mean'], pulse_params['std'])
        
        # 이상 패턴 적용
        if np.random.random() < abnormal_patterns['tachycardia_prob']:
            # 빈맥
            pulse = np.random.normal(120, 20)  # 평균 120bpm
            pulse = max(100, min(pulse, 200))
        elif np.random.random() < abnormal_patterns['bradycardia_prob']:
            # 서맥
            pulse = np.random.normal(45, 10)   # 평균 45bpm
            pulse = max(30, min(pulse, 60))
        else:
            # 정상 범위로 제한
            pulse = max(pulse_params['min'], min(pulse, pulse_params['max']))
        
        return int(round(pulse))
    
    def _generate_respiration_rate(self, pat_age_gr: str, ktas_level: str) -> int:
        """호흡수 생성"""
        
        resp_params = self.age_normal_ranges['respiration'].get(pat_age_gr, self.age_normal_ranges['respiration']['30'])
        abnormal_patterns = self.ktas_abnormal_patterns.get(ktas_level, self.ktas_abnormal_patterns['3'])
        
        # 정상 호흡수 생성
        respiration = np.random.normal(resp_params['mean'], resp_params['std'])
        
        # 이상 패턴 적용
        if np.random.random() < abnormal_patterns['tachypnea_prob']:
            # 빈호흡
            age_factor = 1.5 if pat_age_gr in ['01', '09'] else 1.0  # 소아는 더 빠름
            respiration = np.random.normal(resp_params['mean'] * 1.8 * age_factor, resp_params['std'])
            respiration = max(resp_params['mean'] * 1.3, min(respiration, 50))
        else:
            # 정상 범위로 제한
            respiration = max(resp_params['min'], min(respiration, resp_params['max']))
        
        return int(round(respiration))
    
    def _generate_temperature(self, ktas_level: str) -> float:
        """체온 생성"""
        
        temp_params = self.age_normal_ranges['temperature']['normal']
        abnormal_patterns = self.ktas_abnormal_patterns.get(ktas_level, self.ktas_abnormal_patterns['3'])
        
        # 정상 체온 생성
        temperature = np.random.normal(temp_params['mean'], temp_params['std'])
        
        # 이상 패턴 적용
        if np.random.random() < abnormal_patterns['fever_prob']:
            # 발열
            fever_severity = np.random.choice(['mild', 'moderate', 'high'], p=[0.5, 0.3, 0.2])
            if fever_severity == 'mild':
                temperature = np.random.normal(38.0, 0.3)  # 미열
            elif fever_severity == 'moderate':
                temperature = np.random.normal(39.0, 0.4)  # 중등열
            else:
                temperature = np.random.normal(40.0, 0.5)  # 고열
                
            temperature = max(37.5, min(temperature, 42.0))
            
        elif np.random.random() < abnormal_patterns['hypothermia_prob']:
            # 체온저하 (쇼크 등)
            temperature = np.random.normal(34.5, 0.8)
            temperature = max(32.0, min(temperature, 36.0))
        else:
            # 정상 범위로 제한
            temperature = max(temp_params['min'], min(temperature, temp_params['max']))
        
        return round(temperature, 1)
    
    def _generate_oxygen_saturation(self, ktas_level: str) -> int:
        """산소포화도 생성"""
        
        spo2_params = self.age_normal_ranges['oxygen_saturation']['normal']
        abnormal_patterns = self.ktas_abnormal_patterns.get(ktas_level, self.ktas_abnormal_patterns['3'])
        
        # 정상 산소포화도 생성
        spo2 = np.random.normal(spo2_params['mean'], spo2_params['std'])
        
        # 이상 패턴 적용
        if np.random.random() < abnormal_patterns['hypoxia_prob']:
            # 저산소증
            hypoxia_severity = np.random.choice(['mild', 'moderate', 'severe'], p=[0.4, 0.4, 0.2])
            if hypoxia_severity == 'mild':
                spo2 = np.random.normal(92, 2)    # 경도 저산소
            elif hypoxia_severity == 'moderate':
                spo2 = np.random.normal(87, 3)    # 중등도 저산소
            else:
                spo2 = np.random.normal(80, 5)    # 중증 저산소
                
            spo2 = max(70, min(spo2, 94))
        else:
            # 정상 범위로 제한
            spo2 = max(spo2_params['min'], min(spo2, spo2_params['max']))
        
        return int(round(spo2))
    
    def _batch_update_vitals(self, vital_updates: List[Dict[str, Any]]):
        """배치 생체징후 업데이트"""
        
        if not vital_updates:
            return
        
        self.logger.info(f"Batch updating vital signs for {len(vital_updates)} patients")
        
        try:
            update_sql = """
                UPDATE nedis_synthetic.clinical_records
                SET vst_sbp = ?, vst_dbp = ?, vst_per_pu = ?, 
                    vst_per_br = ?, vst_bdht = ?, vst_oxy = ?
                WHERE index_key = ?
            """
            
            batch_data = []
            for vitals in vital_updates:
                batch_data.append((
                    vitals['vst_sbp'], vitals['vst_dbp'], vitals['vst_per_pu'],
                    vitals['vst_per_br'], vitals['vst_bdht'], vitals['vst_oxy'],
                    vitals['index_key']
                ))
            
            # 청크 단위로 업데이트
            chunk_size = 1000
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                for row in chunk:
                    self.db.execute_query(update_sql, row)
            
            self.logger.info("Vital signs batch update completed")
            
        except Exception as e:
            self.logger.error(f"Vital signs batch update failed: {e}")
            raise
    
    def _get_vitals_summary(self, vital_updates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """생체징후 생성 요약 통계"""
        
        if not vital_updates:
            return {}
        
        vitals_df = pd.DataFrame(vital_updates)
        
        # -1 값 (측정 안함)을 제외하고 통계 계산
        summary = {}
        
        for vital in ['vst_sbp', 'vst_dbp', 'vst_per_pu', 'vst_per_br', 'vst_oxy']:
            measured_values = vitals_df[vitals_df[vital] != -1][vital]
            
            if len(measured_values) > 0:
                summary[vital] = {
                    'measured_count': len(measured_values),
                    'measurement_rate': len(measured_values) / len(vitals_df),
                    'mean': float(measured_values.mean()),
                    'std': float(measured_values.std()),
                    'min': int(measured_values.min()),
                    'max': int(measured_values.max()),
                    'median': float(measured_values.median())
                }
            else:
                summary[vital] = {
                    'measured_count': 0,
                    'measurement_rate': 0.0
                }
        
        # 체온은 float 처리
        if 'vst_bdht' in vitals_df.columns:
            temp_measured = vitals_df[vitals_df['vst_bdht'] != -1.0]['vst_bdht']
            
            if len(temp_measured) > 0:
                summary['vst_bdht'] = {
                    'measured_count': len(temp_measured),
                    'measurement_rate': len(temp_measured) / len(vitals_df),
                    'mean': float(temp_measured.mean()),
                    'std': float(temp_measured.std()),
                    'min': float(temp_measured.min()),
                    'max': float(temp_measured.max()),
                    'median': float(temp_measured.median())
                }
            else:
                summary['vst_bdht'] = {
                    'measured_count': 0,
                    'measurement_rate': 0.0
                }
        
        # 이상값 분석
        abnormal_analysis = self._analyze_abnormal_vitals(vitals_df)
        summary['abnormal_patterns'] = abnormal_analysis
        
        return summary
    
    def _analyze_abnormal_vitals(self, vitals_df: pd.DataFrame) -> Dict[str, Any]:
        """이상 생체징후 패턴 분석"""
        
        abnormal_counts = {}
        total_measured = len(vitals_df)
        
        # 혈압 이상
        hypertension = vitals_df[vitals_df['vst_sbp'] >= 160]['vst_sbp'].count()
        hypotension = vitals_df[vitals_df['vst_sbp'] <= 90]['vst_sbp'].count()
        
        # 맥박 이상
        tachycardia = vitals_df[vitals_df['vst_per_pu'] >= 100]['vst_per_pu'].count()
        bradycardia = vitals_df[vitals_df['vst_per_pu'] <= 60]['vst_per_pu'].count()
        
        # 호흡 이상
        tachypnea = vitals_df[vitals_df['vst_per_br'] >= 24]['vst_per_br'].count()
        
        # 체온 이상
        fever = vitals_df[vitals_df['vst_bdht'] >= 37.5]['vst_bdht'].count()
        hypothermia = vitals_df[vitals_df['vst_bdht'] <= 35.5]['vst_bdht'].count()
        
        # 산소포화도 이상
        hypoxia = vitals_df[vitals_df['vst_oxy'] <= 94]['vst_oxy'].count()
        
        abnormal_analysis = {
            'hypertension': {'count': int(hypertension), 'rate': float(hypertension / total_measured)},
            'hypotension': {'count': int(hypotension), 'rate': float(hypotension / total_measured)},
            'tachycardia': {'count': int(tachycardia), 'rate': float(tachycardia / total_measured)},
            'bradycardia': {'count': int(bradycardia), 'rate': float(bradycardia / total_measured)},
            'tachypnea': {'count': int(tachypnea), 'rate': float(tachypnea / total_measured)},
            'fever': {'count': int(fever), 'rate': float(fever / total_measured)},
            'hypothermia': {'count': int(hypothermia), 'rate': float(hypothermia / total_measured)},
            'hypoxia': {'count': int(hypoxia), 'rate': float(hypoxia / total_measured)}
        }
        
        return abnormal_analysis
    
    def generate_batch_vital_signs(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        날짜 범위에 대한 배치 생체징후 생성
        
        Args:
            start_date: 시작 날짜 ('YYYYMMDD')
            end_date: 종료 날짜 ('YYYYMMDD')
            
        Returns:
            배치 생체징후 생성 결과
        """
        
        self.logger.info(f"Starting batch vital signs generation: {start_date} to {end_date}")
        
        try:
            # 날짜 범위 생성
            date_range = pd.date_range(
                start=pd.to_datetime(start_date, format='%Y%m%d'),
                end=pd.to_datetime(end_date, format='%Y%m%d'),
                freq='D'
            )
            
            batch_results = []
            total_patients = 0
            successful_dates = 0
            failed_dates = 0
            
            for date in date_range:
                date_str = date.strftime('%Y%m%d')
                
                try:
                    result = self.generate_all_vital_signs(date_str)
                    
                    if result['success']:
                        successful_dates += 1
                        total_patients += result['patients_processed']
                    else:
                        failed_dates += 1
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Vital signs generation failed for date {date_str}: {e}")
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
                'avg_patients_per_day': total_patients / successful_dates if successful_dates > 0 else 0,
                'results': batch_results
            }
            
            self.logger.info(
                f"Batch vital signs generation completed: "
                f"{total_patients} patients processed, {successful_dates}/{len(date_range)} successful days"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Batch vital signs generation failed: {e}")
            return {'success': False, 'error': str(e)}