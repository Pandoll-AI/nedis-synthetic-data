"""
Clinical DAG (Directed Acyclic Graph) Generator

DAG 기반 순차 임상 속성 생성기입니다.
의료적 인과관계를 고려하여 임상 속성들을 올바른 순서로 생성합니다.

생성 순서:
1. ptmiinmn (내원수단) - 독립적
2. ptmimnsy (주증상) - 내원수단과 연관
3. ptmikts1 (KTAS 등급) - 주증상, 연령, 내원수단에 조건적
4. ptmidept (주요치료과) - KTAS와 주증상에 조건적
5. ptmiemrt (응급치료결과) - KTAS에 강하게 의존
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime, timedelta
import random

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class ClinicalDAGGenerator:
    """DAG 기반 임상 속성 생성기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        DAG 임상 생성기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # DAG 생성 순서 정의 (의료적 인과관계 순서)
        self.dag_order = [
            'ptmiinmn',      # 내원수단
            'ptmimnsy',      # 주증상
            'ptmikts1',      # KTAS 등급
            'ptmidept',      # 주요치료과
            'ptmiemrt'       # 응급치료결과
        ]
        
        # 시간대별 도착 패턴 (시간별 가중치)
        self.hourly_weights = {
            0: 0.015, 1: 0.010, 2: 0.008, 3: 0.007, 4: 0.006, 5: 0.008,
            6: 0.015, 7: 0.025, 8: 0.040, 9: 0.055, 10: 0.065, 11: 0.070,
            12: 0.075, 13: 0.080, 14: 0.085, 15: 0.090, 16: 0.085, 17: 0.080,
            18: 0.075, 19: 0.065, 20: 0.055, 21: 0.045, 22: 0.035, 23: 0.025
        }
        
        # KTAS별 응급치료결과 확률 (의학적 가이드라인 기반)
        self.ktas_outcome_probs = {
            '1': {'11': 0.05, '31': 0.40, '32': 0.35, '41': 0.15, '14': 0.03, '43': 0.02},  # 중증
            '2': {'11': 0.25, '31': 0.50, '32': 0.20, '41': 0.03, '14': 0.02, '43': 0.00},  # 응급
            '3': {'11': 0.60, '31': 0.30, '32': 0.08, '41': 0.01, '14': 0.01, '43': 0.00},  # 긴급
            '4': {'11': 0.80, '31': 0.15, '32': 0.03, '41': 0.005, '14': 0.015, '43': 0.00}, # 준응급
            '5': {'11': 0.95, '31': 0.03, '32': 0.01, '41': 0.001, '14': 0.009, '43': 0.00}  # 비응급
        }
        
    def initialize_clinical_records_table(self):
        """임상 레코드 테이블 초기화 (한 번만 실행)"""
        
        self.logger.info("Initializing clinical records table")
        
        try:
            # 임상 레코드 테이블 생성 (존재하지 않는 경우만)
            create_table_sql = """
                CREATE TABLE IF NOT EXISTS nedis_synthetic.clinical_records (
                    index_key VARCHAR PRIMARY KEY,
                    ptmiemcd VARCHAR NOT NULL,
                    ptmiidno VARCHAR NOT NULL,
                    ptmiindt VARCHAR NOT NULL,
                    ptmiintm VARCHAR NOT NULL,
                    ptmibrtd VARCHAR NOT NULL,
                    ptmisexx VARCHAR NOT NULL,
                    ptmizipc VARCHAR NOT NULL,

                    -- 내원 관련
                    ptmiinmn VARCHAR,        -- 내원수단

                    -- 임상 중증도
                    ptmikts1 VARCHAR,       -- KTAS 등급
                    ptmikpr1 VARCHAR,       -- KTAS 프로토콜 코드

                    -- 증상 및 치료
                    ptmimnsy VARCHAR,        -- 주증상
                    ptmidept VARCHAR,        -- 주요치료과

                    -- 치료 결과
                    ptmiemrt VARCHAR,        -- 응급치료결과

                    -- 퇴실 시간
                    ptmiotdt VARCHAR,        -- 퇴실일자
                    ptmiottm VARCHAR,        -- 퇴실시간

                    -- 생체징후
                    ptmihibp INTEGER DEFAULT -1,      -- 수축기혈압
                    ptmilobp INTEGER DEFAULT -1,      -- 이완기혈압
                    ptmipuls INTEGER DEFAULT -1,      -- 맥박수
                    ptmibrth INTEGER DEFAULT -1,      -- 호흡수
                    ptmibdht DOUBLE DEFAULT -1, -- 체온
                    ptmivoxs INTEGER DEFAULT -1,      -- 산소포화도

                    -- 입원 관련 (해당 시에만)
                    ptmihsdt VARCHAR,        -- 입원일자
                    ptmihstm VARCHAR,        -- 입원시간
                    ptmidcrt VARCHAR,        -- 입원결과

                    -- 메타데이터
                    generation_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    generation_method VARCHAR DEFAULT 'dag_sequential'
                )
            """
            
            self.db.execute_query(create_table_sql)
            self.logger.info("Clinical records table created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize clinical records table: {e}")
            raise
    
    def generate_clinical_attributes(self, date_str: str) -> Dict[str, Any]:
        """
        특정 날짜의 병원 할당 기반으로 임상 속성 생성
        
        Args:
            date_str: 생성할 날짜 ('YYYYMMDD' 형식)
            
        Returns:
            생성 결과 딕셔너리
        """
        
        self.logger.info(f"Generating clinical attributes for date: {date_str}")
        
        try:
            # 해당 날짜의 병원 할당 데이터 로드
            allocation_data = self.db.fetch_dataframe("""
                SELECT
                    ptmiindt, ptmiemcd, ptmizipc, ptmibrtd, ptmisexx, allocated_count
                FROM nedis_synthetic.hospital_allocations
                WHERE ptmiindt = ?
                ORDER BY ptmiemcd, ptmizipc, ptmibrtd, ptmisexx
            """, [date_str])
            
            if len(allocation_data) == 0:
                self.logger.warning(f"No allocation data found for date: {date_str}")
                return {'success': False, 'reason': 'No allocation data'}
            
            # 해당 날짜의 기존 데이터 삭제 (재실행 시 중복 방지)
            self.db.execute_query("""
                DELETE FROM nedis_synthetic.clinical_records
                WHERE ptmiindt = ?
            """, [date_str])
            
            total_records_to_generate = allocation_data['allocated_count'].sum()
            self.logger.info(f"Generating {total_records_to_generate} clinical records from {len(allocation_data)} allocation groups")
            
            generated_records = []
            patient_counter = 1
            
            # 각 할당 그룹별로 레코드 생성
            for _, allocation in allocation_data.iterrows():
                group_records = self._generate_group_records(
                    allocation, patient_counter
                )
                generated_records.extend(group_records)
                patient_counter += len(group_records)
            
            # 배치 삽입
            self._batch_insert_records(generated_records)
            
            # 생성 결과 요약
            result = {
                'success': True,
                'date': date_str,
                'records_generated': len(generated_records),
                'allocation_groups_processed': len(allocation_data),
                'generation_summary': self._get_generation_summary(generated_records)
            }
            
            self.logger.info(f"Clinical attributes generation completed: {len(generated_records)} records")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate clinical attributes: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_group_records(self, allocation_row: pd.Series, start_patient_id: int) -> List[Dict[str, Any]]:
        """
        특정 할당 그룹(동일 병원-인구그룹)의 임상 레코드들 생성
        
        Args:
            allocation_row: 할당 행 데이터
            start_patient_id: 시작 환자 ID
            
        Returns:
            생성된 레코드 리스트
        """
        
        ptmiindt = allocation_row['ptmiindt']
        ptmiemcd = allocation_row['ptmiemcd']
        ptmizipc = allocation_row['ptmizipc']
        ptmibrtd = allocation_row['ptmibrtd']
        ptmisexx = allocation_row['ptmisexx']
        count = int(allocation_row['allocated_count'])
        
        records = []
        
        
        for i in range(count):
            patient_id = start_patient_id + i
            
            # 기본 식별 정보
            index_key = f"{ptmiemcd}_{patient_id:08d}_{ptmiindt}"
            ptmiidno = f"SYN{patient_id:08d}"
            
            # 도착 시간 생성 (시간대별 가중치 기반)
            ptmiintm = self._generate_arrival_time()

            # DAG 순서에 따른 임상 속성 순차 생성
            clinical_attrs = self._generate_sequential_attributes(
                ptmiemcd, ptmibrtd, ptmisexx, ptmizipc
            )

            # 퇴실 시간 계산
            ptmiotdt, ptmiottm = self._calculate_discharge_time(
                ptmiindt, ptmiintm, clinical_attrs['ptmikts1']
            )

            # 레코드 구성
            record = {
                'index_key': index_key,
                'ptmiemcd': ptmiemcd,
                'ptmiidno': ptmiidno,
                'ptmiindt': ptmiindt,
                'ptmiintm': ptmiintm,
                'ptmibrtd': ptmibrtd,
                'ptmisexx': ptmisexx,
                'ptmizipc': ptmizipc,
                'ptmiotdt': ptmiotdt,
                'ptmiottm': ptmiottm,
                **clinical_attrs
            }
            
            records.append(record)
        
        return records
    
    def _generate_arrival_time(self) -> str:
        """시간대별 가중치를 고려한 도착 시간 생성"""
        
        # 시간 선택 (가중치 기반)
        hours = list(self.hourly_weights.keys())
        weights = np.array(list(self.hourly_weights.values()))
        
        # 확률 정규화 (합이 1이 되도록)
        weights = weights / weights.sum()
        
        hour = np.random.choice(hours, p=weights)
        
        # 분 무작위 선택
        minute = np.random.randint(0, 60)
        
        return f"{hour:02d}{minute:02d}"
    
    def _generate_sequential_attributes(self, ptmiemcd: str, ptmibrtd: str, 
                                      ptmisexx: str, ptmizipc: str) -> Dict[str, Any]:
        """
        DAG 순서에 따른 순차적 임상 속성 생성
        
        Args:
            ptmiemcd: 병원 코드
            ptmibrtd: 연령군
            ptmisexx: 성별
            ptmizipc: 거주지 코드
            
        Returns:
            생성된 임상 속성 딕셔너리
        """
        
        attributes = {}
        
        try:
            # 1. 내원수단 (ptmiinmn) 생성
            attributes['ptmiinmn'] = self._generate_visit_method(ptmibrtd)

            # 2. 주증상 (ptmimnsy) 생성
            attributes['ptmimnsy'] = self._generate_chief_complaint(ptmibrtd, ptmisexx)

            # 3. KTAS 등급 (ptmikts1) 생성 - 조건부 확률 사용
            attributes['ptmikts1'] = self._generate_ktas_level(
                ptmibrtd, ptmisexx, ptmiemcd, attributes['ptmiinmn']
            )
            attributes['ptmikpr1'] = str(attributes['ptmikts1'])

            # 4. 주요치료과 (ptmidept) 생성
            attributes['ptmidept'] = self._generate_treatment_department(
                attributes['ptmikts1'], attributes['ptmimnsy']
            )

            # 5. 응급치료결과 (ptmiemrt) 생성 - KTAS에 강하게 의존
            attributes['ptmiemrt'] = self._generate_treatment_result(
                attributes['ptmikts1'], ptmibrtd
            )
            
        except Exception as e:
            self.logger.error(f"Error in sequential attribute generation: {e}")
            self.logger.error(f"Context: ptmiemcd={ptmiemcd}, ptmibrtd={ptmibrtd}, ptmisexx={ptmisexx}")
            raise
        
        return attributes
    
    def _generate_visit_method(self, ptmibrtd: str) -> str:
        """
        연령대 기반 내원수단 생성
        
        Args:
            ptmibrtd: 연령군
            
        Returns:
            내원수단 코드
        """
        
        # 연령대별 내원수단 확률 (실제 NEDIS 패턴 기반)
        age_ptmiinmnod_probs = {
            '01': {'1': 0.60, '3': 0.25, '6': 0.10, '9': 0.05},  # 영아 - 119 구급차 높음
            '09': {'1': 0.40, '3': 0.20, '6': 0.30, '9': 0.10},  # 유아
            '10': {'1': 0.15, '3': 0.15, '6': 0.60, '9': 0.10},  # 10대 - 도보 높음
            '20': {'1': 0.10, '3': 0.10, '6': 0.70, '9': 0.10},  # 20대
            '30': {'1': 0.12, '3': 0.13, '6': 0.65, '9': 0.10},  # 30대
            '40': {'1': 0.15, '3': 0.15, '6': 0.60, '9': 0.10},  # 40대
            '50': {'1': 0.20, '3': 0.18, '6': 0.52, '9': 0.10},  # 50대
            '60': {'1': 0.25, '3': 0.20, '6': 0.45, '9': 0.10},  # 60대
            '70': {'1': 0.35, '3': 0.25, '6': 0.30, '9': 0.10},  # 70대 - 구급차 증가
            '80': {'1': 0.45, '3': 0.30, '6': 0.20, '9': 0.05},  # 80대
            '90': {'1': 0.50, '3': 0.35, '6': 0.10, '9': 0.05}   # 90대
        }
        
        probs = age_ptmiinmnod_probs.get(ptmibrtd, age_ptmiinmnod_probs['30'])  # 기본값
        methods = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # 확률 정규화 (합이 1이 되도록)
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(methods, p=probabilities)
    
    def _generate_chief_complaint(self, ptmibrtd: str, ptmisexx: str) -> str:
        """
        연령/성별 기반 주증상 생성
        
        Args:
            ptmibrtd: 연령군
            ptmisexx: 성별
            
        Returns:
            주증상 코드
        """
        
        # 단순화된 주증상 카테고리 (실제로는 더 세분화)
        age_sex_ptmimnsy_probs = {
            ('01', '1'): {'10': 0.30, '20': 0.25, '30': 0.20, '40': 0.15, '50': 0.10},  # 영아 남성
            ('01', '2'): {'10': 0.30, '20': 0.25, '30': 0.20, '40': 0.15, '50': 0.10},  # 영아 여성
            # ... 다른 연령/성별 조합들도 유사하게 정의
        }
        
        # 기본 확률 (모든 연령/성별에 적용)
        default_probs = {'10': 0.20, '20': 0.18, '30': 0.15, '40': 0.12, '50': 0.10, 
                        '60': 0.08, '70': 0.07, '80': 0.06, '90': 0.04}
        
        probs = age_sex_ptmimnsy_probs.get((ptmibrtd, ptmisexx), default_probs)
        symptoms = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # 확률 정규화 (합이 1이 되도록)
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(symptoms, p=probabilities)
    
    def _generate_ktas_level(self, ptmibrtd: str, ptmisexx: str, 
                           ptmiemcd: str, ptmiinmn: str) -> str:
        """
        조건부 확률 테이블을 사용한 KTAS 등급 생성
        
        Args:
            ptmibrtd: 연령군
            ptmisexx: 성별
            ptmiemcd: 병원 코드
            ptmiinmn: 내원수단
            
        Returns:
            KTAS 등급
        """
        
        try:
            # 병원 종별 조회
            hospital_info = self.db.fetch_dataframe("""
                SELECT gubun FROM nedis_meta.hospital_capacity
                WHERE ptmiemcd = ?
            """, [ptmiemcd])

            gubun = hospital_info.iloc[0]['gubun'] if len(hospital_info) > 0 else '지역기관'

            # 조건부 확률 조회
            ktas_probs = self.db.fetch_dataframe("""
                SELECT ptmikts1, probability
                FROM nedis_meta.ktas_conditional_prob
                WHERE ptmibrtd = ? AND ptmisexx = ? AND gubun = ? AND ptmiinmn = ?
                ORDER BY probability DESC
            """, [ptmibrtd, ptmisexx, gubun, ptmiinmn])

            if len(ktas_probs) == 0:
                # 조건부 확률이 없으면 전체 평균 사용
                ktas_probs = self.db.fetch_dataframe("""
                    SELECT ptmikts1, AVG(probability) as probability
                    FROM nedis_meta.ktas_conditional_prob
                    WHERE ptmibrtd = ? AND ptmisexx = ?
                    GROUP BY ptmikts1
                    ORDER BY probability DESC
                """, [ptmibrtd, ptmisexx])
            
            if len(ktas_probs) == 0:
                # 여전히 없으면 기본 KTAS 분포 사용
                default_probs = np.array([0.007, 0.053, 0.385, 0.485, 0.070])
                default_probs = default_probs / default_probs.sum()  # 확률 정규화
                return np.random.choice(['1', '2', '3', '4', '5'], p=default_probs)
            
            # 확률 정규화
            ktas_levels = ktas_probs['ptmikts1'].values
            probabilities = ktas_probs['probability'].values
            probabilities = probabilities / probabilities.sum()
            
            return np.random.choice(ktas_levels, p=probabilities)
            
        except Exception as e:
            self.logger.warning(f"KTAS generation fallback for {ptmiemcd}: {e}")
            # 기본 분포로 폴백
            default_probs = np.array([0.007, 0.053, 0.385, 0.485, 0.070])
            default_probs = default_probs / default_probs.sum()  # 확률 정규화
            return np.random.choice(['1', '2', '3', '4', '5'], p=default_probs)
    
    def _generate_treatment_department(self, ktas_level: str, ptmimnsy: str) -> str:
        """
        KTAS와 주증상 기반 주요치료과 생성
        
        Args:
            ktas_level: KTAS 등급
            ptmimnsy: 주증상
            
        Returns:
            치료과 코드
        """
        
        # KTAS별 치료과 확률 (응급의학과 중심)
        ktas_dept_probs = {
            '1': {'01': 0.60, '02': 0.15, '03': 0.10, '04': 0.10, '99': 0.05},  # 중증 - 응급의학과 집중
            '2': {'01': 0.50, '02': 0.20, '03': 0.15, '04': 0.10, '99': 0.05},  # 응급
            '3': {'01': 0.40, '02': 0.25, '03': 0.20, '04': 0.10, '99': 0.05},  # 긴급
            '4': {'01': 0.30, '02': 0.30, '03': 0.25, '04': 0.10, '99': 0.05},  # 준응급
            '5': {'01': 0.20, '02': 0.35, '03': 0.30, '04': 0.10, '99': 0.05}   # 비응급 - 타과 증가
        }
        
        probs = ktas_dept_probs.get(ktas_level, ktas_dept_probs['3'])
        departments = list(probs.keys())
        probabilities = np.array(list(probs.values()))
        
        # 확률 정규화 (합이 1이 되도록)
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(departments, p=probabilities)
    
    def _generate_treatment_result(self, ktas_level: str, ptmibrtd: str) -> str:
        """
        KTAS 등급 기반 응급치료결과 생성 (의학적 가이드라인 적용)
        
        Args:
            ktas_level: KTAS 등급
            ptmibrtd: 연령군 (고령자 중증도 조정)
            
        Returns:
            치료결과 코드
        """
        
        base_probs = self.ktas_outcome_probs.get(ktas_level, self.ktas_outcome_probs['3'])
        
        # 고령자 중증도 조정 (입원율 증가, 귀가율 감소)
        if ptmibrtd in ['70', '80', '90']:
            adjusted_probs = base_probs.copy()
            
            # 고령자는 입원 확률 증가
            home_discharge_reduction = 0.1  # 귀가율 10% 포인트 감소
            admission_increase = home_discharge_reduction  # 입원율 증가
            
            adjusted_probs['11'] = max(0.01, adjusted_probs['11'] - home_discharge_reduction)
            adjusted_probs['31'] = min(0.90, adjusted_probs['31'] + admission_increase)
            
            base_probs = adjusted_probs
        
        outcomes = list(base_probs.keys())
        probabilities = list(base_probs.values())
        
        # 확률 정규화 (합이 1이 되도록)
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()
        
        return np.random.choice(outcomes, p=probabilities)
    
    def _calculate_discharge_time(self, ptmiindt: str, ptmiintm: str, ktas_level: str) -> Tuple[str, str]:
        """
        KTAS 등급 기반 체류시간 계산 및 퇴실시간 결정
        
        Args:
            ptmiindt: 방문 날짜
            ptmiintm: 방문 시간
            ktas_level: KTAS 등급
            
        Returns:
            (퇴실 날짜, 퇴실 시간) 튜플
        """
        
        # KTAS별 체류시간 분포 파라미터 (분 단위)
        duration_params = {
            '1': {'mean': 240, 'std': 120, 'max': 720},   # 4시간 ± 2시간
            '2': {'mean': 180, 'std': 90, 'max': 600},    # 3시간 ± 1.5시간
            '3': {'mean': 120, 'std': 60, 'max': 480},    # 2시간 ± 1시간
            '4': {'mean': 90, 'std': 45, 'max': 360},     # 1.5시간 ± 45분
            '5': {'mean': 60, 'std': 30, 'max': 240}      # 1시간 ± 30분
        }
        
        params = duration_params.get(ktas_level, duration_params['3'])
        
        # 로그정규분포를 사용한 체류시간 생성 (현실적인 분포)
        log_mean = np.log(params['mean'])
        log_std = 0.5  # 로그정규분포의 표준편차
        
        duration_minutes = np.random.lognormal(log_mean, log_std)
        duration_minutes = min(duration_minutes, params['max'])  # 최대값 제한
        duration_minutes = max(duration_minutes, 15)  # 최소 15분
        
        # 방문시간을 datetime으로 변환
        arrival_datetime = datetime.strptime(f"{ptmiindt}{ptmiintm}", "%Y%m%d%H%M")
        
        # 체류시간 추가
        discharge_datetime = arrival_datetime + timedelta(minutes=int(duration_minutes))
        
        # 날짜 변경 처리 (자정 넘어가는 경우)
        ptmiotdt = discharge_datetime.strftime("%Y%m%d")
        ptmiottm = discharge_datetime.strftime("%H%M")
        
        return ptmiotdt, ptmiottm
    
    def _batch_insert_records(self, records: List[Dict[str, Any]]):
        """배치 레코드 삽입"""
        
        self.logger.info(f"Batch inserting {len(records)} clinical records")
        
        try:
            insert_sql = """
                INSERT INTO nedis_synthetic.clinical_records
                (index_key, ptmiemcd, ptmiidno, ptmiindt, ptmiintm, ptmibrtd, ptmisexx, ptmizipc,
                 ptmiinmn, ptmikts1, ptmikpr1, ptmimnsy, ptmidept, ptmiemrt, ptmiotdt, ptmiottm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """

            batch_data = []
            for record in records:
                batch_data.append((
                    record['index_key'], record['ptmiemcd'], record['ptmiidno'],
                    record['ptmiindt'], record['ptmiintm'], record['ptmibrtd'],
                    record['ptmisexx'], record['ptmizipc'], record['ptmiinmn'],
                    record['ptmikts1'], record['ptmikpr1'], record['ptmimnsy'],
                    record['ptmidept'], record['ptmiemrt'],
                    record['ptmiotdt'], record['ptmiottm']
                ))
            
            # 배치 실행 (청크 단위)
            chunk_size = 1000
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                for row in chunk:
                    self.db.execute_query(insert_sql, row)
            
            self.logger.info("Batch insertion completed successfully")
            
        except Exception as e:
            self.logger.error(f"Batch insertion failed: {e}")
            raise
    
    def _get_generation_summary(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """생성 결과 요약 통계"""
        
        if not records:
            return {}
        
        records_df = pd.DataFrame(records)
        
        summary = {
            'total_records': len(records),
            'unique_hospitals': records_df['ptmiemcd'].nunique(),
            'ktas_distribution': records_df['ptmikts1'].value_counts().to_dict(),
            'treatment_result_distribution': records_df['ptmiemrt'].value_counts().to_dict(),
            'visit_method_distribution': records_df['ptmiinmn'].value_counts().to_dict(),
            'age_group_distribution': records_df['ptmibrtd'].value_counts().to_dict(),
            'gender_distribution': records_df['ptmisexx'].value_counts().to_dict()
        }

        # 의학적 유효성 체크
        ktas_severe = records_df['ptmikts1'].isin(['1', '2']).sum()
        severe_home_discharge = records_df[
            (records_df['ptmikts1'].isin(['1', '2'])) &
            (records_df['ptmiemrt'] == '11')
        ].shape[0]
        
        summary['medical_validity'] = {
            'severe_patients': int(ktas_severe),
            'severe_home_discharge': int(severe_home_discharge),
            'severe_home_discharge_rate': float(severe_home_discharge / ktas_severe) if ktas_severe > 0 else 0.0
        }
        
        return summary
    
    def generate_batch_clinical_attributes(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        날짜 범위에 대한 배치 임상 속성 생성
        
        Args:
            start_date: 시작 날짜 ('YYYYMMDD')
            end_date: 종료 날짜 ('YYYYMMDD')
            
        Returns:
            배치 생성 결과
        """
        
        self.logger.info(f"Starting batch clinical attributes generation: {start_date} to {end_date}")
        
        try:
            # 날짜 범위 생성
            date_range = pd.date_range(
                start=pd.to_datetime(start_date, format='%Y%m%d'),
                end=pd.to_datetime(end_date, format='%Y%m%d'),
                freq='D'
            )
            
            batch_results = []
            total_records = 0
            successful_dates = 0
            failed_dates = 0
            
            for date in date_range:
                date_str = date.strftime('%Y%m%d')
                
                try:
                    result = self.generate_clinical_attributes(date_str)
                    
                    if result['success']:
                        successful_dates += 1
                        total_records += result['records_generated']
                    else:
                        failed_dates += 1
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Clinical generation failed for date {date_str}: {e}")
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
                'total_records_generated': total_records,
                'avg_records_per_day': total_records / successful_dates if successful_dates > 0 else 0,
                'results': batch_results
            }
            
            self.logger.info(
                f"Batch clinical generation completed: {total_records} records, "
                f"{successful_dates}/{len(date_range)} successful days"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Batch clinical generation failed: {e}")
            return {'success': False, 'error': str(e)}