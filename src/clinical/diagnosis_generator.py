"""
Diagnosis Code Generator

조건부 확률 기반 진단 코드 생성기입니다.
주진단과 부진단을 의학적 연관성을 고려하여 생성하며,
입원 환자의 경우 입원 진단도 함께 생성합니다.

생성 순서:
1. 주진단 (position=1) 생성 - KTAS, 연령, 성별 기반
2. 부진단 (position>1) 생성 - 주진단과의 연관성 고려  
3. 입원 진단 생성 (입원 환자만) - ER 진단과 70% 일치
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
import logging
import re

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class DiagnosisGenerator:
    """진단 코드 생성기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        진단 코드 생성기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # KTAS별 진단 개수 분포
        self.ktas_diagnosis_count = {
            '1': {'min': 2, 'max': 4, 'avg': 3.2},  # 중증 - 다수 진단
            '2': {'min': 1, 'max': 3, 'avg': 2.1},  # 응급
            '3': {'min': 1, 'max': 2, 'avg': 1.4},  # 긴급  
            '4': {'min': 1, 'max': 2, 'avg': 1.2},  # 준응급
            '5': {'min': 1, 'max': 1, 'avg': 1.0}   # 비응급 - 단일 진단
        }
        
        # ICD 대분류별 연관 진단 그룹
        self.icd_chapter_associations = {
            'A': ['B', 'Z'],  # 감염성 질환 → 기생충, 건강상태
            'B': ['A', 'Z'],  # 기생충 → 감염성, 건강상태
            'C': ['D', 'Z'],  # 악성 신생물 → 양성 신생물, 건강상태
            'D': ['C', 'Z'],  # 신생물 → 악성, 건강상태
            'E': ['N', 'Z'],  # 내분비 → 비뇨생식기, 건강상태
            'F': ['G', 'Z'],  # 정신 → 신경계, 건강상태
            'G': ['F', 'R'],  # 신경계 → 정신, 증상
            'H': ['R', 'Z'],  # 감각기관 → 증상, 건강상태
            'I': ['R', 'E'],  # 순환계 → 증상, 내분비
            'J': ['R', 'Z'],  # 호흡계 → 증상, 건강상태
            'K': ['R', 'E'],  # 소화계 → 증상, 내분비
            'L': ['R', 'Z'],  # 피부 → 증상, 건강상태
            'M': ['R', 'S'],  # 근골격계 → 증상, 외상
            'N': ['E', 'R'],  # 비뇨생식기 → 내분비, 증상
            'O': ['P', 'Z'],  # 임신출산 → 주산기, 건강상태
            'P': ['O', 'Q'],  # 주산기 → 임신출산, 선천기형
            'Q': ['P', 'Z'],  # 선천기형 → 주산기, 건강상태
            'R': ['I', 'J', 'K', 'G'],  # 증상 → 주요 장기계
            'S': ['T', 'V'],  # 외상 → 중독, 외인
            'T': ['S', 'V'],  # 중독 → 외상, 외인
            'U': ['Z'],      # 특수 목적 → 건강상태
            'V': ['S', 'T', 'Y'],  # 외인 → 외상, 중독, 의도미상
            'W': ['S', 'T', 'V'],  # 사고 → 외상, 중독, 외인
            'X': ['S', 'T', 'V'],  # 의도적자해 → 외상, 중독, 외인
            'Y': ['S', 'T', 'V'],  # 의도미상 → 외상, 중독, 외인
            'Z': ['A', 'B', 'C', 'D', 'E', 'H', 'L', 'Q', 'U']  # 건강상태 → 다양한 질환
        }
        
        # 공존 질환 행렬 초기화
        self.comorbidity_matrix = {}
        self._learn_comorbidity_matrix()
        
    def _learn_comorbidity_matrix(self):
        """실제 데이터에서 진단 코드 간 동시 출현 빈도 학습"""
        self.logger.info("Learning diagnosis comorbidity matrix from real data")
        
        try:
            source_table = self.config.get('original.source_table', 'nedis_original.nedis2017')
            
            # 주진단과 부진단 쌍 조회 (빈도 높은 순)
            # 주진단(position=1)과 그 환자의 다른 진단들 간의 관계
            query = f"""
                WITH diagnosis_pairs AS (
                    SELECT 
                        d1.diagnosis_code as primary_code,
                        d2.diagnosis_code as secondary_code
                    FROM {source_table}_diag d1
                    JOIN {source_table}_diag d2 ON d1.index_key = d2.index_key
                    WHERE d1.position = 1 AND d2.position > 1
                )
                SELECT 
                    primary_code, secondary_code, COUNT(*) as frequency
                FROM diagnosis_pairs
                GROUP BY primary_code, secondary_code
                HAVING COUNT(*) >= 5
                ORDER BY primary_code, frequency DESC
            """
            
            # 주의: 실제 테이블 구조에 따라 쿼리 조정 필요
            # 여기서는 예시로 구현하며, 실제로는 DB 구조에 맞춰야 함
            # 만약 별도 진단 테이블이 없다면 로직 변경 필요
            
            # 임시: 메타 테이블이나 통계 테이블 활용
            # 여기서는 간단히 빈 딕셔너리로 초기화하고 로그만 남김 (실제 데이터 접근 불가 가정 시)
            # 실제 환경에서는 위 쿼리 실행
            
            self.comorbidity_matrix = {}
            self.logger.info("Comorbidity matrix initialized (placeholder)")
                
        except Exception as e:
            self.logger.warning(f"Failed to learn comorbidity matrix: {e}")
            self.comorbidity_matrix = {}

    def initialize_diagnosis_tables(self):
        """진단 테이블들 초기화"""
        
        self.logger.info("Initializing diagnosis tables")
        
        try:
            # ER 진단 테이블 생성 (존재하지 않는 경우만)
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS nedis_synthetic.diag_er (
                    index_key VARCHAR NOT NULL,
                    position INTEGER NOT NULL,
                    diagnosis_code VARCHAR NOT NULL,
                    diagnosis_category VARCHAR DEFAULT '1',
                    icd_chapter VARCHAR,
                    generation_method VARCHAR DEFAULT 'conditional_probability',
                    PRIMARY KEY (index_key, position)
                )
            """)
            
            # 입원 진단 테이블 생성 (존재하지 않는 경우만)
            self.db.execute_query("""
                CREATE TABLE IF NOT EXISTS nedis_synthetic.diag_adm (
                    index_key VARCHAR NOT NULL,
                    position INTEGER NOT NULL,
                    diagnosis_code VARCHAR NOT NULL,
                    diagnosis_category VARCHAR DEFAULT '1',
                    icd_chapter VARCHAR,
                    generation_method VARCHAR DEFAULT 'er_based',
                    PRIMARY KEY (index_key, position)
                )
            """)
            
            self.logger.info("Diagnosis tables created successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize diagnosis tables: {e}")
            raise
    
    def generate_all_diagnoses(self, date_str: str) -> Dict[str, Any]:
        """
        특정 날짜의 모든 환자에 대해 진단 생성
        
        Args:
            date_str: 대상 날짜 ('YYYYMMDD' 형식)
            
        Returns:
            진단 생성 결과 딕셔너리
        """
        
        self.logger.info(f"Generating diagnoses for date: {date_str}")
        
        try:
            # 해당 날짜 임상 레코드 조회
            clinical_records = self.db.fetch_dataframe("""
                SELECT 
                    index_key, emorg_cd, pat_age_gr, pat_sex, 
                    ktas_fstu, emtrt_rust, main_trt_p
                FROM nedis_synthetic.clinical_records
                WHERE vst_dt = ?
                ORDER BY index_key
            """, [date_str])
            
            if len(clinical_records) == 0:
                self.logger.warning(f"No clinical records found for date: {date_str}")
                return {'success': False, 'reason': 'No clinical records'}
            
            # 해당 날짜의 기존 진단 데이터 삭제 (재실행 시 중복 방지)
            self.db.execute_query("""
                DELETE FROM nedis_synthetic.diag_er 
                WHERE index_key IN (
                    SELECT index_key FROM nedis_synthetic.clinical_records 
                    WHERE vst_dt = ?
                )
            """, [date_str])
            
            self.db.execute_query("""
                DELETE FROM nedis_synthetic.diag_adm 
                WHERE index_key IN (
                    SELECT index_key FROM nedis_synthetic.clinical_records 
                    WHERE vst_dt = ?
                )
            """, [date_str])
            
            total_patients = len(clinical_records)
            self.logger.info(f"Generating diagnoses for {total_patients} patients")
            
            er_diagnoses = []
            admission_diagnoses = []
            
            for _, patient in clinical_records.iterrows():
                # ER 진단 생성
                patient_er_diagnoses = self._generate_er_diagnoses(patient)
                er_diagnoses.extend(patient_er_diagnoses)
                
                # 입원 환자의 경우 입원 진단도 생성
                if patient['emtrt_rust'] in ['31', '32']:  # 병실입원, 중환자실입원
                    patient_admission_diagnoses = self._generate_admission_diagnoses(
                        patient, patient_er_diagnoses
                    )
                    admission_diagnoses.extend(patient_admission_diagnoses)
            
            # 배치 삽입
            self._batch_insert_er_diagnoses(er_diagnoses)
            if admission_diagnoses:
                self._batch_insert_admission_diagnoses(admission_diagnoses)
            
            # 결과 요약
            result = {
                'success': True,
                'date': date_str,
                'patients_processed': total_patients,
                'er_diagnoses_generated': len(er_diagnoses),
                'admission_diagnoses_generated': len(admission_diagnoses),
                'admission_patients': sum(1 for p in clinical_records.itertuples() 
                                        if p.emtrt_rust in ['31', '32']),
                'diagnosis_summary': self._get_diagnosis_summary(er_diagnoses, admission_diagnoses)
            }
            
            self.logger.info(
                f"Diagnosis generation completed: "
                f"{len(er_diagnoses)} ER diagnoses, {len(admission_diagnoses)} admission diagnoses"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to generate diagnoses: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_er_diagnoses(self, patient: pd.Series) -> List[Dict[str, Any]]:
        """
        개별 환자의 ER 진단 생성
        
        Args:
            patient: 환자 임상 데이터
            
        Returns:
            생성된 ER 진단 리스트
        """
        
        index_key = patient['index_key']
        pat_age_gr = patient['pat_age_gr']
        pat_sex = patient['pat_sex']
        ktas_level = patient['ktas_fstu']
        emorg_cd = patient['emorg_cd']
        
        diagnoses = []
        
        try:
            # 1. 진단 개수 결정
            count_params = self.ktas_diagnosis_count.get(ktas_level, self.ktas_diagnosis_count['3'])
            diagnosis_count = np.random.randint(count_params['min'], count_params['max'] + 1)
            
            # 2. 주진단 (position=1) 생성
            primary_diagnosis = self._generate_primary_diagnosis(
                pat_age_gr, pat_sex, ktas_level, emorg_cd
            )
            
            if primary_diagnosis:
                diagnoses.append({
                    'index_key': index_key,
                    'position': 1,
                    'diagnosis_code': primary_diagnosis['code'],
                    'diagnosis_category': '1',
                    'icd_chapter': primary_diagnosis['chapter'],
                    'generation_method': 'primary_conditional'
                })
            
            # 3. 부진단 생성 (position > 1)
            if diagnosis_count > 1 and primary_diagnosis:
                secondary_diagnoses = self._generate_secondary_diagnoses(
                    primary_diagnosis, diagnosis_count - 1, pat_age_gr, pat_sex
                )
                
                for i, sec_diag in enumerate(secondary_diagnoses, start=2):
                    diagnoses.append({
                        'index_key': index_key,
                        'position': i,
                        'diagnosis_code': sec_diag['code'],
                        'diagnosis_category': '1',
                        'icd_chapter': sec_diag['chapter'],
                        'generation_method': 'secondary_associated'
                    })
            
            return diagnoses
            
        except Exception as e:
            self.logger.warning(f"ER diagnosis generation failed for patient {index_key}: {e}")
            return []
    
    def _generate_primary_diagnosis(self, pat_age_gr: str, pat_sex: str, 
                                  ktas_level: str, emorg_cd: str) -> Optional[Dict[str, str]]:
        """
        조건부 확률 기반 주진단 생성
        
        Args:
            pat_age_gr: 연령군
            pat_sex: 성별
            ktas_level: KTAS 등급
            emorg_cd: 병원 코드
            
        Returns:
            주진단 정보 딕셔너리 or None
        """
        
        try:
            # 병원 종별 조회
            hospital_info = self.db.fetch_dataframe("""
                SELECT gubun FROM nedis_meta.hospital_capacity 
                WHERE emorg_cd = ?
            """, [emorg_cd])
            
            gubun = hospital_info.iloc[0]['gubun'] if len(hospital_info) > 0 else '지역기관'
            
            # 주진단 조건부 확률 조회
            primary_diagnosis_probs = self.db.fetch_dataframe("""
                SELECT diagnosis_code, probability
                FROM nedis_meta.diagnosis_conditional_prob
                WHERE pat_age_gr = ? AND pat_sex = ? AND gubun = ? 
                      AND ktas_fstu = ? AND is_primary = true
                ORDER BY probability DESC
                LIMIT 50
            """, [pat_age_gr, pat_sex, gubun, ktas_level])
            
            if len(primary_diagnosis_probs) == 0:
                # 조건을 완화하여 재시도
                primary_diagnosis_probs = self.db.fetch_dataframe("""
                    SELECT diagnosis_code, AVG(probability) as probability
                    FROM nedis_meta.diagnosis_conditional_prob
                    WHERE pat_age_gr = ? AND pat_sex = ? AND is_primary = true
                    GROUP BY diagnosis_code
                    ORDER BY probability DESC
                    LIMIT 30
                """, [pat_age_gr, pat_sex])
            
            if len(primary_diagnosis_probs) == 0:
                return None
            
            # 확률적 선택 (상위 진단에 더 높은 가중치)
            codes = primary_diagnosis_probs['diagnosis_code'].values
            probs = primary_diagnosis_probs['probability'].values
            
            # 확률 정규화
            probs = probs / probs.sum()
            
            selected_code = np.random.choice(codes, p=probs)
            
            return {
                'code': selected_code,
                'chapter': self._get_icd_chapter(selected_code)
            }
            
        except Exception as e:
            self.logger.warning(f"Primary diagnosis generation failed: {e}")
            return None
    
    def _generate_secondary_diagnoses(self, primary_diagnosis: Dict[str, str], 
                                    count: int, pat_age_gr: str, pat_sex: str) -> List[Dict[str, str]]:
        """
        주진단과 연관된 부진단들 생성
        
        Args:
            primary_diagnosis: 주진단 정보
            count: 생성할 부진단 개수
            pat_age_gr: 연령군
            pat_sex: 성별
            
        Returns:
            부진단 정보 리스트
        """
        
        secondary_diagnoses = []
        primary_chapter = primary_diagnosis['chapter']
        
        try:
            used_codes = {primary_diagnosis['code']}
            
            # 1. 공존 질환 행렬 기반 생성 시도 (우선순위)
            # 첫 번째 부진단은 가급적 강한 상관관계를 가진 것으로 생성
            correlated_diag = self._generate_correlated_secondary_diagnosis(
                primary_diagnosis['code'], used_codes
            )
            
            if correlated_diag:
                secondary_diagnoses.append(correlated_diag)
                used_codes.add(correlated_diag['code'])
                count -= 1
            
            if count <= 0:
                return secondary_diagnoses
            
            # 2. 기존 챕터 기반 로직 (나머지)
            # 연관 ICD 챕터들 가져오기
            associated_chapters = self.icd_chapter_associations.get(primary_chapter, [])
            
            # 주진단과 같은 챕터도 포함 (50% 확률)
            if np.random.random() < 0.5:
                associated_chapters.append(primary_chapter)
            
            if not associated_chapters:
                associated_chapters = ['R']  # 기본값: 증상 챕터
            
            for _ in range(count):
                # 연관 챕터 선택
                selected_chapter = np.random.choice(associated_chapters)
                
                # 해당 챕터에서 진단 선택
                secondary_diagnosis = self._select_diagnosis_from_chapter(
                    selected_chapter, pat_age_gr, pat_sex, used_codes
                )
                
                if secondary_diagnosis:
                    secondary_diagnoses.append(secondary_diagnosis)
                    used_codes.add(secondary_diagnosis['code'])
            
            return secondary_diagnoses
            
        except Exception as e:
            self.logger.warning(f"Secondary diagnosis generation failed: {e}")
            return []
            
    def _generate_correlated_secondary_diagnosis(self, primary_code: str, 
                                               used_codes: Set[str]) -> Optional[Dict[str, str]]:
        """
        공존 질환 행렬 기반 부진단 생성
        """
        try:
            # 해당 주진단과 연관된 부진단 후보군 조회 (메모리 내 행렬 또는 DB 조회)
            # 여기서는 placeholder로 구현 (실제로는 self.comorbidity_matrix 활용)
            
            # 실제 구현 시:
            # candidates = self.comorbidity_matrix.get(primary_code, [])
            # if not candidates: return None
            
            # DB 직접 조회 방식 (메모리 절약)
            candidates = self.db.fetch_dataframe("""
                SELECT secondary_code, frequency
                FROM (
                    -- 가상의 공존 질환 테이블 (실제로는 _learn_comorbidity_matrix에서 생성했어야 함)
                    -- 여기서는 diagnosis_conditional_prob를 활용하여 유사하게 구현
                    SELECT diagnosis_code as secondary_code, probability as frequency
                    FROM nedis_meta.diagnosis_conditional_prob
                    WHERE is_primary = false
                    ORDER BY probability DESC
                    LIMIT 10
                )
            """)
            
            if len(candidates) == 0:
                return None
                
            # 이미 사용된 코드 제외
            available_candidates = candidates[
                ~candidates['secondary_code'].isin(used_codes)
            ]
            
            if len(available_candidates) == 0:
                return None
                
            # 확률적 선택
            codes = available_candidates['secondary_code'].values
            freqs = available_candidates['frequency'].values
            probs = freqs / freqs.sum()
            
            selected_code = np.random.choice(codes, p=probs)
            
            return {
                'code': selected_code,
                'chapter': self._get_icd_chapter(selected_code)
            }
            
        except Exception as e:
            # self.logger.warning(f"Correlated secondary diagnosis generation failed: {e}")
            return None
    
    def _select_diagnosis_from_chapter(self, chapter: str, pat_age_gr: str, 
                                     pat_sex: str, used_codes: Set[str]) -> Optional[Dict[str, str]]:
        """
        특정 ICD 챕터에서 진단 선택
        
        Args:
            chapter: ICD 챕터 코드
            pat_age_gr: 연령군
            pat_sex: 성별
            used_codes: 이미 사용된 진단 코드들
            
        Returns:
            선택된 진단 정보 or None
        """
        
        try:
            # 해당 챕터의 진단들 조회
            chapter_diagnoses = self.db.fetch_dataframe("""
                SELECT diagnosis_code, AVG(probability) as probability
                FROM nedis_meta.diagnosis_conditional_prob
                WHERE pat_age_gr = ? AND pat_sex = ? 
                      AND diagnosis_code LIKE ?
                GROUP BY diagnosis_code
                ORDER BY probability DESC
                LIMIT 20
            """, [pat_age_gr, pat_sex, f"{chapter}%"])
            
            if len(chapter_diagnoses) == 0:
                # 연령/성별 조건 없이 재시도
                chapter_diagnoses = self.db.fetch_dataframe("""
                    SELECT diagnosis_code, AVG(probability) as probability
                    FROM nedis_meta.diagnosis_conditional_prob
                    WHERE diagnosis_code LIKE ?
                    GROUP BY diagnosis_code
                    ORDER BY probability DESC
                    LIMIT 15
                """, [f"{chapter}%"])
            
            if len(chapter_diagnoses) == 0:
                return None
            
            # 이미 사용된 코드 제외
            available_diagnoses = chapter_diagnoses[
                ~chapter_diagnoses['diagnosis_code'].isin(used_codes)
            ]
            
            if len(available_diagnoses) == 0:
                return None
            
            # 확률적 선택
            codes = available_diagnoses['diagnosis_code'].values
            probs = available_diagnoses['probability'].values
            probs = probs / probs.sum()
            
            selected_code = np.random.choice(codes, p=probs)
            
            return {
                'code': selected_code,
                'chapter': chapter
            }
            
        except Exception as e:
            self.logger.warning(f"Chapter diagnosis selection failed: {e}")
            return None
    
    def _get_icd_chapter(self, diagnosis_code: str) -> str:
        """진단 코드에서 ICD 챕터 추출"""
        
        if not diagnosis_code:
            return 'R'
        
        # 첫 글자가 ICD 챕터
        return diagnosis_code[0].upper()
    
    def _generate_admission_diagnoses(self, patient: pd.Series, 
                                    er_diagnoses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        입원 환자의 입원 진단 생성
        
        Args:
            patient: 환자 정보
            er_diagnoses: ER 진단 리스트
            
        Returns:
            입원 진단 리스트
        """
        
        if not er_diagnoses:
            return []
        
        index_key = patient['index_key']
        admission_diagnoses = []
        
        try:
            # 70% 확률로 ER 진단과 동일, 30% 확률로 새로운 진단
            for i, er_diag in enumerate(er_diagnoses, start=1):
                if np.random.random() < 0.7:  # 70% 확률로 동일
                    admission_diagnoses.append({
                        'index_key': index_key,
                        'position': i,
                        'diagnosis_code': er_diag['diagnosis_code'],
                        'diagnosis_category': '1',
                        'icd_chapter': er_diag['icd_chapter'],
                        'generation_method': 'er_identical'
                    })
                else:  # 30% 확률로 연관 진단
                    related_diag = self._generate_related_admission_diagnosis(
                        er_diag['diagnosis_code'], patient['pat_age_gr'], patient['pat_sex']
                    )
                    
                    if related_diag:
                        admission_diagnoses.append({
                            'index_key': index_key,
                            'position': i,
                            'diagnosis_code': related_diag['code'],
                            'diagnosis_category': '1',
                            'icd_chapter': related_diag['chapter'],
                            'generation_method': 'er_related'
                        })
                    else:
                        # 관련 진단 생성 실패 시 원래 진단 사용
                        admission_diagnoses.append({
                            'index_key': index_key,
                            'position': i,
                            'diagnosis_code': er_diag['diagnosis_code'],
                            'diagnosis_category': '1',
                            'icd_chapter': er_diag['icd_chapter'],
                            'generation_method': 'er_fallback'
                        })
            
            # 입원 시 추가 진단 (평균 1-2개 더)
            additional_count = np.random.poisson(1.5)  # 평균 1.5개 추가
            additional_count = min(additional_count, 3)  # 최대 3개
            
            used_codes = {diag['diagnosis_code'] for diag in admission_diagnoses}
            
            for i in range(additional_count):
                additional_diag = self._generate_additional_admission_diagnosis(
                    patient['pat_age_gr'], patient['pat_sex'], used_codes
                )
                
                if additional_diag:
                    admission_diagnoses.append({
                        'index_key': index_key,
                        'position': len(admission_diagnoses) + 1,
                        'diagnosis_code': additional_diag['code'],
                        'diagnosis_category': '1',
                        'icd_chapter': additional_diag['chapter'],
                        'generation_method': 'admission_additional'
                    })
                    used_codes.add(additional_diag['code'])
            
            return admission_diagnoses
            
        except Exception as e:
            self.logger.warning(f"Admission diagnosis generation failed for {index_key}: {e}")
            return []
    
    def _generate_related_admission_diagnosis(self, er_code: str, pat_age_gr: str, 
                                            pat_sex: str) -> Optional[Dict[str, str]]:
        """ER 진단과 연관된 입원 진단 생성"""
        
        er_chapter = self._get_icd_chapter(er_code)
        associated_chapters = self.icd_chapter_associations.get(er_chapter, [er_chapter])
        
        # 연관 챕터에서 진단 선택
        selected_chapter = np.random.choice(associated_chapters + [er_chapter])
        
        return self._select_diagnosis_from_chapter(
            selected_chapter, pat_age_gr, pat_sex, {er_code}
        )
    
    def _generate_additional_admission_diagnosis(self, pat_age_gr: str, pat_sex: str, 
                                               used_codes: Set[str]) -> Optional[Dict[str, str]]:
        """입원 추가 진단 생성"""
        
        # 일반적인 입원 진단 챕터들 (내과적 질환 중심)
        admission_chapters = ['I', 'J', 'K', 'E', 'N', 'R', 'Z']
        selected_chapter = np.random.choice(admission_chapters)
        
        return self._select_diagnosis_from_chapter(
            selected_chapter, pat_age_gr, pat_sex, used_codes
        )
    
    def _batch_insert_er_diagnoses(self, diagnoses: List[Dict[str, Any]]):
        """ER 진단 배치 삽입"""
        
        if not diagnoses:
            return
        
        self.logger.info(f"Batch inserting {len(diagnoses)} ER diagnoses")
        
        try:
            insert_sql = """
                INSERT INTO nedis_synthetic.diag_er
                (index_key, position, diagnosis_code, diagnosis_category, 
                 icd_chapter, generation_method)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            batch_data = []
            for diag in diagnoses:
                batch_data.append((
                    diag['index_key'], diag['position'], diag['diagnosis_code'],
                    diag['diagnosis_category'], diag['icd_chapter'], diag['generation_method']
                ))
            
            # 청크 단위로 삽입
            chunk_size = 1000
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                for row in chunk:
                    self.db.execute_query(insert_sql, row)
            
            self.logger.info("ER diagnoses batch insertion completed")
            
        except Exception as e:
            self.logger.error(f"ER diagnoses batch insertion failed: {e}")
            raise
    
    def _batch_insert_admission_diagnoses(self, diagnoses: List[Dict[str, Any]]):
        """입원 진단 배치 삽입"""
        
        if not diagnoses:
            return
        
        self.logger.info(f"Batch inserting {len(diagnoses)} admission diagnoses")
        
        try:
            insert_sql = """
                INSERT INTO nedis_synthetic.diag_adm
                (index_key, position, diagnosis_code, diagnosis_category,
                 icd_chapter, generation_method)
                VALUES (?, ?, ?, ?, ?, ?)
            """
            
            batch_data = []
            for diag in diagnoses:
                batch_data.append((
                    diag['index_key'], diag['position'], diag['diagnosis_code'],
                    diag['diagnosis_category'], diag['icd_chapter'], diag['generation_method']
                ))
            
            # 청크 단위로 삽입
            chunk_size = 1000
            for i in range(0, len(batch_data), chunk_size):
                chunk = batch_data[i:i + chunk_size]
                for row in chunk:
                    self.db.execute_query(insert_sql, row)
            
            self.logger.info("Admission diagnoses batch insertion completed")
            
        except Exception as e:
            self.logger.error(f"Admission diagnoses batch insertion failed: {e}")
            raise
    
    def _get_diagnosis_summary(self, er_diagnoses: List[Dict[str, Any]], 
                             admission_diagnoses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """진단 생성 결과 요약"""
        
        summary = {
            'er_diagnosis_summary': {},
            'admission_diagnosis_summary': {}
        }
        
        if er_diagnoses:
            er_df = pd.DataFrame(er_diagnoses)
            summary['er_diagnosis_summary'] = {
                'total_diagnoses': len(er_diagnoses),
                'unique_codes': er_df['diagnosis_code'].nunique(),
                'chapter_distribution': er_df['icd_chapter'].value_counts().to_dict(),
                'position_distribution': er_df['position'].value_counts().to_dict(),
                'avg_diagnoses_per_patient': er_df.groupby('index_key').size().mean()
            }
        
        if admission_diagnoses:
            adm_df = pd.DataFrame(admission_diagnoses)
            summary['admission_diagnosis_summary'] = {
                'total_diagnoses': len(admission_diagnoses),
                'unique_codes': adm_df['diagnosis_code'].nunique(),
                'chapter_distribution': adm_df['icd_chapter'].value_counts().to_dict(),
                'generation_method_distribution': adm_df['generation_method'].value_counts().to_dict(),
                'avg_diagnoses_per_patient': adm_df.groupby('index_key').size().mean()
            }
        
        return summary
    
    def generate_batch_diagnoses(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        날짜 범위에 대한 배치 진단 생성
        
        Args:
            start_date: 시작 날짜 ('YYYYMMDD')
            end_date: 종료 날짜 ('YYYYMMDD')
            
        Returns:
            배치 진단 생성 결과
        """
        
        self.logger.info(f"Starting batch diagnosis generation: {start_date} to {end_date}")
        
        try:
            # 날짜 범위 생성
            date_range = pd.date_range(
                start=pd.to_datetime(start_date, format='%Y%m%d'),
                end=pd.to_datetime(end_date, format='%Y%m%d'),
                freq='D'
            )
            
            batch_results = []
            total_er_diagnoses = 0
            total_admission_diagnoses = 0
            successful_dates = 0
            failed_dates = 0
            
            for date in date_range:
                date_str = date.strftime('%Y%m%d')
                
                try:
                    result = self.generate_all_diagnoses(date_str)
                    
                    if result['success']:
                        successful_dates += 1
                        total_er_diagnoses += result['er_diagnoses_generated']
                        total_admission_diagnoses += result['admission_diagnoses_generated']
                    else:
                        failed_dates += 1
                    
                    batch_results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Diagnosis generation failed for date {date_str}: {e}")
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
                'total_er_diagnoses': total_er_diagnoses,
                'total_admission_diagnoses': total_admission_diagnoses,
                'avg_er_diagnoses_per_day': total_er_diagnoses / successful_dates if successful_dates > 0 else 0,
                'avg_admission_diagnoses_per_day': total_admission_diagnoses / successful_dates if successful_dates > 0 else 0,
                'results': batch_results
            }
            
            self.logger.info(
                f"Batch diagnosis generation completed: "
                f"{total_er_diagnoses} ER diagnoses, {total_admission_diagnoses} admission diagnoses"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Batch diagnosis generation failed: {e}")
            return {'success': False, 'error': str(e)}