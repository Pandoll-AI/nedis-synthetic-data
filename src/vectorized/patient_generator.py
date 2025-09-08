#!/usr/bin/env python3
"""
벡터화 환자 생성기 (VectorizedPatientGenerator)

날짜 정보 없이 전체 환자 집단을 벡터화 방식으로 생성하는 모듈입니다.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


@dataclass
class PatientGenerationConfig:
    """환자 생성 설정"""
    total_records: int = 322573
    batch_size: int = 50000
    random_seed: Optional[int] = None
    memory_efficient: bool = True


class VectorizedPatientGenerator:
    """벡터화 환자 생성기"""
    
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
        
        # 확률 분포 캐싱을 위한 딕셔너리
        self._cached_distributions = {}
        self._hospital_data = None
        self._distance_matrix = None
        
    def generate_all_patients(self, gen_config: PatientGenerationConfig) -> pd.DataFrame:
        """
        전체 환자 벡터화 생성
        
        Args:
            gen_config: 생성 설정
            
        Returns:
            생성된 환자 데이터프레임 (날짜 없음)
        """
        self.logger.info(f"Starting vectorized patient generation: {gen_config.total_records:,} records")
        
        if gen_config.random_seed is not None:
            np.random.seed(gen_config.random_seed)
        
        # 필요한 데이터 로드 및 캐싱
        self._load_reference_data()
        
        if gen_config.memory_efficient and gen_config.total_records > gen_config.batch_size:
            # 메모리 효율적 처리 (청크별)
            return self._generate_patients_chunked(gen_config)
        else:
            # 전체 벡터화 처리
            return self._generate_patients_vectorized(gen_config.total_records)
    
    def _load_reference_data(self):
        """참조 데이터 로드 및 캐싱"""
        self.logger.info("Loading reference data for vectorized generation")
        
        # 병원 데이터 로드
        if self._hospital_data is None:
            self._hospital_data = self.db.fetch_dataframe("""
                SELECT emorg_cd, adr, hospname, daily_capacity_mean as capacity_beds, 
                       attractiveness_score
                FROM nedis_meta.hospital_capacity
            """)
            
        # 거리 매트릭스 로드
        if self._distance_matrix is None:
            self._distance_matrix = self.db.fetch_dataframe("""
                SELECT from_do_cd, to_emorg_cd, distance_km
                FROM nedis_meta.distance_matrix
            """).pivot(index='from_do_cd', columns='to_emorg_cd', values='distance_km')
        
        # 확률 분포들 캐싱
        self._cache_probability_distributions()
    
    def _cache_probability_distributions(self):
        """확률 분포들을 메모리에 캐싱"""
        self.logger.info("Caching probability distributions")
        
        # 인구통계 분포
        demographic_dist = self.db.fetch_dataframe("""
            SELECT pat_do_cd, pat_age_gr, pat_sex, 
                   COUNT(*) as count,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM nedis_original.nedis2017
            GROUP BY pat_do_cd, pat_age_gr, pat_sex
            ORDER BY pat_do_cd, pat_age_gr, pat_sex
        """)
        self._cached_distributions['demographics'] = demographic_dist
        
        # 내원수단 분포 (연령별)
        visit_method_dist = self.db.fetch_dataframe("""
            SELECT pat_age_gr, vst_meth,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY pat_age_gr) as probability
            FROM nedis_original.nedis2017
            WHERE vst_meth IS NOT NULL AND vst_meth != ''
            GROUP BY pat_age_gr, vst_meth
        """)
        self._cached_distributions['visit_method'] = visit_method_dist
        
        # 주증상 분포 (연령,성별)
        chief_complaint_dist = self.db.fetch_dataframe("""
            SELECT pat_age_gr, pat_sex, msypt,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY pat_age_gr, pat_sex) as probability
            FROM nedis_original.nedis2017
            WHERE msypt IS NOT NULL AND msypt != ''
            GROUP BY pat_age_gr, pat_sex, msypt
        """)
        self._cached_distributions['chief_complaint'] = chief_complaint_dist
        
        # KTAS 분포 (연령,성별,내원수단)
        ktas_dist = self.db.fetch_dataframe("""
            SELECT pat_age_gr, pat_sex, vst_meth, ktas_fstu,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY pat_age_gr, pat_sex, vst_meth) as probability
            FROM nedis_original.nedis2017
            WHERE ktas_fstu IN ('1','2','3','4','5') AND vst_meth IS NOT NULL
            GROUP BY pat_age_gr, pat_sex, vst_meth, ktas_fstu
        """)
        self._cached_distributions['ktas'] = ktas_dist
        
        # 치료결과 분포 (KTAS,연령)
        treatment_result_dist = self.db.fetch_dataframe("""
            SELECT ktas_fstu, pat_age_gr, emtrt_rust,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY ktas_fstu, pat_age_gr) as probability
            FROM nedis_original.nedis2017
            WHERE ktas_fstu IN ('1','2','3','4','5') AND emtrt_rust IS NOT NULL
            GROUP BY ktas_fstu, pat_age_gr, emtrt_rust
        """)
        self._cached_distributions['treatment_result'] = treatment_result_dist
        
        # 치료과 분포 (연령,성별)
        department_dist = self.db.fetch_dataframe("""
            SELECT pat_age_gr, pat_sex, main_trt_p,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY pat_age_gr, pat_sex) as probability
            FROM nedis_original.nedis2017
            WHERE main_trt_p IS NOT NULL AND main_trt_p != ''
            GROUP BY pat_age_gr, pat_sex, main_trt_p
        """)
        self._cached_distributions['department'] = department_dist
    
    def _generate_patients_chunked(self, gen_config: PatientGenerationConfig) -> pd.DataFrame:
        """청크별 환자 생성 (메모리 효율적)"""
        self.logger.info(f"Generating patients in chunks of {gen_config.batch_size:,}")
        
        chunks = []
        remaining = gen_config.total_records
        chunk_id = 0
        
        while remaining > 0:
            chunk_size = min(gen_config.batch_size, remaining)
            self.logger.info(f"Processing chunk {chunk_id + 1}: {chunk_size:,} records")
            
            chunk_df = self._generate_patients_vectorized(chunk_size)
            chunks.append(chunk_df)
            
            remaining -= chunk_size
            chunk_id += 1
        
        # 청크들을 합병
        self.logger.info("Merging chunks")
        result_df = pd.concat(chunks, ignore_index=True)
        
        # 메모리 정리
        del chunks
        
        return result_df
    
    def _generate_patients_vectorized(self, total_records: int) -> pd.DataFrame:
        """벡터화 환자 생성 (핵심 로직)"""
        self.logger.info(f"Vectorized generation of {total_records:,} patients")
        
        # Stage 1: 인구통계 벡터 생성
        demographics_df = self._generate_demographics_vectorized(total_records)
        
        # Stage 2: 초기 병원 할당 (중력 모델)
        hospital_assignments = self._generate_hospital_assignments_vectorized(demographics_df)
        demographics_df['initial_hospital'] = hospital_assignments
        
        # Stage 3: 독립적 임상 속성 생성 (완전 벡터화)
        clinical_attrs = self._generate_independent_clinical_attributes(demographics_df)
        
        # Stage 4: 조건부 임상 속성 생성 (Semi-벡터화)
        conditional_attrs = self._generate_conditional_clinical_attributes(demographics_df, clinical_attrs)
        
        # 모든 속성 합병
        result_df = pd.concat([demographics_df, clinical_attrs, conditional_attrs], axis=1)
        
        # 고유 키 생성
        result_df['index_key'] = [f"SYNTH_{i:08d}" for i in range(len(result_df))]
        result_df['pat_reg_no'] = [f"P{i:08d}" for i in range(len(result_df))]
        
        self.logger.info(f"Generated {len(result_df):,} patients successfully")
        return result_df
    
    def _generate_demographics_vectorized(self, total_records: int) -> pd.DataFrame:
        """인구통계 벡터 생성"""
        demo_dist = self._cached_distributions['demographics']
        
        # 다항분포 샘플링
        demo_counts = np.random.multinomial(
            total_records, 
            demo_dist['probability'].values
        )
        
        # 결과 조합
        demographics_list = []
        for i, (_, row) in enumerate(demo_dist.iterrows()):
            count = demo_counts[i]
            if count > 0:
                demographics_list.extend([{
                    'pat_do_cd': row['pat_do_cd'],
                    'pat_age_gr': row['pat_age_gr'],
                    'pat_sex': row['pat_sex']
                }] * count)
        
        # 셔플링 (랜덤성 증가)
        np.random.shuffle(demographics_list)
        
        return pd.DataFrame(demographics_list)
    
    def _generate_hospital_assignments_vectorized(self, demographics_df: pd.DataFrame) -> np.ndarray:
        """중력 모델 기반 초기 병원 할당"""
        self.logger.info("Vectorized hospital assignment using gravity model")
        
        # 지역별로 그룹화하여 병원 선택 확률 계산
        hospital_assignments = np.empty(len(demographics_df), dtype='object')
        
        for region in demographics_df['pat_do_cd'].unique():
            region_mask = demographics_df['pat_do_cd'] == region
            region_indices = np.where(region_mask)[0]
            
            if len(region_indices) == 0:
                continue
                
            # 해당 지역의 병원별 매력도 계산 (중력 모델)
            region_hospitals = self._hospital_data.copy()
            
            # 거리 기반 매력도 계산
            if region in self._distance_matrix.index:
                distances = self._distance_matrix.loc[region].dropna()
                
                # 중력 모델: Attractiveness = Size^alpha / Distance^beta
                alpha = 1.0  # 규모 파라미터
                beta = 2.0   # 거리 감쇠 파라미터
                
                attractiveness = {}
                for hospital in distances.index:
                    if hospital in region_hospitals['emorg_cd'].values:
                        hospital_size = region_hospitals[region_hospitals['emorg_cd'] == hospital]['capacity_beds'].iloc[0]
                        distance = distances[hospital]
                        
                        # 최소 거리 제한 (0으로 나누기 방지)
                        distance = max(distance, 0.1)
                        
                        attractiveness[hospital] = (hospital_size ** alpha) / (distance ** beta)
                
                # 확률 정규화
                total_attractiveness = sum(attractiveness.values())
                if total_attractiveness > 0:
                    hospital_probs = np.array([attractiveness.get(h, 0) for h in attractiveness.keys()])
                    hospital_probs = hospital_probs / hospital_probs.sum()
                    hospital_list = list(attractiveness.keys())
                    
                    # 벡터화 샘플링
                    chosen_hospitals = np.random.choice(
                        hospital_list,
                        size=len(region_indices),
                        p=hospital_probs
                    )
                    
                    hospital_assignments[region_indices] = chosen_hospitals
        
        return hospital_assignments
    
    def _generate_independent_clinical_attributes(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """독립적 임상 속성 생성 (완전 벡터화)"""
        self.logger.info("Generating independent clinical attributes")
        
        result_attrs = {}
        
        # 내원수단 생성
        result_attrs['vst_meth'] = self._vectorized_attribute_sampling(
            demographics_df, 
            self._cached_distributions['visit_method'],
            group_cols=['pat_age_gr'],
            target_col='vst_meth'
        )
        
        # 주증상 생성
        result_attrs['msypt'] = self._vectorized_attribute_sampling(
            demographics_df,
            self._cached_distributions['chief_complaint'],
            group_cols=['pat_age_gr', 'pat_sex'],
            target_col='msypt'
        )
        
        # 주요치료과 생성
        result_attrs['main_trt_p'] = self._vectorized_attribute_sampling(
            demographics_df,
            self._cached_distributions['department'],
            group_cols=['pat_age_gr', 'pat_sex'],
            target_col='main_trt_p'
        )
        
        return pd.DataFrame(result_attrs)
    
    def _generate_conditional_clinical_attributes(self, demographics_df: pd.DataFrame, 
                                                 clinical_attrs: pd.DataFrame) -> pd.DataFrame:
        """조건부 임상 속성 생성 (Semi-벡터화)"""
        self.logger.info("Generating conditional clinical attributes")
        
        # 기본 데이터 결합
        combined_df = pd.concat([demographics_df, clinical_attrs], axis=1)
        
        # KTAS 생성 (내원수단 의존)
        ktas_values = self._semi_vectorized_ktas_generation(combined_df)
        
        # 치료결과 생성 (KTAS 의존)
        treatment_results = self._semi_vectorized_treatment_result_generation(combined_df, ktas_values)
        
        return pd.DataFrame({
            'ktas_fstu': ktas_values,
            'ktas01': [int(k) if k.isdigit() else 3 for k in ktas_values],
            'emtrt_rust': treatment_results
        })
    
    def _vectorized_attribute_sampling(self, demographics_df: pd.DataFrame, 
                                     distribution_df: pd.DataFrame,
                                     group_cols: list, target_col: str) -> np.ndarray:
        """벡터화된 속성 샘플링"""
        results = np.empty(len(demographics_df), dtype='object')
        
        # 그룹별로 처리
        for group_values, group_indices in demographics_df.groupby(group_cols).groups.items():
            if isinstance(group_values, str):
                group_values = [group_values]
            
            # 해당 그룹의 확률 분포 찾기
            query_conditions = []
            for i, col in enumerate(group_cols):
                query_conditions.append(f"{col} == '{group_values[i]}'")
            
            query_str = " & ".join(query_conditions)
            group_dist = distribution_df.query(query_str) if query_conditions else distribution_df
            
            if len(group_dist) > 0:
                # 확률 정규화
                probs = group_dist['probability'].values
                probs = probs / probs.sum()
                
                # 벡터화 샘플링
                chosen_values = np.random.choice(
                    group_dist[target_col].values,
                    size=len(group_indices),
                    p=probs
                )
                
                results[group_indices] = chosen_values
            else:
                # 기본값 설정
                default_value = self._get_default_value(target_col)
                results[group_indices] = default_value
        
        return results
    
    def _semi_vectorized_ktas_generation(self, combined_df: pd.DataFrame) -> np.ndarray:
        """Semi-벡터화 KTAS 생성"""
        results = np.empty(len(combined_df), dtype='object')
        ktas_dist = self._cached_distributions['ktas']
        
        # 조건부 그룹별로 배치 처리
        group_cols = ['pat_age_gr', 'pat_sex', 'vst_meth']
        
        for group_values, group_indices in combined_df.groupby(group_cols).groups.items():
            # 해당 조건의 KTAS 확률 분포
            condition_df = ktas_dist.query(
                f"pat_age_gr == '{group_values[0]}' & " +
                f"pat_sex == '{group_values[1]}' & " +
                f"vst_meth == '{group_values[2]}'"
            )
            
            if len(condition_df) > 0:
                probs = condition_df['probability'].values
                probs = probs / probs.sum()
                
                chosen_ktas = np.random.choice(
                    condition_df['ktas_fstu'].values,
                    size=len(group_indices),
                    p=probs
                )
                
                results[group_indices] = chosen_ktas
            else:
                # 기본 KTAS 분포 사용
                results[group_indices] = np.random.choice(['3', '4', '5'], len(group_indices), p=[0.3, 0.5, 0.2])
        
        return results
    
    def _semi_vectorized_treatment_result_generation(self, combined_df: pd.DataFrame, 
                                                   ktas_values: np.ndarray) -> np.ndarray:
        """Semi-벡터화 치료결과 생성"""
        results = np.empty(len(combined_df), dtype='object')
        treatment_dist = self._cached_distributions['treatment_result']
        
        # KTAS별 배치 처리
        for ktas_level in ['1', '2', '3', '4', '5']:
            ktas_mask = ktas_values == ktas_level
            ktas_indices = np.where(ktas_mask)[0]
            
            if len(ktas_indices) == 0:
                continue
            
            # 해당 KTAS 레벨의 환자들을 연령별로 세분화
            ktas_patients_df = combined_df.iloc[ktas_indices]
            
            for age_group in ktas_patients_df['pat_age_gr'].unique():
                age_mask = ktas_patients_df['pat_age_gr'] == age_group
                final_indices = ktas_indices[age_mask]
                
                # 조건부 확률 분포
                condition_df = treatment_dist.query(
                    f"ktas_fstu == '{ktas_level}' & pat_age_gr == '{age_group}'"
                )
                
                if len(condition_df) > 0:
                    probs = condition_df['probability'].values
                    probs = probs / probs.sum()
                    
                    chosen_results = np.random.choice(
                        condition_df['emtrt_rust'].values,
                        size=len(final_indices),
                        p=probs
                    )
                    
                    results[final_indices] = chosen_results
                else:
                    # KTAS별 기본 분포
                    default_dist = self._get_default_treatment_distribution(ktas_level)
                    results[final_indices] = np.random.choice(
                        list(default_dist.keys()),
                        len(final_indices),
                        p=list(default_dist.values())
                    )
        
        return results
    
    def _get_default_value(self, attribute: str) -> str:
        """속성별 기본값 반환"""
        defaults = {
            'vst_meth': '11',  # 도보
            'msypt': 'R50',    # 일반적인 증상
            'main_trt_p': '01', # 내과
            'ktas_fstu': '3',  # 보통 응급
            'emtrt_rust': '11' # 귀가
        }
        return defaults.get(attribute, '')
    
    def _get_default_treatment_distribution(self, ktas_level: str) -> Dict[str, float]:
        """KTAS별 기본 치료결과 분포"""
        distributions = {
            '1': {'31': 0.7, '32': 0.2, '12': 0.1},  # 소생급 - 대부분 입원
            '2': {'31': 0.5, '32': 0.2, '11': 0.3},  # 응급급
            '3': {'31': 0.3, '11': 0.7},             # 긴급급
            '4': {'11': 0.8, '31': 0.2},             # 준응급급 - 대부분 귀가
            '5': {'11': 0.9, '31': 0.1}              # 비응급급 - 거의 귀가
        }
        return distributions.get(ktas_level, {'11': 0.8, '31': 0.2})