#!/usr/bin/env python3
"""
벡터화 환자 생성기 (VectorizedPatientGenerator)

동적 패턴 분석을 사용하여 하드코딩 없이 환자를 생성하는 모듈입니다.
계층적 대안 및 지역 기반 병원 할당을 포함합니다.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass

from ..core.database import DatabaseManager
from ..core.config import ConfigManager
from ..analysis.pattern_analyzer import PatternAnalyzer, PatternAnalysisConfig


@dataclass
class PatientGenerationConfig:
    """환자 생성 설정"""
    total_records: int = 322573
    batch_size: int = 50000
    random_seed: Optional[int] = None
    memory_efficient: bool = True


class VectorizedPatientGenerator:
    """동적 패턴 기반 벡터화 환자 생성기"""
    
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
        
        # 동적 패턴 데이터 저장
        self._dynamic_patterns = None
        self._hospital_data = None
        # 하드코딩된 분포 제거: 모든 분포는 동적 패턴에서 로드되며,
        # 패턴이 부족한 경우 ConfigManager의 설정을 통해 폴백됩니다.

        # 원본 소스 테이블(연도) 선택: config 'original.source_table' 또는 'original.year'
        table_from_config = self.config.get('original.source_table')
        if table_from_config and isinstance(table_from_config, str):
            self.src_table = table_from_config
        else:
            year = self.config.get('original.year')
            if isinstance(year, int):
                self.src_table = f"nedis_original.nedis{year}"
            else:
                self.src_table = "nedis_original.emihptmi"
        
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
        """동적 패턴 분석 및 데이터 로드"""
        self.logger.info("Loading dynamic patterns for vectorized generation")
        
        # 병원 데이터 로드 (용량 및 유형 정보 포함)
        if self._hospital_data is None:
            try:
                self._hospital_data = self.db.fetch_dataframe("""
                    SELECT 
                        ptmiemcd, 
                        adr, 
                        hospname, 
                        daily_capacity_mean as capacity_beds,
                        CASE 
                            WHEN daily_capacity_mean >= 300 THEN 'large'
                            WHEN daily_capacity_mean >= 100 THEN 'medium' 
                            ELSE 'small'
                        END as hospital_type,
                        SUBSTR(adr, 1, 2) as major_region,
                        adr as region_code
                    FROM nedis_meta.hospital_capacity
                    WHERE daily_capacity_mean IS NOT NULL
                """)
            except Exception:
                # 시도 2: 원본 DB가 ATTACH 되어있는 경우(alias 파생)
                alias = None
                src_table = self.pattern_analyzer._patterns_cache.get('src_table') if hasattr(self.pattern_analyzer, '_patterns_cache') else None
                # 대신 config에서 추출
                st = self.config.get('original.source_table')
                if isinstance(st, str) and '.' in st:
                    alias = st.split('.')[0]
                if alias:
                    try:
                        self._hospital_data = self.db.fetch_dataframe(f"""
                            SELECT 
                                ptmiemcd, 
                                adr, 
                                hospname, 
                                daily_capacity_mean as capacity_beds,
                                CASE 
                                    WHEN daily_capacity_mean >= 300 THEN 'large'
                                    WHEN daily_capacity_mean >= 100 THEN 'medium' 
                                    ELSE 'small'
                                END as hospital_type,
                                SUBSTR(adr, 1, 2) as major_region,
                                adr as region_code
                            FROM {alias}.nedis_meta.hospital_capacity
                            WHERE daily_capacity_mean IS NOT NULL
                        """)
                    except Exception:
                        self._hospital_data = None
                else:
                    self._hospital_data = None
        
        # 동적 패턴 분석 수행
        if self._dynamic_patterns is None:
            self.logger.info("Performing dynamic pattern analysis")
            self._dynamic_patterns = self.pattern_analyzer.analyze_all_patterns()
            
            # 패턴 분석 결과 로깅
            self.logger.info(f"Dynamic pattern analysis completed:")
            for pattern_type, pattern_data in self._dynamic_patterns.items():
                if isinstance(pattern_data, dict) and 'patterns' in pattern_data:
                    count = len(pattern_data['patterns'])
                    self.logger.info(f"  - {pattern_type}: {count} patterns")
                elif pattern_type == 'metadata':
                    self.logger.info(f"  - Analysis timestamp: {pattern_data.get('analysis_timestamp', 'N/A')}")
    
    def _get_dynamic_distribution(self, pattern_type: str) -> pd.DataFrame:
        """동적 패턴에서 분포 데이터 추출"""
        if self._dynamic_patterns is None:
            raise RuntimeError("Dynamic patterns not loaded. Call _load_reference_data() first.")
        
        pattern_data = self._dynamic_patterns.get(pattern_type, {})
        if 'patterns' not in pattern_data:
            self.logger.warning(f"No patterns found for {pattern_type}")
            return pd.DataFrame()
        
        # 패턴 데이터를 DataFrame 형태로 변환 (공통 포맷)
        patterns = pattern_data['patterns']
        rows = []
        
        if pattern_type == 'demographic_patterns':
            for key, data in patterns.items():
                age_sex = key.split('_')
                if len(age_sex) == 2:
                    rows.append({
                        'ptmibrtd': age_sex[0],
                        'ptmisexx': age_sex[1],
                        'count': data['count'],
                        'probability': data['probability']
                    })
        elif pattern_type == 'visit_method_patterns':
            for key, value_map in patterns.items():
                for ptmiinmn, stats in value_map.items():
                    rows.append({
                        'ptmibrtd': key,
                        'ptmiinmn': ptmiinmn,
                        'probability': stats['probability']
                    })
        elif pattern_type == 'chief_complaint_patterns':
            for key, value_map in patterns.items():
                age, sex = key.split('_') if '_' in key else (key, '')
                for ptmimnsy, stats in value_map.items():
                    rows.append({
                        'ptmibrtd': age,
                        'ptmisexx': sex,
                        'ptmimnsy': ptmimnsy,
                        'probability': stats['probability']
                    })
        elif pattern_type == 'department_patterns':
            for key, value_map in patterns.items():
                age, sex = key.split('_') if '_' in key else (key, '')
                for ptmidept, stats in value_map.items():
                    rows.append({
                        'ptmibrtd': age,
                        'ptmisexx': sex,
                        'ptmidept': ptmidept,
                        'probability': stats['probability']
                    })
        elif pattern_type == 'treatment_result_patterns':
            for key, value_map in patterns.items():
                ktas, age = key.split('_') if '_' in key else (key, '')
                for ptmiemrt, stats in value_map.items():
                    rows.append({
                        'ptmikts1': ktas,
                        'ptmibrtd': age,
                        'ptmiemrt': ptmiemrt,
                        'probability': stats['probability']
                    })
        
        return pd.DataFrame(rows) if rows else pd.DataFrame()
    
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

        # 전역 고유 키 재생성 (중복 방지)
        result_df['index_key'] = [f"SYNTH_{i:08d}" for i in range(len(result_df))]
        result_df['ptmiidno'] = [f"P{i:08d}" for i in range(len(result_df))]

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

        # Stage 5: 결측값 적용 (원본 결측 패턴 재현)
        result_df = self._apply_missing_values(result_df)

        # 고유 키 생성
        result_df['index_key'] = [f"SYNTH_{i:08d}" for i in range(len(result_df))]
        result_df['ptmiidno'] = [f"P{i:08d}" for i in range(len(result_df))]

        self.logger.info(f"Generated {len(result_df):,} patients successfully")
        return result_df
    
    def _apply_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """원본 데이터의 구조적 결측 패턴을 재현.

        KTAS·주증상·진료과는 동시 결측(co-missing) 구조를 가지므로,
        1) 먼저 KTAS 결측 여부를 결정 (primary_missing_rate)
        2) KTAS 결측 시 주증상·진료과도 함께 결측 처리
        3) KTAS 유효 시에도 소수의 독립 결측 적용
        4) 기타 변수(ptmizipc, ptmiemrt 등)는 독립 결측 적용
        """
        missing_rates = self._dynamic_patterns.get('missing_value_rates', {})
        if not missing_rates or 'error' in missing_rates:
            self.logger.warning("No missing value rate data available; skipping")
            return df

        corr = missing_rates.get('correlated_group', {})
        cond = missing_rates.get('conditional', {}).get('given_ktas_valid', {})
        indep = missing_rates.get('independent', {})
        n = len(df)
        rng = np.random.RandomState(hash('missing') % (2**31))

        # --- Ensure string dtype for columns that may receive '-' ---
        str_cols = ['ptmikts1', 'ptmikpr1', 'ptmimnsy', 'ptmidept',
                    'ptmizipc', 'ptmiemrt', 'ptmiinmn']
        for col in str_cols:
            if col in df.columns and df[col].dtype != object:
                df[col] = df[col].astype(str)

        # --- Correlated group: KTAS → symptom + department ---
        primary_rate = corr.get('primary_missing_rate', 0.0)
        if primary_rate > 0 and 'ptmikts1' in df.columns:
            mask_primary = rng.random(n) < primary_rate
            n_primary = mask_primary.sum()
            self.logger.info(
                "Applying correlated missing: %d/%d (%.1f%%) records",
                n_primary, n, n_primary / n * 100,
            )
            df.loc[mask_primary, 'ptmikts1'] = '-'
            if 'ptmikpr1' in df.columns:
                df.loc[mask_primary, 'ptmikpr1'] = '-'
            if 'ptmimnsy' in df.columns:
                df.loc[mask_primary, 'ptmimnsy'] = '-'
            if 'ptmidept' in df.columns:
                df.loc[mask_primary, 'ptmidept'] = '-'

            # When KTAS is valid, apply residual independent missing
            # Only for correlated group members (ptmimnsy, ptmidept)
            valid_mask = ~mask_primary
            corr_members = set(corr.get('members', []))
            for col, rate in cond.items():
                if col not in corr_members:
                    continue  # Independent vars handled separately below
                if rate > 0 and col in df.columns:
                    residual = rng.random(n) < rate
                    df.loc[valid_mask & residual, col] = '-'

        # --- Independent variables (not in correlated group) ---
        # These use the overall independent rate, not conditional
        indep_only = ['ptmizipc', 'ptmiemrt', 'ptmiinmn']
        for col in indep_only:
            rate = indep.get(col, 0.0)
            if rate > 0 and col in df.columns:
                mask = rng.random(n) < rate
                if mask.any():
                    df.loc[mask, col] = '-'

        return df

    def _generate_demographics_vectorized(self, total_records: int) -> pd.DataFrame:
        """동적 패턴을 사용한 인구통계 벡터 생성"""
        self.logger.info("Generating demographics using dynamic patterns")
        
        # 원본 데이터에서 직접 인구통계 분포 추출
        demo_query = """
            SELECT ptmizipc, ptmibrtd, ptmisexx, 
                   COUNT(*) as count,
                   COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM {src}
            WHERE ptmizipc IS NOT NULL 
              AND ptmibrtd IS NOT NULL 
              AND ptmisexx IS NOT NULL
            GROUP BY ptmizipc, ptmibrtd, ptmisexx
            ORDER BY ptmizipc, ptmibrtd, ptmisexx
        """
        
        demo_dist = self.db.fetch_dataframe(demo_query.format(src=self.src_table))
        
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
                    'ptmizipc': row['ptmizipc'],
                    'ptmibrtd': row['ptmibrtd'],
                    'ptmisexx': row['ptmisexx']
                }] * count)
        
        # 셔플링 (랜덤성 증가)
        np.random.shuffle(demographics_list)
        
        return pd.DataFrame(demographics_list)
    
    def _generate_hospital_assignments_vectorized(self, demographics_df: pd.DataFrame) -> np.ndarray:
        """동적 패턴 기반 지역별 병원 할당"""
        self.logger.info("Vectorized hospital assignment using dynamic regional patterns")
        
        # 병원 할당 패턴 가져오기
        hospital_patterns = self._dynamic_patterns.get('hospital_allocation', {})
        if 'patterns' not in hospital_patterns:
            self.logger.warning("No hospital allocation patterns found, using fallback method")
            return self._fallback_hospital_assignment(demographics_df)
        
        allocation_patterns = hospital_patterns['patterns']
        hospital_assignments = np.empty(len(demographics_df), dtype='object')
        
        # 지역별로 그룹화하여 병원 선택
        for region in demographics_df['ptmizipc'].unique():
            region_mask = demographics_df['ptmizipc'] == region
            region_indices = np.where(region_mask)[0]
            
            if len(region_indices) == 0:
                continue
            
            # 지역 코드별 병원 패턴 조회
            region_pattern = allocation_patterns.get(str(region))
            
            if region_pattern and 'hospitals' in region_pattern:
                # 해당 지역의 병원 확률 분포 사용
                hospitals = list(region_pattern['hospitals'].keys())
                probabilities = [region_pattern['hospitals'][h]['probability'] 
                               for h in hospitals]
                
                # 확률 정규화
                if sum(probabilities) > 0:
                    probabilities = np.array(probabilities)
                    probabilities = probabilities / probabilities.sum()
                    
                    # 벡터화 샘플링
                    chosen_hospitals = np.random.choice(
                        hospitals,
                        size=len(region_indices),
                        p=probabilities
                    )
                    hospital_assignments[region_indices] = chosen_hospitals
                else:
                    # 확률이 0인 경우 균등 분포
                    chosen_hospitals = np.random.choice(
                        hospitals,
                        size=len(region_indices)
                    )
                    hospital_assignments[region_indices] = chosen_hospitals
            else:
                # 해당 지역 패턴이 없는 경우 계층적 대안 사용
                major_region = str(region)[:2] if len(str(region)) >= 2 else str(region)
                hierarchical_patterns = hospital_patterns.get('hierarchical_fallback', {})
                
                if major_region in hierarchical_patterns:
                    major_pattern = hierarchical_patterns[major_region]
                    hospitals = list(major_pattern['hospitals'].keys())
                    probabilities = [major_pattern['hospitals'][h]['probability'] 
                                   for h in hospitals]
                    
                    if sum(probabilities) > 0:
                        probabilities = np.array(probabilities)
                        probabilities = probabilities / probabilities.sum()
                        
                        chosen_hospitals = np.random.choice(
                            hospitals,
                            size=len(region_indices),
                            p=probabilities
                        )
                        hospital_assignments[region_indices] = chosen_hospitals
                    else:
                        # 최종 대안: 랜덤 병원 선택
                        hospital_assignments[region_indices] = self._assign_random_hospitals(region_indices)
                else:
                    # 최종 대안: 랜덤 병원 선택
                    hospital_assignments[region_indices] = self._assign_random_hospitals(region_indices)
        
        return hospital_assignments
    
    def _fallback_hospital_assignment(self, demographics_df: pd.DataFrame) -> np.ndarray:
        """대안 병원 할당 방법 (패턴이 없는 경우)"""
        self.logger.info("Using fallback hospital assignment method")
        
        # 지역별로 가능한 병원 목록 생성
        region_hospitals = {}
        
        for _, hospital in self._hospital_data.iterrows():
            region = hospital['major_region']  # 2자리 지역코드 사용
            if region not in region_hospitals:
                region_hospitals[region] = []
            region_hospitals[region].append(hospital['ptmiemcd'])
        
        hospital_assignments = np.empty(len(demographics_df), dtype='object')
        
        for region in demographics_df['ptmizipc'].unique():
            region_mask = demographics_df['ptmizipc'] == region
            region_indices = np.where(region_mask)[0]
            
            # 지역에 해당하는 병원들 찾기
            major_region = str(region)[:2] if len(str(region)) >= 2 else str(region)
            available_hospitals = region_hospitals.get(major_region, [])
            
            if available_hospitals:
                # 해당 지역 병원 중에서 랜덤 선택
                chosen_hospitals = np.random.choice(
                    available_hospitals,
                    size=len(region_indices)
                )
                hospital_assignments[region_indices] = chosen_hospitals
            else:
                # 전체 병원에서 랜덤 선택
                hospital_assignments[region_indices] = self._assign_random_hospitals(region_indices)
        
        return hospital_assignments
    
    def _assign_random_hospitals(self, indices: np.ndarray) -> np.ndarray:
        """랜덤 병원 할당"""
        all_hospitals = self._hospital_data['ptmiemcd'].tolist()
        if all_hospitals:
            return np.random.choice(all_hospitals, size=len(indices))
        else:
            # 기본값으로 더미 병원 코드 사용
            return np.array(['DEFAULT_HOSPITAL'] * len(indices))
    
    def _generate_independent_clinical_attributes(self, demographics_df: pd.DataFrame) -> pd.DataFrame:
        """독립적 임상 속성 생성 (내원수단만)"""
        self.logger.info("Generating independent clinical attributes")

        result_attrs = {}

        visit_method_dist = self._get_dynamic_distribution('visit_method_patterns')

        # 내원수단 생성
        result_attrs['ptmiinmn'] = self._vectorized_attribute_sampling(
            demographics_df,
            visit_method_dist,
            group_cols=['ptmibrtd'],
            target_col='ptmiinmn'
        )

        return pd.DataFrame(result_attrs)
    
    def _generate_conditional_clinical_attributes(self, demographics_df: pd.DataFrame,
                                                 clinical_attrs: pd.DataFrame) -> pd.DataFrame:
        """조건부 임상 속성 생성: KTAS → 주증상 → 진료과 → 치료결과"""
        self.logger.info("Generating conditional clinical attributes")

        combined_df = pd.concat([demographics_df, clinical_attrs], axis=1)

        # 1. KTAS 생성 (지역/병원유형 의존)
        ktas_values = self._semi_vectorized_ktas_generation(combined_df)

        # 2. 주증상 생성: P(ptmimnsy | ptmikts1, ptmibrtd, ptmisexx)
        self.logger.info("Generating chief complaints conditional on KTAS")
        symptom_values = self._conditional_symptom_generation(
            combined_df, ktas_values)

        # 3. 진료과 생성: P(ptmidept | ptmikts1, ptmimnsy)
        self.logger.info("Generating departments conditional on KTAS + symptom")
        department_values = self._conditional_department_generation(
            ktas_values, symptom_values)

        # 4. 치료결과 생성 (KTAS 의존)
        treatment_results = self._semi_vectorized_treatment_result_generation(
            combined_df, ktas_values)

        # 5. KTAS 프로토콜 코드 생성: 5자리 코드 + 감염코드(1자리) = 6자리
        kpr1_values = self._generate_ktas_protocol_codes(
            ktas_values, symptom_values, combined_df['ptmibrtd'].values)

        return pd.DataFrame({
            'ptmikts1': ktas_values,
            'ptmikpr1': kpr1_values,
            'ptmimnsy': symptom_values,
            'ptmidept': department_values,
            'ptmiemrt': treatment_results,
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
                # 폴백: 설정 기반 기본값 사용 (하드코딩 제거)
                default_map = self.config.get('fallback.distributions', {}).get(target_col, None)
                if isinstance(default_map, dict) and len(default_map) > 0:
                    keys = list(default_map.keys())
                    vals = np.array(list(default_map.values()), dtype=float)
                    vals = vals / vals.sum()
                    results[group_indices] = np.random.choice(keys, size=len(group_indices), p=vals)
                else:
                    # 최종 폴백: 관측값 없이 빈 값
                    results[group_indices] = ''
        
        return results
    
    def _semi_vectorized_ktas_generation(self, combined_df: pd.DataFrame) -> np.ndarray:
        """계층적 대안을 사용한 KTAS 생성 (배치/그룹 벡터화)"""
        self.logger.info("Generating KTAS using hierarchical fallback patterns (batch)")
        results = np.empty(len(combined_df), dtype='object')

        # 병원 유형 정보 추가
        combined_df = self._add_hospital_type_info(combined_df)

        # 그룹별로 KTAS 배치 샘플링
        group_cols = ['ptmizipc', 'hospital_type']
        for group_values, group_indices in combined_df.groupby(group_cols).groups.items():
            region_code, hospital_type = group_values if isinstance(group_values, tuple) else (group_values, 'medium')

            # 계층적 KTAS 분포 조회
            ktas_probs = self.pattern_analyzer.get_hierarchical_ktas_distribution(
                str(region_code), str(hospital_type)
            )

            if ktas_probs and sum(ktas_probs.values()) > 0:
                ktas_levels = list(ktas_probs.keys())
                probabilities = np.array(list(ktas_probs.values()), dtype=float)
                probabilities = probabilities / probabilities.sum()
                results[group_indices] = np.random.choice(ktas_levels, size=len(group_indices), p=probabilities)
            else:
                # 폴백: 설정에서 기본 KTAS 분포 사용
                default_map = self.config.get('fallback.distributions', {}).get('ptmikts1', {'3': 0.3, '4': 0.5, '5': 0.2})
                levels = list(default_map.keys())
                probs = np.array(list(default_map.values()), dtype=float)
                probs = probs / probs.sum()
                results[group_indices] = np.random.choice(levels, size=len(group_indices), p=probs)

        return results
    
    def _add_hospital_type_info(self, df: pd.DataFrame) -> pd.DataFrame:
        """병원 유형 정보 추가"""
        # 병원 유형 매핑 생성
        hospital_type_map = {}
        if self._hospital_data is not None and len(self._hospital_data) > 0:
            for _, hospital in self._hospital_data.iterrows():
                hospital_type_map[hospital['ptmiemcd']] = hospital['hospital_type']
        
        # DataFrame에 병원 유형 추가
        df = df.copy()
        df['hospital_type'] = df['initial_hospital'].map(hospital_type_map).fillna('medium')
        
        return df
    
    def _get_hospital_type(self, hospital_code: str) -> str:
        """병원 코드로 병원 유형 조회"""
        if self._hospital_data is None or len(self._hospital_data) == 0:
            return 'medium'
        hospital_info = self._hospital_data[self._hospital_data['ptmiemcd'] == hospital_code]
        
        if len(hospital_info) > 0:
            return hospital_info.iloc[0]['hospital_type']
        else:
            return 'medium'  # 기본값
    
    def _sample_from_pattern(self, pattern_dict: Dict, size: int,
                            rng: np.random.RandomState) -> np.ndarray:
        """패턴 딕셔너리에서 확률적 샘플링"""
        keys = list(pattern_dict.keys())
        probs = np.array([v['probability'] for v in pattern_dict.values()])
        probs = probs / probs.sum()
        return rng.choice(keys, size=size, p=probs)

    def _conditional_symptom_generation(self, combined_df: pd.DataFrame,
                                        ktas_values: np.ndarray) -> np.ndarray:
        """KTAS/연령/성별 조건부 주증상 생성.

        계층적 대안: ktas_age_sex → ktas_age → ktas → age_sex
        """
        n = len(combined_df)
        results = np.empty(n, dtype='object')
        results[:] = ''

        patterns = self._dynamic_patterns.get('chief_complaint_patterns', {}).get('patterns', {})
        if not patterns:
            self.logger.warning("No chief complaint patterns; returning empty")
            return results

        rng = np.random.RandomState(hash('symptom') % (2**31))
        ages = combined_df['ptmibrtd'].astype(str).values
        sexes = combined_df['ptmisexx'].astype(str).values

        # Build index per (ktas, age, sex)
        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[(ktas_values[i], ages[i], sexes[i])].append(i)

        for (ktas, age, sex), indices in groups.items():
            idx_arr = np.array(indices)
            # Hierarchical fallback
            for key in [f"{ktas}_{age}_{sex}", f"{ktas}_{age}",
                        f"{ktas}", f"all_{age}_{sex}"]:
                if key in patterns and len(patterns[key]) > 0:
                    results[idx_arr] = self._sample_from_pattern(
                        patterns[key], len(idx_arr), rng)
                    break

        return results

    def _conditional_department_generation(self, ktas_values: np.ndarray,
                                           symptom_values: np.ndarray) -> np.ndarray:
        """KTAS/주증상 조건부 진료과 생성.

        계층적 대안: ktas_symptom → ktas → sym_symptom → all
        """
        n = len(ktas_values)
        results = np.empty(n, dtype='object')
        results[:] = ''

        patterns = self._dynamic_patterns.get('department_patterns', {}).get('patterns', {})
        if not patterns:
            self.logger.warning("No department patterns; returning empty")
            return results

        rng = np.random.RandomState(hash('department') % (2**31))

        from collections import defaultdict
        groups = defaultdict(list)
        for i in range(n):
            groups[(ktas_values[i], symptom_values[i])].append(i)

        for (ktas, sym), indices in groups.items():
            idx_arr = np.array(indices)
            for key in [f"{ktas}_{sym}", f"{ktas}", f"sym_{sym}", "all"]:
                if key in patterns and len(patterns[key]) > 0:
                    results[idx_arr] = self._sample_from_pattern(
                        patterns[key], len(idx_arr), rng)
                    break

        return results

    # KTAS 대분류: 주증상(UMLS) → 대분류 코드 매핑
    # 실제 KTAS 프로토콜에 기반한 신체계통별 분류
    _SYMPTOM_MAJOR_CLASS = {
        # 심혈관계 (1)
        'C0008031': '1', 'C0235710': '1', 'C0030252': '1',
        'C0039231': '1', 'C0018790': '1', 'C0020538': '1',
        'C0020649': '1', 'C0036572': '1',
        # 신경계 (2)
        'C0018681': '2', 'C0012833': '2', 'C0233407': '2',
        'C0237477': '2', 'C0421619': '2', 'C0036917': '2',
        'C0242429': '2', 'C0151827': '2',
        # 소화기계 (3)
        'C0000737': '3', 'C0027497': '3', 'C0042963': '3',
        'C0011991': '3', 'C0232493': '3',
        # 호흡기계 (4)
        'C0013404': '4', 'C0004093': '4', 'C0239589': '4',
        'C0010200': '4', 'C0231528': '4',
        # 근골격/외상 (5)
        'C0015967': '5', 'C0016512': '5', 'C0238656': '5',
        'C0023222': '5', 'C0231749': '5', 'C0007859': '5',
        # 비뇨생식계 (6)
        'C0042109': '6', 'C0232849': '6', 'C0404555': '6',
        # 안/이비인후 (7)
        'C0015230': '7', 'C0008035': '7', 'C0432981': '7',
        # 피부/알레르기 (8)
        'C0004604': '8', 'C0039070': '8', 'C0037404': '8',
        'C0278286': '8', 'C0024031': '8',
    }

    def _generate_ktas_protocol_codes(self, ktas_values: np.ndarray,
                                       symptom_values: np.ndarray,
                                       age_values: np.ndarray) -> np.ndarray:
        """KTAS 프로토콜 코드 합성 생성.

        구조: [연령구분 1자리][대분류 1자리][중분류 1자리][소분류 2자리][감염코드 1자리] = 6자리
          - 연령구분: A(≥15세) / P(<15세) — 연령군 코드로 판단
          - 대분류: 주증상 기반 신체계통 매핑
          - 중분류+소분류: 해시 기반 결정론적 생성
          - 감염코드: 0(비감염) 90%, 1(감염) 5%, 9(미상) 5%
        """
        n = len(ktas_values)
        results = np.empty(n, dtype='object')
        rng = np.random.RandomState(hash('ktas_protocol') % (2**31))

        # 소아 연령군 코드들 (15세 미만)
        child_age_codes = {'01', '09', '10'}

        for i in range(n):
            ktas = ktas_values[i]
            if not str(ktas).isdigit():
                results[i] = '-'
                continue

            # 1단계: 연령구분
            age_code = str(age_values[i])
            cd1 = 'P' if age_code in child_age_codes else 'A'

            # 2단계: 대분류 (주증상 → 신체계통)
            sym = str(symptom_values[i])
            cd2 = self._SYMPTOM_MAJOR_CLASS.get(sym, str(rng.randint(1, 10)))

            # 3단계: 중분류 (주증상 해시 기반)
            cd3 = str((hash(sym) % 9) + 1)

            # 4단계: 소분류 (주증상+KTAS 해시 기반)
            cd4 = f"{((hash(sym + ktas) % 20) + 1):02d}"

            # 감염코드
            r = rng.random()
            infection = '0' if r < 0.90 else ('1' if r < 0.95 else '9')

            results[i] = f"{cd1}{cd2}{cd3}{cd4}{infection}"

        return results

    def _semi_vectorized_treatment_result_generation(self, combined_df: pd.DataFrame,
                                                   ktas_values: np.ndarray) -> np.ndarray:
        """Semi-벡터화 치료결과 생성 (동적 분포 기반)"""
        results = np.empty(len(combined_df), dtype='object')

        # 동적 조건부 분포 로드
        tr_dist = self._get_dynamic_distribution('treatment_result_patterns')

        # KTAS별 배치 처리
        for ktas_level in ['1', '2', '3', '4', '5']:
            ktas_mask = ktas_values == ktas_level
            ktas_indices = np.where(ktas_mask)[0]
            if len(ktas_indices) == 0:
                continue

            ktas_patients_df = combined_df.iloc[ktas_indices]
            for age_group in ktas_patients_df['ptmibrtd'].unique():
                age_mask = ktas_patients_df['ptmibrtd'] == age_group
                final_indices = ktas_indices[age_mask]
                if len(final_indices) == 0:
                    continue

                # 조건부 확률 분포
                condition_df = tr_dist.query(
                    f"ptmikts1 == '{ktas_level}' & ptmibrtd == '{age_group}'"
                )
                if len(condition_df) > 0:
                    probs = condition_df['probability'].values.astype(float)
                    probs = probs / probs.sum()
                    chosen_results = np.random.choice(
                        condition_df['ptmiemrt'].values,
                        size=len(final_indices),
                        p=probs
                    )
                    results[final_indices] = chosen_results
                else:
                    # 설정 기반 폴백 분포 (KTAS별)
                    fallback = self.config.get('fallback.distributions', {}).get('ptmiemrt_by_ktas', {})
                    default_map = fallback.get(ktas_level, {})
                    if isinstance(default_map, dict) and len(default_map) > 0:
                        keys = list(default_map.keys())
                        vals = np.array(list(default_map.values()), dtype=float)
                        vals = vals / vals.sum()
                        results[final_indices] = np.random.choice(keys, size=len(final_indices), p=vals)
                    else:
                        results[final_indices] = ''

        return results
    
    # 하드코딩된 기본값/분포 제거됨. 폴백은 설정(ConfigManager)에서 로드합니다.
