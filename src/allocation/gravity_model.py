"""
Hospital Gravity Model (Huff Model) Implementation

중력모형(Huff Model)을 사용한 병원 선택 확률 계산 및 
용량 제약을 고려한 병원 할당 시스템을 구현합니다.

Distance-based accessibility와 hospital attractiveness를 종합적으로 고려합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import math

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class HospitalGravityAllocator:
    """중력모형 기반 병원 할당 시스템"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager, gamma: float = 1.5):
        """
        중력모형 병원 할당자 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
            gamma: 거리 감쇠 파라미터 (1.0 ~ 2.5 권장)
        """
        self.db = db_manager
        self.config = config
        self.gamma = gamma
        self.logger = logging.getLogger(__name__)
        
        # 지역별 중심 좌표 (approximate centers in Korea)
        self.region_coordinates = {
            '11': (37.5665, 126.9780),  # 서울특별시
            '26': (35.1595, 129.0756),  # 부산광역시
            '27': (35.1379, 126.7358),  # 대구광역시
            '28': (37.4563, 126.7052),  # 인천광역시
            '29': (35.1598, 126.8513),  # 광주광역시
            '30': (36.3504, 127.3845),  # 대전광역시
            '31': (35.5384, 129.3114),  # 울산광역시
            '36': (36.5400, 127.2893),  # 세종특별자치시
            '41': (37.4138, 127.5183),  # 경기도
            '42': (37.8853, 127.7300),  # 강원도
            '43': (36.5184, 126.8000),  # 충청북도
            '44': (36.5184, 126.8000),  # 충청남도
            '45': (35.7175, 127.1530),  # 전라북도
            '46': (34.8679, 126.9910),  # 전라남도
            '47': (35.9078, 128.8081),  # 경상북도
            '48': (35.4606, 128.2132),  # 경상남도
            '50': (33.4996, 126.5312),  # 제주특별자치도
        }
        
        self.region_names = {
            '11': '서울특별시', '26': '부산광역시', '27': '대구광역시', 
            '28': '인천광역시', '29': '광주광역시', '30': '대전광역시',
            '31': '울산광역시', '36': '세종특별자치시', '41': '경기도',
            '42': '강원도', '43': '충청북도', '44': '충청남도',
            '45': '전라북도', '46': '전라남도', '47': '경상북도',
            '48': '경상남도', '50': '제주특별자치도'
        }
        
    def calculate_distance(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        """
        두 좌표 간 거리 계산 (Haversine formula)
        
        Args:
            coord1: 첫 번째 좌표 (위도, 경도)
            coord2: 두 번째 좌표 (위도, 경도)
            
        Returns:
            거리 (km)
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
        
        # 지구 반지름 (km)
        R = 6371.0
        
        # 라디안 변환
        lat1_r = math.radians(lat1)
        lon1_r = math.radians(lon1)
        lat2_r = math.radians(lat2)
        lon2_r = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        
        a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance = R * c
        return distance
        
    def initialize_distance_matrix(self):
        """거리 매트릭스 생성 및 저장"""
        
        self.logger.info("Initializing distance matrix between regions and hospitals")
        
        try:
            # 기존 거리 매트릭스 테이블 삭제
            self.db.execute_query("DROP TABLE IF EXISTS nedis_meta.distance_matrix")
            
            # 거리 매트릭스 테이블 생성
            self.db.execute_query("""
                CREATE TABLE nedis_meta.distance_matrix (
                    from_do_cd VARCHAR,
                    to_emorg_cd VARCHAR,
                    distance_km DOUBLE,
                    travel_time_min DOUBLE,
                    PRIMARY KEY (from_do_cd, to_emorg_cd)
                )
            """)
            
            # 병원 정보 로드
            hospitals = self.db.fetch_dataframe("""
                SELECT emorg_cd, adr, hospname 
                FROM nedis_meta.hospital_capacity
            """)
            
            # 지역별-병원별 거리 계산
            distance_records = []
            
            for region_code, region_coord in self.region_coordinates.items():
                for _, hospital in hospitals.iterrows():
                    hospital_code = hospital['emorg_cd']
                    hospital_region = hospital['adr']
                    
                    # 병원 지역 코드 매핑 (간단한 휴리스틱)
                    hospital_coord = self._get_hospital_coordinate(hospital_region, hospital_code)
                    
                    # 거리 계산
                    distance = self.calculate_distance(region_coord, hospital_coord)
                    
                    # 여행시간 추정 (거리 기반, 평균 속도 40km/h)
                    travel_time = distance / 40.0 * 60  # 분 단위
                    
                    distance_records.append({
                        'from_do_cd': region_code,
                        'to_emorg_cd': hospital_code,
                        'distance_km': distance,
                        'travel_time_min': travel_time
                    })
            
            # 배치 삽입
            distance_df = pd.DataFrame(distance_records)
            
            for _, row in distance_df.iterrows():
                self.db.execute_query("""
                    INSERT INTO nedis_meta.distance_matrix
                    (from_do_cd, to_emorg_cd, distance_km, travel_time_min)
                    VALUES (?, ?, ?, ?)
                """, (row['from_do_cd'], row['to_emorg_cd'], 
                      row['distance_km'], row['travel_time_min']))
            
            self.logger.info(f"Distance matrix created with {len(distance_records)} entries")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize distance matrix: {e}")
            raise
            
    def _get_hospital_coordinate(self, hospital_region: str, hospital_code: str) -> Tuple[float, float]:
        """
        병원 좌표 추정 (지역명 기반)
        
        Args:
            hospital_region: 병원 소재 지역명
            hospital_code: 병원 코드
            
        Returns:
            추정 좌표 (위도, 경도)
        """
        # 지역명에서 지역 코드 추정
        for code, name in self.region_names.items():
            if any(keyword in hospital_region for keyword in [name[:2], name]):
                return self.region_coordinates[code]
        
        # 기본값: 서울
        return self.region_coordinates['11']
        
    def calculate_hospital_attractiveness(self):
        """병원 매력도 계산 및 업데이트"""
        
        self.logger.info("Calculating hospital attractiveness scores")
        
        try:
            # 병원 매력도 계산
            attractiveness_query = """
                UPDATE nedis_meta.hospital_capacity
                SET attractiveness_score = 
                    daily_capacity_mean * 
                    CASE gubun 
                        WHEN '권역센터' THEN 2.5
                        WHEN '지역센터' THEN 1.8
                        WHEN '지역기관' THEN 1.0
                        ELSE 1.0
                    END *
                    -- 용량 안정성 보정
                    (1.0 + 1.0 / (1.0 + daily_capacity_std / daily_capacity_mean))
            """
            
            self.db.execute_query(attractiveness_query)
            
            # 매력도 정규화 (0-100 스케일)
            normalize_query = """
                UPDATE nedis_meta.hospital_capacity
                SET attractiveness_score = 
                    (attractiveness_score - (SELECT MIN(attractiveness_score) FROM nedis_meta.hospital_capacity)) /
                    (SELECT MAX(attractiveness_score) - MIN(attractiveness_score) FROM nedis_meta.hospital_capacity) * 100
            """
            
            self.db.execute_query(normalize_query)
            
            # 결과 확인
            results = self.db.fetch_dataframe("""
                SELECT emorg_cd, gubun, daily_capacity_mean, attractiveness_score
                FROM nedis_meta.hospital_capacity
                ORDER BY attractiveness_score DESC
            """)
            
            self.logger.info(f"Attractiveness scores calculated for {len(results)} hospitals")
            self.logger.info(f"Top hospital: {results.iloc[0]['emorg_cd']} (score: {results.iloc[0]['attractiveness_score']:.2f})")
            
        except Exception as e:
            self.logger.error(f"Failed to calculate attractiveness: {e}")
            raise
            
    def calculate_allocation_probabilities(self):
        """Huff 모델을 사용한 병원 선택 확률 계산"""
        
        self.logger.info("Calculating hospital allocation probabilities using Huff model")
        
        try:
            # 확률 테이블 초기화
            self.db.execute_query("DROP TABLE IF EXISTS nedis_meta.hospital_choice_prob")
            self.db.execute_query("""
                CREATE TABLE nedis_meta.hospital_choice_prob (
                    pat_do_cd VARCHAR,
                    pat_age_gr VARCHAR,
                    pat_sex VARCHAR,
                    emorg_cd VARCHAR,
                    probability DOUBLE,
                    utility_score DOUBLE,
                    rank INTEGER,
                    PRIMARY KEY (pat_do_cd, pat_age_gr, pat_sex, emorg_cd)
                )
            """)
            
            # 모든 인구 그룹 조회
            population_groups = self.db.fetch_dataframe("""
                SELECT DISTINCT pat_do_cd, pat_age_gr, pat_sex
                FROM nedis_meta.population_margins
                WHERE yearly_visits > 0
            """)
            
            self.logger.info(f"Processing {len(population_groups)} population groups")
            
            probability_records = []
            
            for idx, group in population_groups.iterrows():
                region = group['pat_do_cd']
                age = group['pat_age_gr']
                sex = group['pat_sex']
                
                # 해당 지역에서 모든 병원까지의 거리와 매력도 조회
                hospitals_data = self.db.fetch_dataframe("""
                    SELECT 
                        h.emorg_cd,
                        h.attractiveness_score,
                        h.gubun,
                        h.daily_capacity_mean,
                        d.distance_km,
                        d.travel_time_min
                    FROM nedis_meta.hospital_capacity h
                    JOIN nedis_meta.distance_matrix d ON h.emorg_cd = d.to_emorg_cd
                    WHERE d.from_do_cd = ?
                """, [region])
                
                if len(hospitals_data) == 0:
                    continue
                
                # Huff 모델 유틸리티 계산
                utilities = []
                
                for _, hospital in hospitals_data.iterrows():
                    attractiveness = hospital['attractiveness_score']
                    distance = hospital['distance_km']
                    
                    # 연령대별 거리 민감도 조정
                    age_factor = self._get_age_distance_sensitivity(age)
                    adjusted_gamma = self.gamma * age_factor
                    
                    # 유틸리티 계산: A_j * d_ij^(-γ)
                    if distance > 0:
                        utility = attractiveness * (distance ** (-adjusted_gamma))
                    else:
                        utility = attractiveness * 1000  # 같은 지역
                    
                    utilities.append(utility)
                
                # 확률 계산 (유틸리티 정규화)
                total_utility = sum(utilities)
                
                if total_utility > 0:
                    for i, (_, hospital) in enumerate(hospitals_data.iterrows()):
                        probability = utilities[i] / total_utility
                        
                        probability_records.append({
                            'pat_do_cd': region,
                            'pat_age_gr': age,
                            'pat_sex': sex,
                            'emorg_cd': hospital['emorg_cd'],
                            'probability': probability,
                            'utility_score': utilities[i]
                        })
            
            # 확률 순위 계산 및 저장
            prob_df = pd.DataFrame(probability_records)
            
            # 각 그룹별 확률 순위 계산
            prob_df['rank'] = prob_df.groupby(['pat_do_cd', 'pat_age_gr', 'pat_sex'])['probability'].rank(method='dense', ascending=False)
            
            # 배치 삽입
            for _, row in prob_df.iterrows():
                self.db.execute_query("""
                    INSERT INTO nedis_meta.hospital_choice_prob
                    (pat_do_cd, pat_age_gr, pat_sex, emorg_cd, probability, utility_score, rank)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (row['pat_do_cd'], row['pat_age_gr'], row['pat_sex'], row['emorg_cd'],
                      row['probability'], row['utility_score'], int(row['rank'])))
            
            self.logger.info(f"Hospital choice probabilities calculated: {len(probability_records)} records")
            
            # 확률 분포 요약 통계
            self._log_probability_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to calculate allocation probabilities: {e}")
            raise
            
    def _get_age_distance_sensitivity(self, age_group: str) -> float:
        """
        연령대별 거리 민감도 조정 계수
        
        Args:
            age_group: 연령군 코드 ('01', '09', '10'-'90')
            
        Returns:
            거리 민감도 조정 계수
        """
        age_sensitivity = {
            '01': 0.7,  # 영아 - 가까운 병원 선호
            '09': 0.8,  # 유아
            '10': 1.0,  # 10대 - 기본
            '20': 1.2,  # 20대 - 거리 덜 민감
            '30': 1.1,  # 30대 
            '40': 1.0,  # 40대
            '50': 0.9,  # 50대
            '60': 0.8,  # 60대 - 가까운 병원 선호
            '70': 0.7,  # 70대
            '80': 0.6,  # 80대 - 매우 가까운 병원 선호
            '90': 0.5   # 90대
        }
        
        return age_sensitivity.get(age_group, 1.0)
        
    def _log_probability_summary(self):
        """확률 분포 요약 통계 로그"""
        
        try:
            summary = self.db.fetch_dataframe("""
                SELECT 
                    emorg_cd,
                    COUNT(*) as group_count,
                    AVG(probability) as avg_probability,
                    MIN(probability) as min_probability,
                    MAX(probability) as max_probability,
                    SUM(CASE WHEN rank = 1 THEN 1 ELSE 0 END) as first_choice_count
                FROM nedis_meta.hospital_choice_prob
                GROUP BY emorg_cd
                ORDER BY avg_probability DESC
            """)
            
            self.logger.info("Hospital choice probability summary:")
            for _, row in summary.head(5).iterrows():
                self.logger.info(
                    f"  {row['emorg_cd']}: avg={row['avg_probability']:.4f}, "
                    f"first_choice={row['first_choice_count']} groups"
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to log probability summary: {e}")
    
    def initialize_allocation_table(self):
        """병원 할당 테이블 초기화 (한 번만 실행)"""
        
        self.logger.info("Initializing hospital allocation table")
        
        try:
            # 기존 병원 할당 테이블 삭제
            self.db.execute_query("DROP TABLE IF EXISTS nedis_synthetic.hospital_allocations")
            
            # 병원 할당 테이블 생성
            self.db.execute_query("""
                CREATE TABLE nedis_synthetic.hospital_allocations (
                    vst_dt VARCHAR,
                    emorg_cd VARCHAR,
                    pat_do_cd VARCHAR,
                    pat_age_gr VARCHAR,
                    pat_sex VARCHAR,
                    allocated_count INTEGER,
                    overflow_received INTEGER DEFAULT 0,
                    allocation_method VARCHAR DEFAULT 'gravity',
                    PRIMARY KEY (vst_dt, emorg_cd, pat_do_cd, pat_age_gr, pat_sex)
                )
            """)
            
            self.logger.info("Hospital allocation table initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize hospital allocation table: {e}")
            raise
    
    def allocate_with_capacity_constraints(self, date_str: str):
        """
        용량 제약을 고려한 병원 할당
        
        Args:
            date_str: 할당할 날짜 ('YYYYMMDD' 형식)
        """
        
        self.logger.info(f"Allocating hospital visits for date: {date_str}")
        
        try:
            # 해당 날짜의 일별 볼륨 로드
            daily_volumes = self.db.fetch_dataframe("""
                SELECT pat_do_cd, pat_age_gr, pat_sex, synthetic_daily_count
                FROM nedis_synthetic.daily_volumes
                WHERE vst_dt = ?
                ORDER BY pat_do_cd, pat_age_gr, pat_sex
            """, [date_str])
            
            if len(daily_volumes) == 0:
                self.logger.warning(f"No daily volumes found for date: {date_str}")
                return
            
            # 해당 날짜의 기존 데이터 삭제 (재실행 시 중복 방지)
            self.db.execute_query("""
                DELETE FROM nedis_synthetic.hospital_allocations 
                WHERE vst_dt = ?
            """, [date_str])
            
            allocation_records = []
            total_allocated = 0
            
            # 각 인구 그룹별 병원 할당
            for _, volume_row in daily_volumes.iterrows():
                region = volume_row['pat_do_cd']
                age = volume_row['pat_age_gr']
                sex = volume_row['pat_sex']
                total_count = int(volume_row['synthetic_daily_count'])
                
                if total_count <= 0:
                    continue
                
                # 해당 그룹의 병원 선택 확률 로드
                choice_probs = self.db.fetch_dataframe("""
                    SELECT emorg_cd, probability
                    FROM nedis_meta.hospital_choice_prob
                    WHERE pat_do_cd = ? AND pat_age_gr = ? AND pat_sex = ?
                    ORDER BY probability DESC
                """, [region, age, sex])
                
                if len(choice_probs) == 0:
                    self.logger.warning(f"No choice probabilities for group: {region}-{age}-{sex}")
                    continue
                
                # Multinomial 샘플링으로 병원별 초기 할당
                hospitals = choice_probs['emorg_cd'].values
                probabilities = choice_probs['probability'].values
                
                # 확률 정규화 (부동소수점 오차 방지)
                probabilities = probabilities / probabilities.sum()
                
                try:
                    # Multinomial 샘플링
                    allocation_counts = np.random.multinomial(total_count, probabilities)
                    
                    for hospital, count in zip(hospitals, allocation_counts):
                        if count > 0:
                            allocation_records.append({
                                'vst_dt': date_str,
                                'emorg_cd': hospital,
                                'pat_do_cd': region,
                                'pat_age_gr': age,
                                'pat_sex': sex,
                                'allocated_count': count,
                                'overflow_received': 0,
                                'allocation_method': 'gravity'
                            })
                            
                            total_allocated += count
                    
                except ValueError as e:
                    self.logger.error(f"Multinomial sampling error for group {region}-{age}-{sex}: {e}")
                    continue
            
            # 용량 제약 처리
            allocation_records = self._handle_capacity_constraints(allocation_records, date_str)
            
            # 할당 결과 저장
            for record in allocation_records:
                self.db.execute_query("""
                    INSERT INTO nedis_synthetic.hospital_allocations
                    (vst_dt, emorg_cd, pat_do_cd, pat_age_gr, pat_sex, 
                     allocated_count, overflow_received, allocation_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (record['vst_dt'], record['emorg_cd'], record['pat_do_cd'],
                      record['pat_age_gr'], record['pat_sex'], int(record['allocated_count']),
                      int(record['overflow_received']), record['allocation_method']))
            
            self.logger.info(f"Hospital allocation completed: {len(allocation_records)} allocation records")
            self.logger.info(f"Total allocated visits: {total_allocated}")
            
            # 할당 결과 요약
            self._log_allocation_summary(date_str)
            
        except Exception as e:
            self.logger.error(f"Failed to allocate hospital visits: {e}")
            raise
    
    def _handle_capacity_constraints(self, allocation_records: List[Dict], date_str: str) -> List[Dict]:
        """
        병원 용량 제약 처리 및 overflow 재분배
        
        Args:
            allocation_records: 초기 할당 레코드 리스트
            date_str: 날짜
            
        Returns:
            용량 제약 처리된 할당 레코드 리스트
        """
        
        self.logger.info("Processing capacity constraints and overflow redistribution")
        
        try:
            # 병원별 용량 정보 로드
            hospital_capacities = self.db.fetch_dataframe("""
                SELECT emorg_cd, 
                       daily_capacity_mean + 2 * daily_capacity_std as max_capacity,
                       gubun,
                       adr
                FROM nedis_meta.hospital_capacity
            """)
            
            capacity_dict = {}
            hospital_info = {}
            
            for _, row in hospital_capacities.iterrows():
                capacity_dict[row['emorg_cd']] = int(row['max_capacity'])
                hospital_info[row['emorg_cd']] = {
                    'gubun': row['gubun'],
                    'adr': row['adr']
                }
            
            # 병원별 할당량 집계
            hospital_loads = {}
            allocation_by_hospital = {}
            
            for record in allocation_records:
                hospital = record['emorg_cd']
                count = record['allocated_count']
                
                if hospital not in hospital_loads:
                    hospital_loads[hospital] = 0
                    allocation_by_hospital[hospital] = []
                
                hospital_loads[hospital] += count
                allocation_by_hospital[hospital].append(record)
            
            # 용량 초과 병원 식별 및 overflow 처리
            overflow_records = []
            
            for hospital, total_load in hospital_loads.items():
                max_capacity = capacity_dict.get(hospital, 1000)  # 기본값
                
                if total_load > max_capacity:
                    overflow = total_load - max_capacity
                    
                    self.logger.warning(
                        f"Hospital {hospital} overflow: {overflow} "
                        f"(load: {total_load}, capacity: {max_capacity})"
                    )
                    
                    # 해당 병원의 할당을 용량에 맞게 조정
                    records = allocation_by_hospital[hospital]
                    reduction_factor = max_capacity / total_load
                    
                    overflow_by_group = []
                    
                    for record in records:
                        original_count = record['allocated_count']
                        adjusted_count = int(original_count * reduction_factor)
                        overflow_count = original_count - adjusted_count
                        
                        record['allocated_count'] = adjusted_count
                        
                        if overflow_count > 0:
                            overflow_by_group.append({
                                'pat_do_cd': record['pat_do_cd'],
                                'pat_age_gr': record['pat_age_gr'],
                                'pat_sex': record['pat_sex'],
                                'overflow_count': overflow_count,
                                'original_hospital': hospital
                            })
                    
                    # Overflow 재분배
                    overflow_records.extend(overflow_by_group)
            
            # Overflow 재분배 처리
            if overflow_records:
                allocation_records = self._redistribute_overflow(
                    allocation_records, overflow_records, date_str, hospital_info, capacity_dict
                )
            
            return allocation_records
            
        except Exception as e:
            self.logger.error(f"Failed to handle capacity constraints: {e}")
            return allocation_records
    
    def _redistribute_overflow(self, allocation_records: List[Dict], 
                              overflow_records: List[Dict], date_str: str,
                              hospital_info: Dict, capacity_dict: Dict) -> List[Dict]:
        """
        Overflow 환자를 대안 병원으로 재분배
        
        Args:
            allocation_records: 현재 할당 레코드
            overflow_records: 재분배할 overflow 레코드
            date_str: 날짜
            hospital_info: 병원 정보
            capacity_dict: 병원 용량 정보
            
        Returns:
            재분배 완료된 할당 레코드
        """
        
        for overflow in overflow_records:
            region = overflow['pat_do_cd']
            age = overflow['pat_age_gr']
            sex = overflow['pat_sex']
            overflow_count = overflow['overflow_count']
            original_hospital = overflow['original_hospital']
            
            # 대안 병원 찾기 (동일 지역 내, 용량 여유 있는 병원)
            alternative_hospitals = self.db.fetch_dataframe("""
                SELECT hcp.emorg_cd, hcp.probability
                FROM nedis_meta.hospital_choice_prob hcp
                JOIN nedis_meta.hospital_capacity hc ON hcp.emorg_cd = hc.emorg_cd
                WHERE hcp.pat_do_cd = ? AND hcp.pat_age_gr = ? AND hcp.pat_sex = ?
                      AND hcp.emorg_cd != ?
                      AND hc.adr LIKE ?
                ORDER BY hcp.probability DESC
                LIMIT 3
            """, [region, age, sex, original_hospital, f"%{region}%"])
            
            if len(alternative_hospitals) == 0:
                # 동일 지역에 대안이 없으면 전국 범위에서 찾기
                alternative_hospitals = self.db.fetch_dataframe("""
                    SELECT emorg_cd, probability
                    FROM nedis_meta.hospital_choice_prob
                    WHERE pat_do_cd = ? AND pat_age_gr = ? AND pat_sex = ?
                          AND emorg_cd != ?
                    ORDER BY probability DESC
                    LIMIT 5
                """, [region, age, sex, original_hospital])
            
            # 용량 여유가 있는 병원으로 재분배
            redistributed = 0
            
            for _, alt_hospital_row in alternative_hospitals.iterrows():
                alt_hospital = alt_hospital_row['emorg_cd']
                
                # 현재 병원 부하 확인
                current_load = sum(
                    r['allocated_count'] for r in allocation_records 
                    if r['emorg_cd'] == alt_hospital
                )
                
                max_capacity = capacity_dict.get(alt_hospital, 1000)
                available_capacity = max_capacity - current_load
                
                if available_capacity > 0:
                    # 재분배할 수량 결정
                    redistribute_count = min(overflow_count - redistributed, available_capacity)
                    
                    # 기존 할당 레코드에서 해당 그룹 찾기
                    found = False
                    for record in allocation_records:
                        if (record['emorg_cd'] == alt_hospital and 
                            record['pat_do_cd'] == region and
                            record['pat_age_gr'] == age and 
                            record['pat_sex'] == sex):
                            
                            record['allocated_count'] += redistribute_count
                            record['overflow_received'] += redistribute_count
                            found = True
                            break
                    
                    # 새 할당 레코드 생성 (기존 할당이 없던 경우)
                    if not found:
                        allocation_records.append({
                            'vst_dt': date_str,
                            'emorg_cd': alt_hospital,
                            'pat_do_cd': region,
                            'pat_age_gr': age,
                            'pat_sex': sex,
                            'allocated_count': redistribute_count,
                            'overflow_received': redistribute_count,
                            'allocation_method': 'overflow_redistribution'
                        })
                    
                    redistributed += redistribute_count
                    
                    if redistributed >= overflow_count:
                        break
            
            if redistributed < overflow_count:
                self.logger.warning(
                    f"Could not redistribute all overflow for group {region}-{age}-{sex}: "
                    f"{redistributed}/{overflow_count}"
                )
        
        return allocation_records
    
    def _log_allocation_summary(self, date_str: str):
        """할당 결과 요약 로그"""
        
        try:
            # 병원별 할당 요약
            hospital_summary = self.db.fetch_dataframe("""
                SELECT 
                    ha.emorg_cd,
                    hc.gubun,
                    SUM(ha.allocated_count) as total_allocated,
                    SUM(ha.overflow_received) as total_overflow,
                    hc.daily_capacity_mean + 2 * hc.daily_capacity_std as max_capacity,
                    ROUND(SUM(ha.allocated_count) / (hc.daily_capacity_mean + 2 * hc.daily_capacity_std) * 100, 1) as utilization_rate
                FROM nedis_synthetic.hospital_allocations ha
                JOIN nedis_meta.hospital_capacity hc ON ha.emorg_cd = hc.emorg_cd
                WHERE ha.vst_dt = ?
                GROUP BY ha.emorg_cd, hc.gubun, hc.daily_capacity_mean, hc.daily_capacity_std
                ORDER BY total_allocated DESC
            """, [date_str])
            
            self.logger.info("Hospital allocation summary:")
            for _, row in hospital_summary.head(5).iterrows():
                self.logger.info(
                    f"  {row['emorg_cd']} ({row['gubun']}): "
                    f"allocated={row['total_allocated']}, "
                    f"overflow={row['total_overflow']}, "
                    f"utilization={row['utilization_rate']}%"
                )
            
            # 전체 통계
            total_stats = self.db.fetch_dataframe("""
                SELECT 
                    SUM(allocated_count) as total_visits,
                    SUM(overflow_received) as total_overflow,
                    COUNT(DISTINCT emorg_cd) as hospitals_used,
                    AVG(allocated_count) as avg_per_allocation
                FROM nedis_synthetic.hospital_allocations
                WHERE vst_dt = ?
            """, [date_str])
            
            if len(total_stats) > 0:
                stats = total_stats.iloc[0]
                self.logger.info(
                    f"Total allocation: {stats['total_visits']} visits, "
                    f"{stats['total_overflow']} overflow redistributed, "
                    f"{stats['hospitals_used']} hospitals used"
                )
            
        except Exception as e:
            self.logger.warning(f"Failed to log allocation summary: {e}")