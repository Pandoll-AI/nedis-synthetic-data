"""
인구 볼륨 생성기 모듈

Dirichlet-Multinomial 모델을 사용하여 현실적인 인구 분포를 가진
연간 방문 볼륨을 생성합니다. 베이지안 불확실성을 고려하여
원본 데이터의 통계적 특성을 유지하면서도 다양성을 제공합니다.
"""

import numpy as np
import pandas as pd
from scipy.stats import dirichlet, multinomial
from typing import Dict, List, Tuple, Optional, Any
import logging
from datetime import datetime
import time
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import DatabaseManager
from core.config import ConfigManager


class PopulationVolumeGenerator:
    """인구 볼륨 생성기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        인구 볼륨 생성기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자 인스턴스
            config: 설정 관리자 인스턴스
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Dirichlet 분포 파라미터 (베이지안 평활화)
        self.alpha = config.get('population.dirichlet_alpha', 1.0)
        
        # 성능 최적화를 위한 배치 크기
        self.batch_size = config.get('database.batch_size', 10000)
        
    def generate_yearly_volumes(self, target_total: int = 9_200_000) -> bool:
        """
        연간 볼륨 생성 및 저장
        
        Dirichlet-Multinomial 모델을 사용하여 각 (시도, 연령군, 성별) 조합별
        연간 방문수를 생성합니다. 원본 데이터의 비율을 유지하면서도
        베이지안 불확실성을 반영합니다.
        
        Args:
            target_total: 목표 총 레코드 수
            
        Returns:
            성공 여부
        """
        self.logger.info(f"Starting yearly volume generation (target: {target_total:,})")
        
        try:
            # 1. 원본 시도별 비율 계산
            region_proportions = self._get_regional_proportions()
            if not region_proportions:
                raise Exception("Failed to get regional proportions")
            
            # 2. 시도별 목표 볼륨 할당
            region_targets = self._allocate_regional_targets(region_proportions, target_total)
            
            # 3. 기존 데이터 삭제
            self.db.execute_query("DELETE FROM nedis_synthetic.yearly_volumes")
            
            # 4. 시도별 Dirichlet-Multinomial 생성
            all_synthetic_volumes = []
            total_generated = 0
            
            self.logger.info(f"Processing {len(region_targets)} regions...")
            
            for region_code in tqdm(region_targets.keys(), desc="Generating volumes by region"):
                target_volume = region_targets[region_code]
                
                if target_volume <= 0:
                    continue
                    
                region_volumes = self._generate_region_volumes(region_code, target_volume)
                
                if region_volumes:
                    all_synthetic_volumes.extend(region_volumes)
                    total_generated += sum(vol[3] for vol in region_volumes)  # synthetic_yearly_count
                    
                # 주기적으로 배치 저장 (메모리 효율성)
                if len(all_synthetic_volumes) >= self.batch_size:
                    self._save_volume_batch(all_synthetic_volumes)
                    all_synthetic_volumes = []
            
            # 남은 데이터 저장
            if all_synthetic_volumes:
                self._save_volume_batch(all_synthetic_volumes)
            
            # 5. 결과 검증
            actual_total = self.db.get_table_count("nedis_synthetic.yearly_volumes")
            total_records = self.db.fetch_dataframe(
                "SELECT SUM(synthetic_yearly_count) as total FROM nedis_synthetic.yearly_volumes"
            )['total'][0]
            
            self.logger.info(f"Generated {actual_total:,} combinations with {total_records:,} total records")
            self.logger.info(f"Target vs Actual: {target_total:,} vs {total_records:,} (diff: {abs(target_total - total_records):,})")
            
            # 허용 가능한 오차 범위 확인 (1% 이내)
            error_rate = abs(target_total - total_records) / target_total
            if error_rate > 0.01:
                self.logger.warning(f"Large discrepancy detected: {error_rate:.2%}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Yearly volume generation failed: {e}")
            return False
    
    def _get_regional_proportions(self) -> Dict[str, float]:
        """원본 데이터에서 시도별 비율 계산"""
        try:
            query = """
            SELECT 
                pat_do_cd,
                SUM(yearly_visits) as total_visits
            FROM nedis_meta.population_margins
            WHERE pat_do_cd != '' AND pat_do_cd IS NOT NULL
            GROUP BY pat_do_cd
            HAVING SUM(yearly_visits) > 0
            ORDER BY total_visits DESC
            """
            
            region_data = self.db.fetch_dataframe(query)
            
            if len(region_data) == 0:
                self.logger.error("No region data found in population_margins table")
                return {}
            
            total_visits = region_data['total_visits'].sum()
            
            proportions = {}
            for _, row in region_data.iterrows():
                proportions[row['pat_do_cd']] = row['total_visits'] / total_visits
                
            self.logger.info(f"Calculated proportions for {len(proportions)} regions")
            self.logger.debug(f"Top 5 regions: {dict(list(proportions.items())[:5])}")
            
            return proportions
            
        except Exception as e:
            self.logger.error(f"Failed to get regional proportions: {e}")
            return {}
    
    def _allocate_regional_targets(self, proportions: Dict[str, float], 
                                 total_target: int) -> Dict[str, int]:
        """시도별 목표 볼륨 할당"""
        targets = {}
        allocated_total = 0
        
        # 비례 할당
        for region, proportion in proportions.items():
            target = int(total_target * proportion)
            targets[region] = target
            allocated_total += target
        
        # 반올림으로 인한 차이 보정
        difference = total_target - allocated_total
        if difference != 0:
            # 가장 큰 지역에 차이만큼 할당/차감
            largest_region = max(targets.keys(), key=lambda k: targets[k])
            targets[largest_region] += difference
            
            self.logger.debug(f"Adjusted {largest_region} by {difference:,} to match target")
        
        allocated_sum = sum(targets.values())
        self.logger.info(f"Regional allocation: {allocated_sum:,} total across {len(targets)} regions")
        
        return targets
    
    def _generate_region_volumes(self, region_code: str, target_volume: int) -> List[Tuple]:
        """특정 시도의 연령×성별 볼륨 생성"""
        try:
            # 해당 지역의 원본 분포 조회
            query = """
            SELECT pat_age_gr, pat_sex, yearly_visits
            FROM nedis_meta.population_margins
            WHERE pat_do_cd = ?
              AND yearly_visits > 0
            ORDER BY pat_age_gr, pat_sex
            """
            
            region_data = self.db.fetch_dataframe(query, [region_code])
            
            if len(region_data) == 0:
                self.logger.warning(f"No data found for region {region_code}")
                return []
            
            # Dirichlet 파라미터 설정
            observed_counts = region_data['yearly_visits'].values.astype(float)
            alpha_vector = observed_counts + self.alpha
            
            # Dirichlet 분포에서 확률 벡터 샘플링
            probabilities = dirichlet.rvs(alpha_vector, random_state=None)[0]
            
            # 확률 정규화 (수치적 안정성)
            probabilities = probabilities / probabilities.sum()
            
            # Multinomial 분포로 개수 생성
            synthetic_counts = multinomial.rvs(target_volume, probabilities, 
                                             random_state=None)
            
            # 결과 구성
            results = []
            for i, (_, row) in enumerate(region_data.iterrows()):
                if synthetic_counts[i] > 0:  # 0인 조합은 제외
                    results.append((
                        region_code,
                        row['pat_age_gr'],
                        row['pat_sex'],
                        int(synthetic_counts[i])
                    ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to generate volumes for region {region_code}: {e}")
            return []
    
    def _save_volume_batch(self, volume_batch: List[Tuple]):
        """볼륨 배치 데이터베이스 저장"""
        if not volume_batch:
            return
            
        try:
            # DataFrame으로 변환
            volumes_df = pd.DataFrame(volume_batch, columns=[
                'pat_do_cd', 'pat_age_gr', 'pat_sex', 'synthetic_yearly_count'
            ])
            
            # 데이터 타입 최적화
            volumes_df['synthetic_yearly_count'] = volumes_df['synthetic_yearly_count'].astype('int32')
            
            # DuckDB에 삽입
            self.db.conn.execute("""
                INSERT INTO nedis_synthetic.yearly_volumes 
                (pat_do_cd, pat_age_gr, pat_sex, synthetic_yearly_count)
                SELECT * FROM volumes_df
            """)
            
            self.logger.debug(f"Saved batch of {len(volume_batch):,} volume records")
            
        except Exception as e:
            self.logger.error(f"Failed to save volume batch: {e}")
            raise
    
    def get_generation_summary(self) -> Dict[str, Any]:
        """생성 결과 요약"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'parameters': {
                'dirichlet_alpha': self.alpha,
                'batch_size': self.batch_size
            }
        }
        
        try:
            if not self.db.table_exists("nedis_synthetic.yearly_volumes"):
                summary['error'] = "yearly_volumes table not found"
                return summary
            
            # 기본 통계
            stats_query = """
            SELECT 
                COUNT(*) as total_combinations,
                SUM(synthetic_yearly_count) as total_records,
                AVG(synthetic_yearly_count) as avg_records,
                MIN(synthetic_yearly_count) as min_records,
                MAX(synthetic_yearly_count) as max_records,
                COUNT(DISTINCT pat_do_cd) as unique_regions,
                COUNT(DISTINCT pat_age_gr) as unique_age_groups,
                COUNT(DISTINCT pat_sex) as unique_genders
            FROM nedis_synthetic.yearly_volumes
            """
            
            stats = self.db.fetch_dataframe(stats_query).iloc[0]
            summary['statistics'] = stats.to_dict()
            
            # 시도별 분포
            regional_query = """
            SELECT 
                pat_do_cd,
                COUNT(*) as combinations,
                SUM(synthetic_yearly_count) as total_records,
                AVG(synthetic_yearly_count) as avg_records
            FROM nedis_synthetic.yearly_volumes
            GROUP BY pat_do_cd
            ORDER BY total_records DESC
            LIMIT 10
            """
            
            summary['top_regions'] = self.db.fetch_dataframe(regional_query).to_dict('records')
            
            # 성별 분포
            gender_query = """
            SELECT 
                pat_sex,
                COUNT(*) as combinations,
                SUM(synthetic_yearly_count) as total_records
            FROM nedis_synthetic.yearly_volumes
            GROUP BY pat_sex
            ORDER BY pat_sex
            """
            
            summary['gender_distribution'] = self.db.fetch_dataframe(gender_query).to_dict('records')
            
        except Exception as e:
            summary['error'] = str(e)
            
        return summary
    
    def validate_generated_volumes(self) -> Dict[str, Any]:
        """생성된 볼륨 검증"""
        validation = {
            'timestamp': datetime.now().isoformat(),
            'checks': {},
            'issues': []
        }
        
        try:
            # 1. 음수 또는 0 레코드 확인
            zero_count_query = """
            SELECT COUNT(*) as zero_count
            FROM nedis_synthetic.yearly_volumes
            WHERE synthetic_yearly_count <= 0
            """
            zero_count = self.db.fetch_dataframe(zero_count_query)['zero_count'][0]
            validation['checks']['zero_records'] = zero_count
            
            if zero_count > 0:
                validation['issues'].append(f"{zero_count} combinations with <= 0 records")
            
            # 2. 원본과 합성 데이터 비교
            original_total_query = "SELECT SUM(yearly_visits) as total FROM nedis_meta.population_margins"
            synthetic_total_query = "SELECT SUM(synthetic_yearly_count) as total FROM nedis_synthetic.yearly_volumes"
            
            original_total = self.db.fetch_dataframe(original_total_query)['total'][0]
            synthetic_total = self.db.fetch_dataframe(synthetic_total_query)['total'][0]
            
            validation['checks']['original_total'] = original_total
            validation['checks']['synthetic_total'] = synthetic_total
            
            if original_total > 0:
                ratio = synthetic_total / original_total
                validation['checks']['synthetic_to_original_ratio'] = ratio
                
                # 합리적인 범위인지 확인 (0.5 ~ 2.0)
                if ratio < 0.5 or ratio > 2.0:
                    validation['issues'].append(f"Unusual ratio: synthetic/original = {ratio:.2f}")
            
            # 3. 지역별 분포 일관성 확인
            region_comparison_query = """
            WITH original_regions AS (
                SELECT pat_do_cd, SUM(yearly_visits) as original_count
                FROM nedis_meta.population_margins
                GROUP BY pat_do_cd
            ),
            synthetic_regions AS (
                SELECT pat_do_cd, SUM(synthetic_yearly_count) as synthetic_count
                FROM nedis_synthetic.yearly_volumes
                GROUP BY pat_do_cd
            )
            SELECT 
                COUNT(*) as regions_matched,
                AVG(ABS(o.original_count - s.synthetic_count) / NULLIF(o.original_count, 0)) as avg_relative_error
            FROM original_regions o
            JOIN synthetic_regions s ON o.pat_do_cd = s.pat_do_cd
            """
            
            comparison = self.db.fetch_dataframe(region_comparison_query).iloc[0]
            validation['checks']['region_consistency'] = comparison.to_dict()
            
            # 4. 극값 확인
            extreme_query = """
            WITH stats AS (
                SELECT 
                    AVG(synthetic_yearly_count) as avg_count,
                    STDDEV(synthetic_yearly_count) as std_count
                FROM nedis_synthetic.yearly_volumes
            )
            SELECT 
                COUNT(CASE WHEN yv.synthetic_yearly_count > s.avg_count + 3 * s.std_count THEN 1 END) as extreme_high,
                COUNT(CASE WHEN yv.synthetic_yearly_count < GREATEST(0, s.avg_count - 3 * s.std_count) THEN 1 END) as extreme_low
            FROM nedis_synthetic.yearly_volumes yv, stats s
            """
            
            extreme_stats = self.db.fetch_dataframe(extreme_query).iloc[0]
            validation['checks']['extreme_values'] = extreme_stats.to_dict()
            
            if extreme_stats['extreme_high'] > 0:
                validation['issues'].append(f"{extreme_stats['extreme_high']} extremely high values detected")
            
            # 검증 통과 여부
            validation['passed'] = len(validation['issues']) == 0
            
        except Exception as e:
            validation['error'] = str(e)
            validation['passed'] = False
            
        return validation