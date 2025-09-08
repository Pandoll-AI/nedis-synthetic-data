"""
조건부 확률 추출 모듈

원본 NEDIS 데이터에서 임상 변수들 간의 조건부 확률을 계산하여
합성 데이터 생성 시 사용할 확률 테이블을 생성합니다.
베이지안 평활화를 적용하여 희귀 조합에 대한 안정적인 확률을 제공합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import logging
from datetime import datetime
import warnings

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.database import DatabaseManager
from core.config import ConfigManager


class ConditionalProbabilityExtractor:
    """조건부 확률 추출기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager):
        """
        조건부 확률 추출기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자 인스턴스
            config: 설정 관리자 인스턴스
        """
        self.db = db_manager
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # 베이지안 평활화 파라미터
        self.bayesian_alpha = config.get('clinical.bayesian_alpha', 1.0)
        self.min_count_threshold = config.get('clinical.diagnosis_min_count', 10)
        
    def create_ktas_probability_table(self) -> bool:
        """
        KTAS 조건부 확률 테이블 생성
        
        (연령군, 성별, 병원종별, 내원수단) → KTAS 등급 확률을 계산하여 저장
        베이지안 평활화를 적용하여 희귀 조합에 대해서도 안정적인 확률 제공
        
        Returns:
            성공 여부
        """
        self.logger.info("Starting KTAS conditional probability table creation")
        
        try:
            # 기존 데이터 삭제
            self.db.execute_query("DELETE FROM nedis_meta.ktas_conditional_prob")
            
            # KTAS 조건부 확률 계산 (단순화된 접근법)
            ktas_prob_query = """
            WITH ktas_counts AS (
                SELECT 
                    pat_age_gr,
                    pat_sex,
                    gubun,
                    vst_meth,
                    ktas_fstu,
                    COUNT(*) as count
                FROM nedis_original.nedis2017
                WHERE pat_age_gr != '' AND pat_age_gr IS NOT NULL
                  AND pat_sex IN ('M', 'F')
                  AND gubun != '' AND gubun IS NOT NULL
                  AND vst_meth != '' AND vst_meth IS NOT NULL
                  AND ktas_fstu IN ('1', '2', '3', '4', '5')
                GROUP BY pat_age_gr, pat_sex, gubun, vst_meth, ktas_fstu
            ),
            group_totals AS (
                SELECT 
                    pat_age_gr,
                    pat_sex,
                    gubun,
                    vst_meth,
                    SUM(count) as group_total
                FROM ktas_counts
                GROUP BY pat_age_gr, pat_sex, gubun, vst_meth
                HAVING SUM(count) >= ?  -- 최소 샘플 수 필터
            )
            INSERT INTO nedis_meta.ktas_conditional_prob 
            SELECT 
                kc.pat_age_gr,
                kc.pat_sex,
                kc.gubun,
                kc.vst_meth,
                kc.ktas_fstu,
                -- 베이지안 평활화 적용: (count + α) / (total + α * n_categories)
                (kc.count + ?) / (gt.group_total + ? * 5.0) as probability,
                kc.count as sample_count
            FROM ktas_counts kc
            JOIN group_totals gt 
                ON kc.pat_age_gr = gt.pat_age_gr 
                AND kc.pat_sex = gt.pat_sex
                AND kc.gubun = gt.gubun 
                AND kc.vst_meth = gt.vst_meth
            ORDER BY kc.pat_age_gr, kc.pat_sex, kc.gubun, kc.vst_meth, kc.ktas_fstu
            """
            
            self.db.execute_query(ktas_prob_query, [
                self.min_count_threshold, 
                self.bayesian_alpha, 
                self.bayesian_alpha
            ])
            
            # 결과 검증
            verification_result = self._verify_ktas_probabilities()
            
            count_query = "SELECT COUNT(*) as total FROM nedis_meta.ktas_conditional_prob"
            total_records = self.db.fetch_dataframe(count_query)['total'][0]
            
            self.logger.info(f"Created {total_records:,} KTAS conditional probability records")
            
            if verification_result['issues']:
                self.logger.warning(f"Verification issues: {verification_result['issues']}")
            else:
                self.logger.info("KTAS probability table passed all verification checks")
                
            return True
            
        except Exception as e:
            self.logger.error(f"KTAS probability table creation failed: {e}")
            return False
    
    def create_diagnosis_probability_table(self) -> bool:
        """
        진단 조건부 확률 테이블 생성
        
        (연령군, 성별, 병원종별, KTAS등급) → 진단코드 확률을 계산
        희귀 진단은 상위 카테고리로 그룹화하여 처리
        
        Returns:
            성공 여부
        """
        self.logger.info("Starting diagnosis conditional probability table creation")
        
        # 원본 데이터에 진단 테이블이 있는지 확인
        if not self.db.table_exists("nedis_original.diag_er"):
            self.logger.warning("diag_er table not found, skipping diagnosis probabilities")
            return True
            
        try:
            # 기존 데이터 삭제
            self.db.execute_query("DELETE FROM nedis_meta.diagnosis_conditional_prob")
            
            # 1. 주진단 (position = 1) 확률 테이블 생성
            main_diagnosis_query = """
            WITH diagnosis_counts AS (
                SELECT 
                    n.pat_age_gr,
                    n.pat_sex,
                    n.gubun,
                    n.ktas_fstu,
                    CASE 
                        WHEN diag_count.cnt >= ? THEN d.diagnosis_code
                        ELSE LEFT(d.diagnosis_code, 3) || 'X'  -- 희귀 진단은 상위 3자리 + X로 그룹화
                    END as grouped_diagnosis_code,
                    COUNT(*) as count
                FROM nedis_original.nedis2017 n
                JOIN nedis_original.diag_er d ON n.index_key = d.index_key
                LEFT JOIN (
                    SELECT diagnosis_code, COUNT(*) as cnt
                    FROM nedis_original.diag_er
                    WHERE position = 1
                    GROUP BY diagnosis_code
                ) diag_count ON d.diagnosis_code = diag_count.diagnosis_code
                WHERE n.pat_age_gr != '' AND n.pat_age_gr IS NOT NULL
                  AND n.pat_sex IN ('M', 'F')
                  AND n.gubun != '' AND n.gubun IS NOT NULL
                  AND n.ktas_fstu IN ('1', '2', '3', '4', '5')
                  AND d.position = 1  -- 주진단만
                  AND d.diagnosis_code != '' AND d.diagnosis_code IS NOT NULL
                GROUP BY n.pat_age_gr, n.pat_sex, n.gubun, n.ktas_fstu, grouped_diagnosis_code
            ),
            group_totals AS (
                SELECT 
                    pat_age_gr, pat_sex, gubun, ktas_fstu,
                    SUM(count) as group_total
                FROM diagnosis_counts
                GROUP BY pat_age_gr, pat_sex, gubun, ktas_fstu
                HAVING SUM(count) >= ?  -- 최소 샘플 수
            )
            INSERT INTO nedis_meta.diagnosis_conditional_prob
            SELECT 
                dc.pat_age_gr,
                dc.pat_sex,
                dc.gubun,
                dc.ktas_fstu,
                dc.grouped_diagnosis_code,
                -- 베이지안 평활화
                (dc.count + ?) / (gt.group_total + ? * 
                    (SELECT COUNT(DISTINCT grouped_diagnosis_code) FROM diagnosis_counts dc2 
                     WHERE dc2.pat_age_gr = dc.pat_age_gr AND dc2.pat_sex = dc.pat_sex 
                       AND dc2.gubun = dc.gubun AND dc2.ktas_fstu = dc.ktas_fstu)
                ) as probability,
                true as is_primary,
                dc.count as sample_count
            FROM diagnosis_counts dc
            JOIN group_totals gt 
                ON dc.pat_age_gr = gt.pat_age_gr 
                AND dc.pat_sex = gt.pat_sex
                AND dc.gubun = gt.gubun 
                AND dc.ktas_fstu = gt.ktas_fstu
            """
            
            self.db.execute_query(main_diagnosis_query, [
                self.min_count_threshold,
                self.min_count_threshold, 
                self.bayesian_alpha,
                self.bayesian_alpha
            ])
            
            # 2. 부진단 처리 (선택사항)
            self._create_secondary_diagnosis_probabilities()
            
            # 결과 검증
            count_query = "SELECT COUNT(*) as total FROM nedis_meta.diagnosis_conditional_prob"
            total_records = self.db.fetch_dataframe(count_query)['total'][0]
            
            primary_count_query = """
            SELECT COUNT(*) as primary_count 
            FROM nedis_meta.diagnosis_conditional_prob 
            WHERE is_primary = true
            """
            primary_records = self.db.fetch_dataframe(primary_count_query)['primary_count'][0]
            
            self.logger.info(f"Created {total_records:,} diagnosis probability records")
            self.logger.info(f"Primary diagnosis records: {primary_records:,}")
            
            # 확률 합계 검증
            prob_verification = self._verify_diagnosis_probabilities()
            if prob_verification['issues']:
                self.logger.warning(f"Probability verification issues: {prob_verification['issues']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Diagnosis probability table creation failed: {e}")
            return False
    
    def _create_secondary_diagnosis_probabilities(self):
        """부진단 확률 테이블 생성 (선택사항)"""
        try:
            # 부진단은 주진단에 비해 단순화된 모델 사용
            # 현재는 주진단만 구현하고, 부진단은 향후 확장을 위해 placeholder
            self.logger.debug("Secondary diagnosis probabilities - placeholder for future implementation")
            
        except Exception as e:
            self.logger.warning(f"Secondary diagnosis creation warning: {e}")
    
    def _verify_ktas_probabilities(self) -> Dict[str, Any]:
        """KTAS 확률 테이블 검증"""
        verification = {'issues': []}
        
        try:
            # 1. 확률 합계가 1인지 확인
            prob_sum_query = """
            SELECT 
                pat_age_gr, pat_sex, gubun, vst_meth,
                SUM(probability) as prob_sum,
                COUNT(*) as ktas_count
            FROM nedis_meta.ktas_conditional_prob
            GROUP BY pat_age_gr, pat_sex, gubun, vst_meth
            HAVING ABS(SUM(probability) - 1.0) > 0.01
            LIMIT 10
            """
            
            prob_issues = self.db.fetch_dataframe(prob_sum_query)
            if len(prob_issues) > 0:
                verification['issues'].append(f"{len(prob_issues)} groups with probability sum != 1.0")
                verification['prob_sum_examples'] = prob_issues.to_dict('records')
            
            # 2. 음수 확률 확인
            negative_prob_query = """
            SELECT COUNT(*) as negative_count
            FROM nedis_meta.ktas_conditional_prob
            WHERE probability < 0
            """
            
            negative_count = self.db.fetch_dataframe(negative_prob_query)['negative_count'][0]
            if negative_count > 0:
                verification['issues'].append(f"{negative_count} records with negative probability")
            
            # 3. 기본 통계
            stats_query = """
            SELECT 
                COUNT(*) as total_records,
                COUNT(DISTINCT pat_age_gr || '|' || pat_sex || '|' || gubun || '|' || vst_meth) as unique_groups,
                AVG(probability) as avg_probability,
                MIN(probability) as min_probability,
                MAX(probability) as max_probability
            FROM nedis_meta.ktas_conditional_prob
            """
            
            verification['statistics'] = self.db.fetch_dataframe(stats_query).iloc[0].to_dict()
            
        except Exception as e:
            verification['issues'].append(f"Verification error: {e}")
            
        return verification
    
    def _verify_diagnosis_probabilities(self) -> Dict[str, Any]:
        """진단 확률 테이블 검증"""
        verification = {'issues': []}
        
        try:
            # 확률 합계 검증 (주진단만)
            prob_sum_query = """
            WITH group_sums AS (
                SELECT 
                    pat_age_gr, pat_sex, gubun, ktas_fstu,
                    SUM(probability) as prob_sum
                FROM nedis_meta.diagnosis_conditional_prob
                WHERE is_primary = true
                GROUP BY pat_age_gr, pat_sex, gubun, ktas_fstu
            )
            SELECT 
                COUNT(*) as total_groups,
                COUNT(CASE WHEN ABS(prob_sum - 1.0) > 0.01 THEN 1 END) as problematic_groups,
                AVG(prob_sum) as avg_prob_sum,
                MIN(prob_sum) as min_prob_sum,
                MAX(prob_sum) as max_prob_sum
            FROM group_sums
            """
            
            prob_stats = self.db.fetch_dataframe(prob_sum_query).iloc[0]
            
            if prob_stats['problematic_groups'] > 0:
                pct = prob_stats['problematic_groups'] / prob_stats['total_groups'] * 100
                verification['issues'].append(
                    f"{prob_stats['problematic_groups']} groups ({pct:.1f}%) with probability sum != 1.0"
                )
            
            verification['probability_statistics'] = prob_stats.to_dict()
            
            # 진단 코드 다양성 확인
            diversity_query = """
            SELECT 
                COUNT(DISTINCT diagnosis_code) as unique_diagnoses,
                COUNT(*) as total_records,
                AVG(sample_count) as avg_sample_count
            FROM nedis_meta.diagnosis_conditional_prob
            WHERE is_primary = true
            """
            
            diversity_stats = self.db.fetch_dataframe(diversity_query).iloc[0]
            verification['diversity_statistics'] = diversity_stats.to_dict()
            
        except Exception as e:
            verification['issues'].append(f"Verification error: {e}")
            
        return verification
    
    def create_all_probability_tables(self) -> bool:
        """모든 조건부 확률 테이블 생성"""
        self.logger.info("Creating all conditional probability tables")
        
        success_count = 0
        total_count = 2
        
        # 1. KTAS 확률 테이블
        if self.create_ktas_probability_table():
            success_count += 1
            self.logger.info("✓ KTAS probability table created")
        else:
            self.logger.error("✗ KTAS probability table creation failed")
        
        # 2. 진단 확률 테이블  
        if self.create_diagnosis_probability_table():
            success_count += 1
            self.logger.info("✓ Diagnosis probability table created")
        else:
            self.logger.error("✗ Diagnosis probability table creation failed")
        
        success_rate = success_count / total_count
        self.logger.info(f"Conditional probability extraction completed: {success_count}/{total_count} tables ({success_rate:.1%})")
        
        return success_count == total_count
    
    def get_probability_summary(self) -> Dict[str, Any]:
        """확률 테이블 요약 정보"""
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'ktas_probabilities': {},
            'diagnosis_probabilities': {}
        }
        
        try:
            # KTAS 확률 요약
            if self.db.table_exists("nedis_meta.ktas_conditional_prob"):
                ktas_summary_query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(DISTINCT pat_age_gr || '|' || pat_sex || '|' || gubun || '|' || vst_meth) as unique_combinations,
                    AVG(probability) as avg_probability,
                    AVG(sample_count) as avg_sample_count
                FROM nedis_meta.ktas_conditional_prob
                """
                summary['ktas_probabilities'] = self.db.fetch_dataframe(ktas_summary_query).iloc[0].to_dict()
            
            # 진단 확률 요약
            if self.db.table_exists("nedis_meta.diagnosis_conditional_prob"):
                diagnosis_summary_query = """
                SELECT 
                    COUNT(*) as total_records,
                    COUNT(CASE WHEN is_primary THEN 1 END) as primary_records,
                    COUNT(DISTINCT diagnosis_code) as unique_diagnoses,
                    AVG(probability) as avg_probability,
                    AVG(sample_count) as avg_sample_count
                FROM nedis_meta.diagnosis_conditional_prob
                """
                summary['diagnosis_probabilities'] = self.db.fetch_dataframe(diagnosis_summary_query).iloc[0].to_dict()
                
        except Exception as e:
            summary['error'] = str(e)
            
        return summary