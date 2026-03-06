#!/usr/bin/env python3
"""
동적 패턴 분석기 (Dynamic Pattern Analyzer)

원본 데이터를 분석하여 패턴을 자동으로 발견하고 하드코딩을 제거하는 모듈입니다.
계층적 대안(소분류→대분류→전국) 및 캐싱 시스템을 포함합니다.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import pickle
import hashlib
from datetime import datetime
from dataclasses import dataclass, asdict

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


@dataclass
class PatternAnalysisConfig:
    """패턴 분석 설정"""
    cache_dir: str = "cache/patterns"
    use_cache: bool = True
    min_sample_size: int = 10  # 최소 샘플 수 (통계적 유의성)
    confidence_threshold: float = 0.95
    hierarchical_fallback: bool = True


class AnalysisCache:
    """분석 결과 캐싱 시스템"""
    
    def __init__(self, cache_dir: str = "cache/patterns"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.AnalysisCache")
        
        # 메타데이터 파일
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """캐시 메타데이터 로드"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.warning(f"Failed to load cache metadata: {e}")
        
        return {"cache_entries": {}, "last_updated": None}
    
    def _save_metadata(self):
        """캐시 메타데이터 저장"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self.logger.error(f"Failed to save cache metadata: {e}")
    
    def get_data_hash(self, db_manager: DatabaseManager, table_name: str) -> str:
        """데이터 해시값 계산"""
        try:
            # 테이블의 행 수와 최근 변경일시를 기반으로 해시 생성
            count_query = f"SELECT COUNT(*) as count FROM {table_name}"
            count_result = db_manager.fetch_dataframe(count_query)
            row_count = count_result['count'].iloc[0]
            
            # 샘플 데이터로 해시 계산 (성능 고려)
            sample_query = f"""
                SELECT * FROM {table_name} 
                ORDER BY RANDOM() 
                LIMIT 1000
            """
            sample_data = db_manager.fetch_dataframe(sample_query)
            
            # 해시 계산
            hash_input = f"{table_name}_{row_count}_{sample_data.to_string()}"
            return hashlib.md5(hash_input.encode()).hexdigest()
            
        except Exception as e:
            self.logger.warning(f"Failed to calculate data hash for {table_name}: {e}")
            return f"no_hash_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def load_cached_analysis(self, analysis_type: str, data_hash: str) -> Optional[Dict[str, Any]]:
        """캐시된 분석 결과 로드"""
        cache_key = f"{analysis_type}_{data_hash}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached_result = pickle.load(f)
            
            self.logger.info(f"Loaded cached analysis: {analysis_type}")
            return cached_result
            
        except Exception as e:
            self.logger.warning(f"Failed to load cached analysis {cache_key}: {e}")
            return None
    
    def save_analysis_cache(self, analysis_type: str, data_hash: str, results: Dict[str, Any]):
        """분석 결과 캐시 저장"""
        cache_key = f"{analysis_type}_{data_hash}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(results, f)
            
            # 메타데이터 업데이트
            self.metadata["cache_entries"][cache_key] = {
                "analysis_type": analysis_type,
                "data_hash": data_hash,
                "created_at": datetime.now().isoformat(),
                "file_path": str(cache_file)
            }
            self.metadata["last_updated"] = datetime.now().isoformat()
            self._save_metadata()
            
            self.logger.info(f"Saved analysis cache: {analysis_type}")
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis cache {cache_key}: {e}")
    
    def clear_cache(self, analysis_type: Optional[str] = None):
        """캐시 정리"""
        if analysis_type:
            # 특정 분석 타입 캐시만 정리
            keys_to_remove = []
            for key, entry in self.metadata["cache_entries"].items():
                if entry["analysis_type"] == analysis_type:
                    cache_file = Path(entry["file_path"])
                    if cache_file.exists():
                        cache_file.unlink()
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.metadata["cache_entries"][key]
        else:
            # 전체 캐시 정리
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self.metadata["cache_entries"] = {}
        
        self._save_metadata()
        self.logger.info(f"Cleared cache: {analysis_type or 'all'}")


class PatternAnalyzer:
    """동적 패턴 분석기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager,
                 analysis_config: Optional[PatternAnalysisConfig] = None):
        """
        초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
            analysis_config: 패턴 분석 설정
        """
        self.db = db_manager
        self.config = config
        self.analysis_config = analysis_config or PatternAnalysisConfig()
        self.logger = logging.getLogger(__name__)
        
        # 캐시 시스템
        self.cache = AnalysisCache(self.analysis_config.cache_dir)
        
        # 패턴 데이터 저장
        self._patterns_cache = {}

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
    
    def analyze_all_patterns(self) -> Dict[str, Any]:
        """모든 패턴 분석 수행"""
        self.logger.info("Starting comprehensive pattern analysis")
        
        patterns = {}
        
        # 데이터 해시 계산
        data_hash = self.cache.get_data_hash(self.db, self.src_table)
        
        # 각 패턴 분석
        analysis_methods = [
            ("hospital_allocation", self.analyze_hospital_allocation_patterns),
            ("ktas_distributions", self.analyze_ktas_distributions),
            ("regional_patterns", self.analyze_regional_patterns),
            ("demographic_patterns", self.analyze_demographic_patterns),
            ("temporal_patterns", self.analyze_temporal_patterns),
            ("temporal_conditional_patterns", self.analyze_temporal_conditional_patterns),
            ("visit_method_patterns", self.analyze_visit_method_patterns),
            ("chief_complaint_patterns", self.analyze_chief_complaint_patterns),
            ("department_patterns", self.analyze_department_patterns),
            ("treatment_result_patterns", self.analyze_treatment_result_patterns),
            ("missing_value_rates", self.analyze_missing_value_rates)
        ]
        
        for pattern_name, analysis_method in analysis_methods:
            self.logger.info(f"Analyzing {pattern_name}")
            
            # 캐시 확인
            if self.analysis_config.use_cache:
                cached_result = self.cache.load_cached_analysis(pattern_name, data_hash)
                if cached_result is not None:
                    patterns[pattern_name] = cached_result
                    continue
            
            # 새로운 분석 수행
            try:
                analysis_result = analysis_method()
                patterns[pattern_name] = analysis_result
                
                # 캐시 저장
                if self.analysis_config.use_cache:
                    self.cache.save_analysis_cache(pattern_name, data_hash, analysis_result)
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze {pattern_name}: {e}")
                patterns[pattern_name] = {"error": str(e), "patterns": {}}
        
        # 메타 정보 추가
        patterns["metadata"] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "data_hash": data_hash,
            "total_patterns": len([p for p in patterns.values() if "error" not in p]),
            "config": asdict(self.analysis_config)
        }
        
        self.logger.info(f"Pattern analysis completed: {len(patterns)} pattern types")
        return patterns

    def _resolve_capacity_table(self) -> Optional[str]:
        """Resolve fully-qualified hospital capacity table if present.

        Returns a qualified table name or None if not found.
        """
        # Candidates: alias.nedis_meta.hospital_capacity, nedis_meta.hospital_capacity
        alias = None
        try:
            if hasattr(self, 'src_table') and isinstance(self.src_table, str) and '.' in self.src_table:
                alias = self.src_table.split('.')[0]
        except Exception:
            alias = None

        candidates = []
        if alias:
            candidates.append(f"{alias}.nedis_meta.hospital_capacity")
        candidates.append("nedis_meta.hospital_capacity")

        for cand in candidates:
            try:
                if self.db.table_exists(cand):
                    return cand
            except Exception:
                continue
        return None
    
    def analyze_hospital_allocation_patterns(self) -> Dict[str, Any]:
        """병원 할당 패턴 분석 (지역 기반)"""
        self.logger.info("Analyzing hospital allocation patterns")
        
        # 지역별 병원 선택 패턴 분석
        query = """
            SELECT
                ptmizipc as region_code,
                ptmiemcd as hospital_code,
                COUNT(*) as visit_count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY ptmizipc) as region_probability
            FROM {src}
            WHERE ptmizipc IS NOT NULL AND ptmiemcd IS NOT NULL
            GROUP BY ptmizipc, ptmiemcd
            HAVING COUNT(*) >= {min_samples}
            ORDER BY ptmizipc, visit_count DESC
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)
        
        allocation_data = self.db.fetch_dataframe(query)
        
        # 패턴 구조화
        patterns = {}
        
        for region in allocation_data['region_code'].unique():
            region_data = allocation_data[allocation_data['region_code'] == region]
            
            # 지역별 병원 선택 확률
            hospital_probs = {}
            for _, row in region_data.iterrows():
                hospital_probs[row['hospital_code']] = {
                    'probability': float(row['region_probability']),
                    'visit_count': int(row['visit_count'])
                }
            
            patterns[str(region)] = {
                'hospitals': hospital_probs,
                'total_visits': int(region_data['visit_count'].sum()),
                'unique_hospitals': len(hospital_probs)
            }
        
        # 계층적 대안 패턴 생성
        hierarchical_patterns = self._create_hierarchical_patterns(
            patterns, key_type='region'
        )
        
        return {
            'patterns': patterns,
            'hierarchical_fallback': hierarchical_patterns,
            'total_regions': len(patterns),
            'analysis_method': 'region_based_allocation'
        }
    
    def analyze_ktas_distributions(self) -> Dict[str, Any]:
        """KTAS 분포 분석 (계층적 대안 포함)"""
        self.logger.info("Analyzing KTAS distributions with hierarchical fallback")
        capacity_tbl = self._resolve_capacity_table()

        if capacity_tbl:
            # Use capacity table to derive hospital_type and region via capacity metadata
            detailed_query = """
                SELECT
                    n.ptmizipc as region_code,
                    h.ptmiemcd as hospital_code,
                    CASE
                        WHEN h.daily_capacity_mean >= 300 THEN 'large'
                        WHEN h.daily_capacity_mean >= 100 THEN 'medium'
                        ELSE 'small'
                    END as hospital_type,
                    n.ptmikts1 as ptmikts1,
                    COUNT(*) as count,
                    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(
                        PARTITION BY h.adr,
                        CASE WHEN h.daily_capacity_mean >= 300 THEN 'large'
                             WHEN h.daily_capacity_mean >= 100 THEN 'medium'
                             ELSE 'small' END
                    ) as probability
                FROM {src} n
                JOIN {cap} h ON n.ptmiemcd = h.ptmiemcd
                WHERE n.ptmikts1 IN ('1', '2', '3', '4', '5')
                  AND h.adr IS NOT NULL
                GROUP BY n.ptmizipc, h.adr, h.ptmiemcd, hospital_type, n.ptmikts1
                HAVING COUNT(*) >= {min_samples}
            """

            detailed_data = self.db.fetch_dataframe(
                detailed_query.format(src=self.src_table, cap=capacity_tbl, min_samples=self.analysis_config.min_sample_size)
            )

            detailed_patterns = {}
            for _, row in detailed_data.iterrows():
                key = f"{row['region_code']}_{row['hospital_type']}"
                detailed_patterns.setdefault(key, {})[row['ptmikts1']] = {
                    'probability': float(row['probability']),
                    'count': int(row['count'])
                }

            major_query = """
                SELECT
                    SUBSTR(h.adr, 1, 2) as major_region,
                    CASE
                        WHEN h.daily_capacity_mean >= 300 THEN 'large'
                        WHEN h.daily_capacity_mean >= 100 THEN 'medium'
                        ELSE 'small'
                    END as hospital_type,
                    n.ptmikts1 as ptmikts1,
                    COUNT(*) as count,
                    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(
                        PARTITION BY SUBSTR(h.adr, 1, 2),
                        CASE WHEN h.daily_capacity_mean >= 300 THEN 'large'
                             WHEN h.daily_capacity_mean >= 100 THEN 'medium'
                             ELSE 'small' END
                    ) as probability
                FROM {src} n
                JOIN {cap} h ON n.ptmiemcd = h.ptmiemcd
                WHERE n.ptmikts1 IN ('1', '2', '3', '4', '5')
                  AND h.adr IS NOT NULL
                GROUP BY major_region, hospital_type, n.ptmikts1
                HAVING COUNT(*) >= {min_samples}
            """

            major_data = self.db.fetch_dataframe(
                major_query.format(src=self.src_table, cap=capacity_tbl, min_samples=self.analysis_config.min_sample_size)
            )
            major_patterns = {}
            for _, row in major_data.iterrows():
                key = f"{row['major_region']}_{row['hospital_type']}"
                major_patterns.setdefault(key, {})[row['ptmikts1']] = {
                    'probability': float(row['probability']),
                    'count': int(row['count'])
                }

            national_query = """
                SELECT
                    CASE
                        WHEN h.daily_capacity_mean >= 300 THEN 'large'
                        WHEN h.daily_capacity_mean >= 100 THEN 'medium'
                        ELSE 'small'
                    END as hospital_type,
                    n.ptmikts1 as ptmikts1,
                    COUNT(*) as count,
                    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY
                        CASE WHEN h.daily_capacity_mean >= 300 THEN 'large'
                             WHEN h.daily_capacity_mean >= 100 THEN 'medium'
                             ELSE 'small' END
                    ) as probability
                FROM {src} n
                JOIN {cap} h ON n.ptmiemcd = h.ptmiemcd
                WHERE n.ptmikts1 IN ('1', '2', '3', '4', '5')
                GROUP BY hospital_type, n.ptmikts1
            """

            national_data = self.db.fetch_dataframe(national_query.format(src=self.src_table, cap=capacity_tbl))
            national_patterns = {}
            for _, row in national_data.iterrows():
                national_patterns.setdefault(row['hospital_type'], {})[row['ptmikts1']] = {
                    'probability': float(row['probability']),
                    'count': int(row['count'])
                }

        else:
            # Fallback without capacity table: approximate using only patient region codes
            detailed_query = """
                SELECT
                    n.ptmizipc as region_code,
                    n.ptmikts1 as ptmikts1,
                    COUNT(*) as count,
                    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(
                        PARTITION BY n.ptmizipc
                    ) as probability
                FROM {src} n
                WHERE n.ptmikts1 IN ('1', '2', '3', '4', '5')
                  AND n.ptmizipc IS NOT NULL
                GROUP BY n.ptmizipc, n.ptmikts1
                HAVING COUNT(*) >= {min_samples}
            """
            detailed_data = self.db.fetch_dataframe(
                detailed_query.format(src=self.src_table, min_samples=self.analysis_config.min_sample_size)
            )
            detailed_patterns = {}
            for _, row in detailed_data.iterrows():
                key = f"{row['region_code']}_medium"
                detailed_patterns.setdefault(key, {})[row['ptmikts1']] = {
                    'probability': float(row['probability']),
                    'count': int(row['count'])
                }

            major_query = """
                SELECT
                    SUBSTR(n.ptmizipc, 1, 2) as major_region,
                    n.ptmikts1 as ptmikts1,
                    COUNT(*) as count,
                    COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(
                        PARTITION BY SUBSTR(n.ptmizipc, 1, 2)
                    ) as probability
                FROM {src} n
                WHERE n.ptmikts1 IN ('1', '2', '3', '4', '5')
                  AND n.ptmizipc IS NOT NULL
                GROUP BY major_region, n.ptmikts1
                HAVING COUNT(*) >= {min_samples}
            """
            major_data = self.db.fetch_dataframe(
                major_query.format(src=self.src_table, min_samples=self.analysis_config.min_sample_size)
            )
            major_patterns = {}
            for _, row in major_data.iterrows():
                key = f"{row['major_region']}_medium"
                major_patterns.setdefault(key, {})[row['ptmikts1']] = {
                    'probability': float(row['probability']),
                    'count': int(row['count'])
                }

            national_patterns = {'medium': {}}

        overall_query = """
            SELECT
                ptmikts1 as ptmikts1,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM {src}
            WHERE ptmikts1 IN ('1', '2', '3', '4', '5')
            GROUP BY ptmikts1
        """

        overall_data = self.db.fetch_dataframe(overall_query.format(src=self.src_table))
        overall_pattern = {}
        for _, row in overall_data.iterrows():
            overall_pattern[row['ptmikts1']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }

        return {
            'detailed_patterns': detailed_patterns,
            'major_patterns': major_patterns,
            'national_patterns': national_patterns,
            'overall_pattern': overall_pattern,
            'hierarchy_levels': 4,
            'analysis_method': 'hierarchical_ktas_analysis'
        }
    
    def get_hierarchical_ktas_distribution(self, region_code: str, hospital_type: str) -> Dict[str, float]:
        """계층적 KTAS 분포 조회"""
        # KTAS 패턴이 캐시되지 않은 경우 분석 수행
        if 'ktas_distributions' not in self._patterns_cache:
            ktas_patterns = self.analyze_ktas_distributions()
            self._patterns_cache['ktas_distributions'] = ktas_patterns
        else:
            ktas_patterns = self._patterns_cache['ktas_distributions']
        
        # 1단계: 소분류 (4자리 지역코드 + 병원유형)
        detailed_key = f"{region_code}_{hospital_type}"
        if detailed_key in ktas_patterns['detailed_patterns']:
            pattern = ktas_patterns['detailed_patterns'][detailed_key]
            self.logger.debug(f"Using detailed pattern for {detailed_key}")
            return {k: v['probability'] for k, v in pattern.items()}
        
        # 2단계: 대분류 (첫 2자리 + 병원유형)
        major_region = region_code[:2] if len(region_code) >= 2 else region_code
        major_key = f"{major_region}_{hospital_type}"
        if major_key in ktas_patterns['major_patterns']:
            pattern = ktas_patterns['major_patterns'][major_key]
            self.logger.debug(f"Using major pattern for {major_key}")
            return {k: v['probability'] for k, v in pattern.items()}
        
        # 3단계: 전국 (병원유형별)
        if hospital_type in ktas_patterns['national_patterns']:
            pattern = ktas_patterns['national_patterns'][hospital_type]
            self.logger.debug(f"Using national pattern for {hospital_type}")
            return {k: v['probability'] for k, v in pattern.items()}
        
        # 4단계: 최종 대안 (전체 평균)
        pattern = ktas_patterns['overall_pattern']
        self.logger.debug("Using overall pattern (final fallback)")
        return {k: v['probability'] for k, v in pattern.items()}
    
    def analyze_regional_patterns(self) -> Dict[str, Any]:
        """지역별 패턴 분석"""
        self.logger.info("Analyzing regional patterns")
        
        # 지역별 기본 통계
        regional_query = """
            SELECT
                ptmizipc as region_code,
                COUNT(*) as total_visits,
                AVG(CASE WHEN ptmikts1 = '1' THEN 1.0 ELSE 0.0 END) as ktas1_rate,
                AVG(CASE WHEN ptmikts1 IN ('1', '2') THEN 1.0 ELSE 0.0 END) as urgent_rate,
                COUNT(DISTINCT ptmiemcd) as unique_hospitals,
                AVG(CASE WHEN ptmisexx = '1' THEN 1.0 ELSE 0.0 END) as male_ratio
            FROM {src}
            WHERE ptmizipc IS NOT NULL
            GROUP BY ptmizipc
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)
        
        regional_data = self.db.fetch_dataframe(regional_query)
        
        patterns = {}
        for _, row in regional_data.iterrows():
            patterns[str(row['region_code'])] = {
                'total_visits': int(row['total_visits']),
                'ktas1_rate': float(row['ktas1_rate']),
                'urgent_rate': float(row['urgent_rate']),
                'unique_hospitals': int(row['unique_hospitals']),
                'male_ratio': float(row['male_ratio'])
            }
        
        return {
            'patterns': patterns,
            'total_regions': len(patterns),
            'analysis_method': 'regional_statistics'
        }
    
    def analyze_demographic_patterns(self) -> Dict[str, Any]:
        """인구통계학적 패턴 분석"""
        self.logger.info("Analyzing demographic patterns")
        
        # 연령-성별 기반 패턴
        demo_query = """
            SELECT
                ptmibrtd as ptmibrtd,
                ptmisexx as ptmisexx,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability,
                AVG(CASE WHEN ptmikts1 IN ('1', '2') THEN 1.0 ELSE 0.0 END) as urgent_rate,
                MODE() WITHIN GROUP (ORDER BY ptmimnsy) as common_symptom,
                MODE() WITHIN GROUP (ORDER BY ptmidept) as common_department
            FROM {src}
            WHERE ptmibrtd IS NOT NULL AND ptmisexx IS NOT NULL
            GROUP BY ptmibrtd, ptmisexx
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)
        
        demo_data = self.db.fetch_dataframe(demo_query)
        
        patterns = {}
        for _, row in demo_data.iterrows():
            key = f"{row['ptmibrtd']}_{row['ptmisexx']}"
            patterns[key] = {
                'count': int(row['count']),
                'probability': float(row['probability']),
                'urgent_rate': float(row['urgent_rate']),
                'common_symptom': row['common_symptom'],
                'common_department': row['common_department']
            }
        
        return {
            'patterns': patterns,
            'total_combinations': len(patterns),
            'analysis_method': 'age_sex_demographics'
        }
    
    def analyze_temporal_patterns(self) -> Dict[str, Any]:
        """시간 패턴 분석"""
        self.logger.info("Analyzing temporal patterns")
        
        # 월별 패턴
        monthly_query = """
            SELECT
                EXTRACT(MONTH FROM STRPTIME(ptmiindt, '%Y%m%d')) as month,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM {src}
            WHERE ptmiindt IS NOT NULL
            GROUP BY month
            ORDER BY month
        """
        
        monthly_data = self.db.fetch_dataframe(monthly_query.format(src=self.src_table))
        monthly_pattern = {}
        for _, row in monthly_data.iterrows():
            monthly_pattern[int(row['month'])] = {
                'count': int(row['count']),
                'probability': float(row['probability'])
            }
        
        # 요일별 패턴
        weekday_query = """
            SELECT
                EXTRACT(DOW FROM STRPTIME(ptmiindt, '%Y%m%d')) as day_of_week,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM {src}
            WHERE ptmiindt IS NOT NULL
            GROUP BY day_of_week
            ORDER BY day_of_week
        """
        
        weekday_data = self.db.fetch_dataframe(weekday_query.format(src=self.src_table))
        weekday_pattern = {}
        for _, row in weekday_data.iterrows():
            weekday_pattern[int(row['day_of_week'])] = {
                'count': int(row['count']),
                'probability': float(row['probability'])
            }
        
        # 시간대별 패턴
        hourly_query = """
            SELECT
                EXTRACT(HOUR FROM STRPTIME(ptmiintm, '%H%M')) as hour,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM {src}
            WHERE ptmiintm IS NOT NULL AND ptmiintm != ''
            GROUP BY hour
            ORDER BY hour
        """
        
        hourly_data = self.db.fetch_dataframe(hourly_query.format(src=self.src_table))
        hourly_pattern = {}
        for _, row in hourly_data.iterrows():
            hourly_pattern[int(row['hour'])] = {
                'count': int(row['count']),
                'probability': float(row['probability'])
            }
        
        return {
            'monthly_pattern': monthly_pattern,
            'weekday_pattern': weekday_pattern,
            'hourly_pattern': hourly_pattern,
            'analysis_method': 'temporal_distributions'
        }

    def analyze_temporal_conditional_patterns(self) -> Dict[str, Any]:
        """조건부 시간 패턴 분석

        학습되는 주요 분포:
        - month × hour
        - dow × hour
        - ptmikts1 × hour
        - ptmibrtd × ptmikts1 × hour
        - ptmibrtd × ptmisexx × ptmikts1 × hour
        - month × dow × hour
        - 월/요일 결합 + 주된 임상 상태 분해
        """
        self.logger.info("Analyzing conditional temporal patterns for joint constraints")

        query = """
            SELECT
                COALESCE(ptmibrtd, 'unknown') AS ptmibrtd,
                COALESCE(ptmisexx, 'unknown') AS ptmisexx,
                COALESCE(ptmikts1, 'unknown') AS ptmikts1,
                EXTRACT(MONTH FROM STRPTIME(ptmiindt, '%Y%m%d')) AS month,
                EXTRACT(DOW FROM STRPTIME(ptmiindt, '%Y%m%d')) AS dow,
                EXTRACT(HOUR FROM STRPTIME(ptmiintm, '%H%M')) AS hour,
                COUNT(*) AS count
            FROM {src}
            WHERE ptmiindt IS NOT NULL
              AND ptmiintm IS NOT NULL
              AND ptmiintm != ''
              AND ptmibrtd IS NOT NULL
              AND ptmisexx IS NOT NULL
              AND ptmikts1 IS NOT NULL
            GROUP BY
                ptmibrtd,
                ptmisexx,
                ptmikts1,
                month,
                dow,
                hour
            HAVING COUNT(*) >= {min_samples}
        """.format(src=self.src_table, min_samples=self.analysis_config.min_sample_size)

        cond_data = self.db.fetch_dataframe(query)

        if cond_data.empty:
            self.logger.warning("Conditional temporal query returned empty result")
            return {
                'patterns': {},
                'analysis_method': 'conditional_temporal_distributions',
            }

        # 날짜 기반 조건 분포
        month_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['month'],
            target_col='hour',
        )
        dow_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['dow'],
            target_col='hour',
        )
        month_dow_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['month', 'dow'],
            target_col='hour',
        )

        # 임상 연관 조건 분포
        ktas_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['ptmikts1'],
            target_col='hour',
        )
        age_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['ptmibrtd'],
            target_col='hour',
        )
        age_sex_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['ptmibrtd', 'ptmisexx'],
            target_col='hour',
        )
        ktas_age_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['ptmikts1', 'ptmibrtd'],
            target_col='hour',
        )
        age_sex_ktas_hour_patterns = self._build_conditional_distribution(
            cond_data,
            key_cols=['ptmibrtd', 'ptmisexx', 'ptmikts1'],
            target_col='hour',
        )

        return {
            'patterns': {
                'month_hour': month_hour_patterns,
                'dow_hour': dow_hour_patterns,
                'month_dow_hour': month_dow_hour_patterns,
                'ktas_hour': ktas_hour_patterns,
                'age_hour': age_hour_patterns,
                'age_sex_hour': age_sex_hour_patterns,
                'ktas_age_hour': ktas_age_hour_patterns,
                'age_sex_ktas_hour': age_sex_ktas_hour_patterns,
            },
            'analysis_method': 'conditional_temporal_distributions',
            'min_sample_size': self.analysis_config.min_sample_size,
        }

    def _build_conditional_distribution(
        self,
        df: pd.DataFrame,
        key_cols: List[str],
        target_col: str,
    ) -> Dict[str, Any]:
        """조건부 분포를 학습해 확률 맵으로 변환."""
        distributions: Dict[str, Any] = {}
        if df.empty:
            return distributions

        if 'count' not in df.columns:
            raise ValueError(
                f"_build_conditional_distribution requires a 'count' column; "
                f"found: {list(df.columns)}"
            )

        group_cols = key_cols + [target_col]
        # Sum actual visit counts (not row counts) when re-aggregating
        # The input df has a 'count' column from the SQL GROUP BY query
        counts = (
            df[group_cols + ['count']]
            .rename(columns={target_col: 'hour'})
            .groupby(key_cols + ['hour'], dropna=False)['count']
            .sum()
            .rename('count')
            .reset_index()
        )

        for keys, grouped in counts.groupby(key_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            total_count = int(grouped['count'].sum())
            if total_count <= 0:
                continue

            pattern = {}
            for _, row in grouped.iterrows():
                hour = int(row['hour'])
                count = int(row['count'])
                pattern[hour] = {
                    'count': count,
                    'probability': count / total_count,
                }

            key = '|'.join([str(k) for k in keys])
            distributions[key] = {
                'total_count': total_count,
                'patterns': pattern,
            }

        return distributions

    def analyze_visit_method_patterns(self) -> Dict[str, Any]:
        """연령대별 내원수단 분포 P(ptmiinmn | ptmibrtd)"""
        self.logger.info("Analyzing visit method patterns by age group")

        query = """
            SELECT
                ptmibrtd as ptmibrtd,
                ptmiinmn as ptmiinmn,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY ptmibrtd) as probability
            FROM {src}
            WHERE ptmibrtd IS NOT NULL AND ptmiinmn IS NOT NULL AND ptmiinmn != '' AND ptmiinmn != '-'
            GROUP BY ptmibrtd, ptmiinmn
            HAVING COUNT(*) >= {min_samples}
            ORDER BY ptmibrtd
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)

        df = self.db.fetch_dataframe(query)

        patterns: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = str(row['ptmibrtd'])
            patterns.setdefault(key, {})[row['ptmiinmn']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }

        return {
            'patterns': patterns,
            'analysis_method': 'visit_method_by_age'
        }

    def analyze_chief_complaint_patterns(self) -> Dict[str, Any]:
        """KTAS/연령/성별 조건부 주증상 분포 P(ptmimnsy | ptmikts1, ptmibrtd, ptmisexx)

        계층적 대안:
          L1: ktas_age_sex  →  L2: ktas_age  →  L3: ktas  →  L4: age_sex (전체)
        """
        self.logger.info("Analyzing chief complaint patterns by KTAS, age, sex")

        query = """
            SELECT
                ptmikts1,
                ptmibrtd,
                ptmisexx,
                ptmimnsy,
                COUNT(*) as count
            FROM {src}
            WHERE ptmikts1 IN ('1','2','3','4','5')
              AND ptmibrtd IS NOT NULL AND ptmisexx IS NOT NULL
              AND ptmimnsy IS NOT NULL AND ptmimnsy != '' AND ptmimnsy != '-'
            GROUP BY ptmikts1, ptmibrtd, ptmisexx, ptmimnsy
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)

        df = self.db.fetch_dataframe(query)

        # Build multi-level lookup
        patterns: Dict[str, Dict[str, Any]] = {}
        # Aggregate at multiple levels for fallback
        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            ktas = str(row['ptmikts1'])
            age = str(row['ptmibrtd'])
            sex = str(row['ptmisexx'])
            sym = str(row['ptmimnsy'])
            cnt = int(row['count'])

            # L1: ktas_age_sex
            agg[f"{ktas}_{age}_{sex}"][sym] += cnt
            # L2: ktas_age
            agg[f"{ktas}_{age}"][sym] += cnt
            # L3: ktas only
            agg[f"{ktas}"][sym] += cnt
            # L4: age_sex (fallback for missing KTAS)
            agg[f"all_{age}_{sex}"][sym] += cnt

        # Convert counts to probabilities
        for key, sym_counts in agg.items():
            total = sum(sym_counts.values())
            patterns[key] = {
                sym: {'probability': cnt / total, 'count': cnt}
                for sym, cnt in sym_counts.items()
            }

        return {
            'patterns': patterns,
            'analysis_method': 'chief_complaint_by_ktas_age_sex'
        }

    def analyze_department_patterns(self) -> Dict[str, Any]:
        """KTAS/주증상 조건부 진료과 분포 P(ptmidept | ptmikts1, ptmimnsy)

        계층적 대안:
          L1: ktas_symptom  →  L2: ktas  →  L3: symptom  →  L4: 전체
        """
        self.logger.info("Analyzing department patterns by KTAS and chief complaint")

        query = """
            SELECT
                ptmikts1,
                ptmimnsy,
                ptmidept,
                COUNT(*) as count
            FROM {src}
            WHERE ptmikts1 IN ('1','2','3','4','5')
              AND ptmimnsy IS NOT NULL AND ptmimnsy != '' AND ptmimnsy != '-'
              AND ptmidept IS NOT NULL AND ptmidept != '' AND ptmidept != '-'
            GROUP BY ptmikts1, ptmimnsy, ptmidept
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)

        df = self.db.fetch_dataframe(query)

        from collections import defaultdict
        agg = defaultdict(lambda: defaultdict(int))

        for _, row in df.iterrows():
            ktas = str(row['ptmikts1'])
            sym = str(row['ptmimnsy'])
            dept = str(row['ptmidept'])
            cnt = int(row['count'])

            agg[f"{ktas}_{sym}"][dept] += cnt
            agg[f"{ktas}"][dept] += cnt
            agg[f"sym_{sym}"][dept] += cnt
            agg["all"][dept] += cnt

        patterns: Dict[str, Dict[str, Any]] = {}
        for key, dept_counts in agg.items():
            total = sum(dept_counts.values())
            patterns[key] = {
                dept: {'probability': cnt / total, 'count': cnt}
                for dept, cnt in dept_counts.items()
            }

        return {
            'patterns': patterns,
            'analysis_method': 'department_by_ktas_symptom'
        }

    def analyze_treatment_result_patterns(self) -> Dict[str, Any]:
        """KTAS/연령별 치료결과 분포 P(ptmiemrt | ptmikts1, ptmibrtd)"""
        self.logger.info("Analyzing treatment result patterns by KTAS and age group")

        query = """
            SELECT
                ptmikts1 as ptmikts1,
                ptmibrtd as ptmibrtd,
                ptmiemrt as ptmiemrt,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY ptmikts1, ptmibrtd) as probability
            FROM {src}
            WHERE ptmikts1 IN ('1','2','3','4','5')
              AND ptmibrtd IS NOT NULL
              AND ptmiemrt IS NOT NULL AND ptmiemrt != ''
            GROUP BY ptmikts1, ptmibrtd, ptmiemrt
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size, src=self.src_table)

        df = self.db.fetch_dataframe(query)

        patterns: Dict[str, Dict[str, Any]] = {}
        for _, row in df.iterrows():
            key = f"{row['ptmikts1']}_{row['ptmibrtd']}"
            patterns.setdefault(key, {})[row['ptmiemrt']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }

        return {
            'patterns': patterns,
            'analysis_method': 'treatment_result_by_ktas_age'
        }
    
    def _create_hierarchical_patterns(self, patterns: Dict[str, Any], 
                                    key_type: str) -> Dict[str, Any]:
        """계층적 패턴 생성"""
        if key_type != 'region':
            return {}
        
        hierarchical = {}
        
        # 대분류별로 그룹화
        for region_code, region_data in patterns.items():
            if len(region_code) >= 2:
                major_region = region_code[:2]
                if major_region not in hierarchical:
                    hierarchical[major_region] = {
                        'hospitals': {},
                        'total_visits': 0,
                        'sub_regions': []
                    }
                
                # 병원 정보 합병
                for hospital, hospital_data in region_data['hospitals'].items():
                    if hospital not in hierarchical[major_region]['hospitals']:
                        hierarchical[major_region]['hospitals'][hospital] = {
                            'probability': 0,
                            'visit_count': 0
                        }
                    
                    hierarchical[major_region]['hospitals'][hospital]['visit_count'] += hospital_data['visit_count']
                
                hierarchical[major_region]['total_visits'] += region_data['total_visits']
                hierarchical[major_region]['sub_regions'].append(region_code)
        
        # 확률 재계산
        for major_region, major_data in hierarchical.items():
            total_visits = major_data['total_visits']
            if total_visits > 0:
                for hospital in major_data['hospitals']:
                    visit_count = major_data['hospitals'][hospital]['visit_count']
                    major_data['hospitals'][hospital]['probability'] = visit_count / total_visits
        
        return hierarchical
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """패턴 분석 요약 정보"""
        try:
            # 캐시된 패턴들의 요약 정보
            summary = {
                'cache_status': {},
                'pattern_counts': {},
                'last_analysis': None
            }
            
            # 캐시 메타데이터에서 정보 추출
            for cache_key, entry in self.cache.metadata["cache_entries"].items():
                analysis_type = entry["analysis_type"]
                summary['cache_status'][analysis_type] = {
                    'cached': True,
                    'created_at': entry['created_at']
                }
            
            summary['last_analysis'] = self.cache.metadata.get("last_updated")
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate pattern summary: {e}")
            return {"error": str(e)}

    def analyze_missing_value_rates(self) -> Dict[str, Any]:
        """결측값 비율 분석 — 구조적 동시 결측 패턴 포함.

        Returns a dict with:
          - 'independent': {col: missing_rate} for variables with independent missingness
          - 'correlated_group': info about the KTAS/symptom/dept co-missing cluster
          - 'conditional': rates conditional on KTAS being valid vs missing
        """
        self.logger.info("Analyzing missing value rates")
        src = self.src_table

        # Variables to analyze (NEDIS 4.0 column names as they appear via VIEW)
        # The VIEW aliases old→new, so we query using NEDIS 4.0 names
        cat_vars = ['ptmikts1', 'ptmimnsy', 'ptmidept', 'ptmizipc', 'ptmiemrt', 'ptmiinmn']

        try:
            total = self.db.fetch_dataframe(
                f"SELECT COUNT(*) as cnt FROM {src}"
            )['cnt'].iloc[0]

            # Per-variable missing rate
            independent = {}
            for col in cat_vars:
                try:
                    q = f"""
                        SELECT COUNT(*) as miss
                        FROM {src}
                        WHERE {col} IS NULL OR CAST({col} AS VARCHAR) IN ('', '-')
                    """
                    miss = self.db.fetch_dataframe(q)['miss'].iloc[0]
                    independent[col] = float(miss) / total
                except Exception:
                    independent[col] = 0.0

            # Correlated group: KTAS, chief complaint, department
            try:
                co_q = f"""
                    SELECT
                        COUNT(*) as total,
                        SUM(CASE WHEN CAST(ptmikts1 AS VARCHAR) IN ('','-')
                                   OR ptmikts1 IS NULL THEN 1 ELSE 0 END) as ktas_miss,
                        SUM(CASE WHEN (CAST(ptmikts1 AS VARCHAR) IN ('','-') OR ptmikts1 IS NULL)
                                  AND (CAST(ptmimnsy AS VARCHAR) IN ('','-') OR ptmimnsy IS NULL)
                                  AND (CAST(ptmidept AS VARCHAR) IN ('','-') OR ptmidept IS NULL)
                             THEN 1 ELSE 0 END) as all_three_miss
                    FROM {src}
                """
                co = self.db.fetch_dataframe(co_q)
                ktas_miss = int(co['ktas_miss'].iloc[0])
                all_three = int(co['all_three_miss'].iloc[0])
                correlated_group = {
                    'members': ['ptmikts1', 'ptmimnsy', 'ptmidept'],
                    'primary_missing_rate': float(ktas_miss) / total,
                    'co_missing_rate': float(all_three) / total,
                    'co_missing_given_primary': float(all_three) / max(ktas_miss, 1),
                }
            except Exception:
                correlated_group = {}

            # Conditional rates: when KTAS is valid
            conditional = {}
            try:
                cond_q = f"""
                    SELECT
                        COUNT(*) as valid_ktas,
                        SUM(CASE WHEN CAST(ptmimnsy AS VARCHAR) IN ('','-')
                                   OR ptmimnsy IS NULL THEN 1 ELSE 0 END) as mnsy_miss,
                        SUM(CASE WHEN CAST(ptmidept AS VARCHAR) IN ('','-')
                                   OR ptmidept IS NULL THEN 1 ELSE 0 END) as dept_miss,
                        SUM(CASE WHEN CAST(ptmizipc AS VARCHAR) IN ('','-')
                                   OR ptmizipc IS NULL THEN 1 ELSE 0 END) as zipc_miss,
                        SUM(CASE WHEN CAST(ptmiemrt AS VARCHAR) IN ('','-')
                                   OR ptmiemrt IS NULL THEN 1 ELSE 0 END) as emrt_miss,
                        SUM(CASE WHEN CAST(ptmiinmn AS VARCHAR) IN ('','-')
                                   OR ptmiinmn IS NULL THEN 1 ELSE 0 END) as inmn_miss
                    FROM {src}
                    WHERE ptmikts1 IS NOT NULL
                      AND CAST(ptmikts1 AS VARCHAR) NOT IN ('', '-')
                      AND ptmikts1 IN ('1','2','3','4','5')
                """
                cond = self.db.fetch_dataframe(cond_q)
                vk = int(cond['valid_ktas'].iloc[0])
                if vk > 0:
                    conditional = {
                        'given_ktas_valid': {
                            'ptmimnsy': float(cond['mnsy_miss'].iloc[0]) / vk,
                            'ptmidept': float(cond['dept_miss'].iloc[0]) / vk,
                            'ptmizipc': float(cond['zipc_miss'].iloc[0]) / vk,
                            'ptmiemrt': float(cond['emrt_miss'].iloc[0]) / vk,
                            'ptmiinmn': float(cond['inmn_miss'].iloc[0]) / vk,
                        }
                    }
            except Exception:
                pass

            result = {
                'independent': independent,
                'correlated_group': correlated_group,
                'conditional': conditional,
                'total_records': int(total),
            }
            self.logger.info(
                "Missing value analysis: KTAS %.1f%%, co-missing %.1f%%",
                independent.get('ptmikts1', 0) * 100,
                correlated_group.get('co_missing_rate', 0) * 100,
            )
            return result

        except Exception as e:
            self.logger.error(f"Failed to analyze missing value rates: {e}")
            return {"error": str(e), "independent": {}, "correlated_group": {}, "conditional": {}}
