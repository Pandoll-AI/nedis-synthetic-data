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
    
    def analyze_all_patterns(self) -> Dict[str, Any]:
        """모든 패턴 분석 수행"""
        self.logger.info("Starting comprehensive pattern analysis")
        
        patterns = {}
        
        # 데이터 해시 계산
        data_hash = self.cache.get_data_hash(self.db, "nedis_original.nedis2017")
        
        # 각 패턴 분석
        analysis_methods = [
            ("hospital_allocation", self.analyze_hospital_allocation_patterns),
            ("ktas_distributions", self.analyze_ktas_distributions),
            ("regional_patterns", self.analyze_regional_patterns),
            ("demographic_patterns", self.analyze_demographic_patterns),
            ("temporal_patterns", self.analyze_temporal_patterns)
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
    
    def analyze_hospital_allocation_patterns(self) -> Dict[str, Any]:
        """병원 할당 패턴 분석 (지역 기반)"""
        self.logger.info("Analyzing hospital allocation patterns")
        
        # 지역별 병원 선택 패턴 분석
        query = """
            SELECT 
                pat_do_cd as region_code,
                emorg_cd as hospital_code,
                COUNT(*) as visit_count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY pat_do_cd) as region_probability
            FROM nedis_original.nedis2017
            WHERE pat_do_cd IS NOT NULL AND emorg_cd IS NOT NULL
            GROUP BY pat_do_cd, emorg_cd
            HAVING COUNT(*) >= {min_samples}
            ORDER BY pat_do_cd, visit_count DESC
        """.format(min_samples=self.analysis_config.min_sample_size)
        
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
        
        # 소분류 (4자리 지역코드) + 병원유형별 KTAS 분포
        detailed_query = """
            SELECT 
                n.pat_do_cd as region_code,
                n.emorg_cd as hospital_code,
                CASE 
                    WHEN h.daily_capacity_mean >= 100 THEN 'large'
                    WHEN h.daily_capacity_mean >= 50 THEN 'medium' 
                    ELSE 'small'
                END as hospital_type,
                n.ktas_fstu,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(
                    PARTITION BY h.pat_do_cd, 
                    CASE WHEN h.capacity_beds >= 300 THEN 'large'
                         WHEN h.capacity_beds >= 100 THEN 'medium' 
                         ELSE 'small' END
                ) as probability
            FROM nedis_original.nedis2017 n
            JOIN nedis_meta.hospital_capacity h ON n.emorg_cd = h.emorg_cd
            WHERE n.ktas_fstu IN ('1', '2', '3', '4', '5')
              AND h.pat_do_cd IS NOT NULL
            GROUP BY h.pat_do_cd, h.emorg_cd, hospital_type, n.ktas_fstu
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size)
        
        detailed_data = self.db.fetch_dataframe(detailed_query)
        
        # 1단계: 소분류 패턴
        detailed_patterns = {}
        for _, row in detailed_data.iterrows():
            key = f"{row['region_code']}_{row['hospital_type']}"
            if key not in detailed_patterns:
                detailed_patterns[key] = {}
            
            detailed_patterns[key][row['ktas_fstu']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }
        
        # 2단계: 대분류 (첫 2자리) 패턴
        major_query = """
            SELECT 
                SUBSTR(h.pat_do_cd, 1, 2) as major_region,
                CASE 
                    WHEN h.capacity_beds >= 300 THEN 'large'
                    WHEN h.capacity_beds >= 100 THEN 'medium' 
                    ELSE 'small'
                END as hospital_type,
                n.ktas_fstu,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(
                    PARTITION BY SUBSTR(h.pat_do_cd, 1, 2),
                    CASE WHEN h.capacity_beds >= 300 THEN 'large'
                         WHEN h.capacity_beds >= 100 THEN 'medium' 
                         ELSE 'small' END
                ) as probability
            FROM nedis_original.nedis2017 n
            JOIN nedis_meta.hospital_capacity h ON n.emorg_cd = h.emorg_cd
            WHERE n.ktas_fstu IN ('1', '2', '3', '4', '5')
              AND h.pat_do_cd IS NOT NULL
            GROUP BY major_region, hospital_type, n.ktas_fstu
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size)
        
        major_data = self.db.fetch_dataframe(major_query)
        
        major_patterns = {}
        for _, row in major_data.iterrows():
            key = f"{row['major_region']}_{row['hospital_type']}"
            if key not in major_patterns:
                major_patterns[key] = {}
            
            major_patterns[key][row['ktas_fstu']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }
        
        # 3단계: 전국 패턴
        national_query = """
            SELECT 
                CASE 
                    WHEN h.capacity_beds >= 300 THEN 'large'
                    WHEN h.capacity_beds >= 100 THEN 'medium' 
                    ELSE 'small'
                END as hospital_type,
                n.ktas_fstu,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER(PARTITION BY 
                    CASE WHEN h.capacity_beds >= 300 THEN 'large'
                         WHEN h.capacity_beds >= 100 THEN 'medium' 
                         ELSE 'small' END
                ) as probability
            FROM nedis_original.nedis2017 n
            JOIN nedis_meta.hospital_capacity h ON n.emorg_cd = h.emorg_cd
            WHERE n.ktas_fstu IN ('1', '2', '3', '4', '5')
            GROUP BY hospital_type, n.ktas_fstu
        """
        
        national_data = self.db.fetch_dataframe(national_query)
        
        national_patterns = {}
        for _, row in national_data.iterrows():
            key = row['hospital_type']
            if key not in national_patterns:
                national_patterns[key] = {}
            
            national_patterns[key][row['ktas_fstu']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }
        
        # 4단계: 최종 대안 (전체 평균)
        overall_query = """
            SELECT 
                ktas_fstu,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM nedis_original.nedis2017
            WHERE ktas_fstu IN ('1', '2', '3', '4', '5')
            GROUP BY ktas_fstu
        """
        
        overall_data = self.db.fetch_dataframe(overall_query)
        overall_pattern = {}
        for _, row in overall_data.iterrows():
            overall_pattern[row['ktas_fstu']] = {
                'probability': float(row['probability']),
                'count': int(row['count'])
            }
        
        return {
            'detailed_patterns': detailed_patterns,  # 소분류
            'major_patterns': major_patterns,        # 대분류
            'national_patterns': national_patterns,  # 병원유형별 전국
            'overall_pattern': overall_pattern,      # 전체 평균
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
                pat_do_cd as region_code,
                COUNT(*) as total_visits,
                AVG(CASE WHEN ktas_fstu = '1' THEN 1.0 ELSE 0.0 END) as ktas1_rate,
                AVG(CASE WHEN ktas_fstu IN ('1', '2') THEN 1.0 ELSE 0.0 END) as urgent_rate,
                COUNT(DISTINCT emorg_cd) as unique_hospitals,
                AVG(CASE WHEN pat_sex = 'M' THEN 1.0 ELSE 0.0 END) as male_ratio
            FROM nedis_original.nedis2017
            WHERE pat_do_cd IS NOT NULL
            GROUP BY pat_do_cd
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size)
        
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
                pat_age_gr,
                pat_sex,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability,
                AVG(CASE WHEN ktas_fstu IN ('1', '2') THEN 1.0 ELSE 0.0 END) as urgent_rate,
                MODE() WITHIN GROUP (ORDER BY msypt) as common_symptom,
                MODE() WITHIN GROUP (ORDER BY main_trt_p) as common_department
            FROM nedis_original.nedis2017
            WHERE pat_age_gr IS NOT NULL AND pat_sex IS NOT NULL
            GROUP BY pat_age_gr, pat_sex
            HAVING COUNT(*) >= {min_samples}
        """.format(min_samples=self.analysis_config.min_sample_size)
        
        demo_data = self.db.fetch_dataframe(demo_query)
        
        patterns = {}
        for _, row in demo_data.iterrows():
            key = f"{row['pat_age_gr']}_{row['pat_sex']}"
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
                EXTRACT(MONTH FROM STRPTIME(vst_dt, '%Y%m%d')) as month,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM nedis_original.nedis2017
            WHERE vst_dt IS NOT NULL
            GROUP BY month
            ORDER BY month
        """
        
        monthly_data = self.db.fetch_dataframe(monthly_query)
        monthly_pattern = {}
        for _, row in monthly_data.iterrows():
            monthly_pattern[int(row['month'])] = {
                'count': int(row['count']),
                'probability': float(row['probability'])
            }
        
        # 요일별 패턴
        weekday_query = """
            SELECT 
                EXTRACT(DOW FROM STRPTIME(vst_dt, '%Y%m%d')) as day_of_week,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM nedis_original.nedis2017
            WHERE vst_dt IS NOT NULL
            GROUP BY day_of_week
            ORDER BY day_of_week
        """
        
        weekday_data = self.db.fetch_dataframe(weekday_query)
        weekday_pattern = {}
        for _, row in weekday_data.iterrows():
            weekday_pattern[int(row['day_of_week'])] = {
                'count': int(row['count']),
                'probability': float(row['probability'])
            }
        
        # 시간대별 패턴
        hourly_query = """
            SELECT 
                EXTRACT(HOUR FROM STRPTIME(vst_tm, '%H%M')) as hour,
                COUNT(*) as count,
                COUNT(*) * 1.0 / SUM(COUNT(*)) OVER() as probability
            FROM nedis_original.nedis2017
            WHERE vst_tm IS NOT NULL AND vst_tm != ''
            GROUP BY hour
            ORDER BY hour
        """
        
        hourly_data = self.db.fetch_dataframe(hourly_query)
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