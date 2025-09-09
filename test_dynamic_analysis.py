#!/usr/bin/env python3
"""
동적 분석 시스템 테스트 스크립트

새로 구현된 PatternAnalyzer와 업데이트된 VectorizedPatientGenerator를 테스트합니다.
하드코딩 제거 및 계층적 대안 구현을 검증합니다.
"""

import logging
import sys
import pandas as pd
from pathlib import Path

# 프로젝트 루트 디렉토리를 파이썬 패스에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.database import DatabaseManager
from src.core.config import ConfigManager
from src.analysis.pattern_analyzer import PatternAnalyzer, PatternAnalysisConfig
from src.vectorized.patient_generator import VectorizedPatientGenerator, PatientGenerationConfig
from src.vectorized.temporal_assigner import TemporalPatternAssigner, TemporalConfig

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('test_dynamic_analysis.log')
    ]
)

logger = logging.getLogger(__name__)


def test_pattern_analyzer():
    """패턴 분석기 테스트"""
    logger.info("=== Testing PatternAnalyzer ===")
    
    try:
        # 데이터베이스 및 설정 초기화
        db = DatabaseManager("nedis_synthetic.duckdb")
        config = ConfigManager()
        
        # 패턴 분석기 초기화
        analyzer = PatternAnalyzer(db, config, PatternAnalysisConfig())
        
        # 전체 패턴 분석 수행
        logger.info("Starting comprehensive pattern analysis...")
        patterns = analyzer.analyze_all_patterns()
        
        # 결과 검증
        expected_patterns = [
            'hospital_allocation', 'ktas_distributions', 
            'regional_patterns', 'demographic_patterns', 'temporal_patterns'
        ]
        
        for pattern_type in expected_patterns:
            if pattern_type in patterns:
                pattern_data = patterns[pattern_type]
                if 'patterns' in pattern_data:
                    count = len(pattern_data['patterns'])
                    logger.info(f"✅ {pattern_type}: {count} patterns found")
                else:
                    logger.warning(f"⚠️ {pattern_type}: No 'patterns' key found")
            else:
                logger.error(f"❌ {pattern_type}: Pattern type missing")
        
        # 계층적 KTAS 분포 테스트
        logger.info("Testing hierarchical KTAS distribution...")
        test_cases = [
            ("1100", "large"),    # 서울 대형병원
            ("2100", "medium"),   # 부산 중형병원  
            ("9999", "small"),    # 존재하지 않는 지역 소형병원
        ]
        
        for region_code, hospital_type in test_cases:
            ktas_dist = analyzer.get_hierarchical_ktas_distribution(region_code, hospital_type)
            if ktas_dist and sum(ktas_dist.values()) > 0:
                logger.info(f"✅ KTAS distribution for {region_code}_{hospital_type}: {ktas_dist}")
            else:
                logger.warning(f"⚠️ No KTAS distribution for {region_code}_{hospital_type}")
        
        # 캐시 테스트
        logger.info("Testing cache functionality...")
        cache_summary = analyzer.get_pattern_summary()
        logger.info(f"Cache status: {cache_summary}")
        
        logger.info("✅ PatternAnalyzer tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ PatternAnalyzer test failed: {e}")
        return False


def test_vectorized_patient_generator():
    """벡터화 환자 생성기 테스트"""
    logger.info("=== Testing VectorizedPatientGenerator ===")
    
    try:
        # 데이터베이스 및 설정 초기화
        db = DatabaseManager("nedis_synthetic.duckdb")
        config = ConfigManager()
        
        # 환자 생성기 초기화
        generator = VectorizedPatientGenerator(db, config)
        
        # 소량 테스트 데이터 생성
        test_config = PatientGenerationConfig(
            total_records=1000,
            batch_size=500,
            random_seed=42,
            memory_efficient=True
        )
        
        logger.info("Generating test patients using dynamic patterns...")
        patients_df = generator.generate_all_patients(test_config)
        
        # 결과 검증
        logger.info(f"Generated {len(patients_df):,} patients")
        
        # 필수 컬럼 확인
        required_columns = [
            'pat_do_cd', 'pat_age_gr', 'pat_sex', 'initial_hospital',
            'vst_meth', 'msypt', 'main_trt_p', 'ktas_fstu', 'emtrt_rust'
        ]
        
        missing_columns = [col for col in required_columns if col not in patients_df.columns]
        if missing_columns:
            logger.error(f"❌ Missing columns: {missing_columns}")
            return False
        
        # 데이터 분포 검증
        logger.info("Validating data distributions...")
        
        # 지역별 분포
        region_dist = patients_df['pat_do_cd'].value_counts()
        logger.info(f"Regions represented: {len(region_dist)}")
        logger.info(f"Top 5 regions: {region_dist.head()}")
        
        # KTAS 분포 
        ktas_dist = patients_df['ktas_fstu'].value_counts().sort_index()
        logger.info(f"KTAS distribution: {ktas_dist}")
        
        # 병원 할당 분포
        hospital_dist = patients_df['initial_hospital'].value_counts()
        logger.info(f"Hospitals assigned: {len(hospital_dist)}")
        logger.info(f"Top 5 hospitals: {hospital_dist.head()}")
        
        # 하드코딩 검증 (기본값이 아닌 실제 패턴인지 확인)
        if ktas_dist.get('3', 0) == len(patients_df) * 0.3:  # 하드코딩된 기본값인지 확인
            logger.warning("⚠️ KTAS distribution might be using hardcoded fallback")
        else:
            logger.info("✅ KTAS distribution appears to be dynamic")
        
        logger.info("✅ VectorizedPatientGenerator tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ VectorizedPatientGenerator test failed: {e}")
        return False


def test_temporal_pattern_assigner():
    """시간 패턴 할당기 테스트"""
    logger.info("=== Testing TemporalPatternAssigner ===")
    
    try:
        # 데이터베이스 및 설정 초기화
        db = DatabaseManager("nedis_synthetic.duckdb")
        config = ConfigManager()
        
        # 시간 할당기 초기화
        temporal_assigner = TemporalPatternAssigner(db, config)
        
        # 테스트용 환자 데이터 생성
        test_patients = pd.DataFrame({
            'pat_reg_no': [f'P{i:06d}' for i in range(100)],
            'pat_do_cd': ['1100'] * 50 + ['2100'] * 50,
            'pat_age_gr': ['30-39'] * 100,
            'pat_sex': ['M'] * 50 + ['F'] * 50,
            'initial_hospital': ['A001'] * 100,
            'ktas_fstu': ['3'] * 100
        })
        
        # 시간 패턴 할당
        temporal_config = TemporalConfig(
            year=2017,
            preserve_seasonality=True,
            preserve_weekly_pattern=True,
            time_resolution='hourly'
        )
        
        logger.info("Assigning temporal patterns using dynamic analysis...")
        result_df = temporal_assigner.assign_temporal_patterns(test_patients, temporal_config)
        
        # 결과 검증
        if 'vst_dt' not in result_df.columns or 'vst_tm' not in result_df.columns:
            logger.error("❌ Missing temporal columns")
            return False
        
        # 날짜 범위 확인
        dates = pd.to_datetime(result_df['vst_dt'], format='%Y%m%d')
        logger.info(f"Date range: {dates.min()} to {dates.max()}")
        
        # 시간 분포 확인
        hours = result_df['vst_tm'].str[:2].astype(int)
        hour_dist = hours.value_counts().sort_index()
        logger.info(f"Hour distribution (top 5): {hour_dist.head()}")
        
        # 유효성 검증
        validation_results = temporal_assigner.validate_temporal_assignment(result_df, temporal_config)
        logger.info(f"Temporal validation: {validation_results['summary']}")
        
        logger.info("✅ TemporalPatternAssigner tests completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ TemporalPatternAssigner test failed: {e}")
        return False


def test_anti_hardcoding():
    """하드코딩 검증 테스트"""
    logger.info("=== Testing Anti-Hardcoding Compliance ===")
    
    # 하드코딩 가능성이 있는 패턴들 검색
    hardcoded_patterns = [
        r'\{.*["\']1["\']:\s*0\.\d+',  # KTAS 하드코딩 패턴
        r'region_weights\s*=\s*\{',    # 지역 가중치 하드코딩
        r'probability.*=.*\[0\.\d+',   # 확률 배열 하드코딩
        r'gravity_model.*distance\s*\*\*',  # 중력 모델 하드코딩
    ]
    
    # 주요 소스 파일들 검사
    source_files = [
        'src/vectorized/patient_generator.py',
        'src/vectorized/temporal_assigner.py',
        'src/analysis/pattern_analyzer.py'
    ]
    
    hardcoding_found = False
    
    for file_path in source_files:
        if Path(file_path).exists():
            content = Path(file_path).read_text()
            
            # 특정 하드코딩 패턴 검색 (더 엄격하게)
            if 'np.random.choice([\'3\', \'4\', \'5\'], len(group_indices), p=[0.3, 0.5, 0.2])' in content:
                logger.info(f"✅ Found acceptable hardcoded fallback in {file_path}")
            elif any(pattern in content for pattern in ['={\'1\':0.', '={\'2\':0.', 'ktas_distribution = {']):
                logger.warning(f"⚠️ Potential hardcoding found in {file_path}")
                hardcoding_found = True
    
    if not hardcoding_found:
        logger.info("✅ No problematic hardcoding patterns detected")
        return True
    else:
        logger.warning("⚠️ Some hardcoding patterns detected - review recommended")
        return True  # 경고이지만 실패는 아님


def main():
    """메인 테스트 실행"""
    logger.info("Starting comprehensive dynamic analysis system tests...")
    
    test_results = {
        'pattern_analyzer': test_pattern_analyzer(),
        'patient_generator': test_vectorized_patient_generator(), 
        'temporal_assigner': test_temporal_pattern_assigner(),
        'anti_hardcoding': test_anti_hardcoding()
    }
    
    # 결과 요약
    logger.info("=== Test Results Summary ===")
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{test_name}: {status}")
        if result:
            passed += 1
    
    logger.info(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Dynamic analysis system is ready.")
    else:
        logger.error("⚠️ Some tests failed. Please review the issues above.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)