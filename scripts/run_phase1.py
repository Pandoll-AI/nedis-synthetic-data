"""
Phase 1 실행 스크립트: 데이터 프로파일링 및 메타데이터 추출

NEDIS 원본 데이터에서 인구학적 통계, 병원 용량 정보, 조건부 확률을 추출하여
합성 데이터 생성의 기초가 되는 메타데이터를 생성합니다.
"""

import sys
import os
from pathlib import Path

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

import logging
from datetime import datetime
import json
import time
import traceback
from typing import Dict, Any, Tuple

from core.database import DatabaseManager
from core.config import ConfigManager  
from population.profiler import NEDISProfiler
from clinical.conditional_probability import ConditionalProbabilityExtractor


def setup_logging():
    """로깅 설정"""
    # logs 디렉토리 생성
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # 로그 파일명에 타임스탬프 포함
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = logs_dir / f"phase1_{timestamp}.log"
    
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file


def check_prerequisites(db_manager: DatabaseManager) -> bool:
    """전제 조건 확인"""
    logger = logging.getLogger(__name__)
    
    try:
        # 원본 데이터 테이블 존재 확인
        if not db_manager.table_exists("nedis_original.nedis2017"):
            logger.error("Original data table 'nedis_original.nedis2017' not found")
            logger.error("Please load the original NEDIS 2017 data first")
            return False
            
        # 기본 레코드 수 확인
        count_query = "SELECT COUNT(*) as total FROM nedis_original.nedis2017"
        total_records = db_manager.fetch_dataframe(count_query)['total'][0]
        
        if total_records == 0:
            logger.error("Original data table is empty")
            return False
            
        logger.info(f"Found {total_records:,} records in original data")
        
        # 필수 스키마 테이블 확인
        required_tables = [
            "nedis_meta.population_margins",
            "nedis_meta.hospital_capacity", 
            "nedis_meta.ktas_conditional_prob",
            "nedis_meta.diagnosis_conditional_prob"
        ]
        
        for table in required_tables:
            if not db_manager.table_exists(table):
                logger.warning(f"Meta table '{table}' not found - will be created")
                
        return True
        
    except Exception as e:
        logger.error(f"Prerequisites check failed: {e}")
        return False


def update_pipeline_progress(db_manager: DatabaseManager, step_name: str, 
                           status: str, records_processed: int = 0, 
                           error_message: str = None):
    """파이프라인 진행 상황 업데이트"""
    try:
        if status == 'running':
            query = """
            INSERT OR REPLACE INTO nedis_meta.pipeline_progress 
                (step_name, status, start_time, records_processed)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            """
            db_manager.execute_query(query, [step_name, status, records_processed])
        
        elif status in ['completed', 'failed']:
            query = """
            UPDATE nedis_meta.pipeline_progress 
            SET status = ?, end_time = CURRENT_TIMESTAMP, 
                records_processed = ?, error_message = ?
            WHERE step_name = ?
            """
            db_manager.execute_query(query, [status, records_processed, error_message, step_name])
            
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to update pipeline progress: {e}")


def run_profiling_step(db_manager: DatabaseManager, config: ConfigManager) -> Tuple[bool, Dict[str, Any]]:
    """인구학적 프로파일링 실행"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Step 1: Population and Hospital Profiling ===")
    update_pipeline_progress(db_manager, "population_profiling", "running")
    
    try:
        profiler = NEDISProfiler(db_manager, config)
        
        # 1-1. 인구학적 마진 추출
        logger.info("Extracting population margins...")
        start_time = time.time()
        
        if not profiler.extract_population_margins():
            raise Exception("Population margins extraction failed")
            
        # 레코드 수 확인
        margin_count = db_manager.get_table_count("nedis_meta.population_margins")
        logger.info(f"✓ Created {margin_count:,} population margin combinations in {time.time() - start_time:.2f}s")
        
        # 1-2. 병원 통계 추출
        logger.info("Extracting hospital statistics...")
        start_time = time.time()
        
        if not profiler.extract_hospital_statistics():
            raise Exception("Hospital statistics extraction failed")
            
        # 통계가 있는 병원 수 확인
        hospital_stats_count = db_manager.fetch_dataframe("""
            SELECT COUNT(*) as count FROM nedis_meta.hospital_capacity 
            WHERE daily_capacity_mean IS NOT NULL
        """)['count'][0]
        
        logger.info(f"✓ Updated statistics for {hospital_stats_count:,} hospitals in {time.time() - start_time:.2f}s")
        
        # 1-3. 기본 리포트 생성
        logger.info("Generating basic profile report...")
        report = profiler.generate_basic_report()
        
        update_pipeline_progress(db_manager, "population_profiling", "completed", 
                               int(margin_count + hospital_stats_count))
        
        return True, report
        
    except Exception as e:
        logger.error(f"Population profiling failed: {e}")
        update_pipeline_progress(db_manager, "population_profiling", "failed", 0, str(e))
        return False, {'error': str(e)}


def run_probability_extraction_step(db_manager: DatabaseManager, config: ConfigManager) -> bool:
    """조건부 확률 추출 실행"""
    logger = logging.getLogger(__name__)
    
    logger.info("=== Step 2: Conditional Probability Extraction ===")
    update_pipeline_progress(db_manager, "conditional_probabilities", "running")
    
    try:
        extractor = ConditionalProbabilityExtractor(db_manager, config)
        
        # 2-1. KTAS 조건부 확률 생성
        logger.info("Creating KTAS conditional probability table...")
        start_time = time.time()
        
        if not extractor.create_ktas_probability_table():
            raise Exception("KTAS probability table creation failed")
            
        ktas_count = db_manager.get_table_count("nedis_meta.ktas_conditional_prob")
        logger.info(f"✓ Created {ktas_count:,} KTAS probability records in {time.time() - start_time:.2f}s")
        
        # 2-2. 진단 조건부 확률 생성
        logger.info("Creating diagnosis conditional probability table...")
        start_time = time.time()
        
        if not extractor.create_diagnosis_probability_table():
            logger.warning("Diagnosis probability table creation failed - continuing without it")
            diagnosis_count = 0
        else:
            diagnosis_count = db_manager.get_table_count("nedis_meta.diagnosis_conditional_prob")
            logger.info(f"✓ Created {diagnosis_count:,} diagnosis probability records in {time.time() - start_time:.2f}s")
        
        update_pipeline_progress(db_manager, "conditional_probabilities", "completed", 
                               int(ktas_count + diagnosis_count))
        
        return True
        
    except Exception as e:
        logger.error(f"Conditional probability extraction failed: {e}")
        update_pipeline_progress(db_manager, "conditional_probabilities", "failed", 0, str(e))
        return False


def save_results(report: Dict[str, Any], summary: Dict[str, Any], log_file: Path):
    """결과 저장"""
    logger = logging.getLogger(__name__)
    
    try:
        # outputs 디렉토리 생성
        outputs_dir = Path("outputs")
        outputs_dir.mkdir(exist_ok=True)
        
        # 타임스탬프
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 기본 프로파일 리포트 저장
        profile_report_path = outputs_dir / f"phase1_profile_report_{timestamp}.json"
        with open(profile_report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
        # 2. 실행 요약 저장
        summary_path = outputs_dir / f"phase1_summary_{timestamp}.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
        logger.info(f"Results saved:")
        logger.info(f"  - Profile report: {profile_report_path}")
        logger.info(f"  - Execution summary: {summary_path}")
        logger.info(f"  - Detailed log: {log_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
        return False


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("NEDIS Synthetic Data Generation - Phase 1")
    print("Data Profiling & Metadata Extraction")
    print("=" * 60)
    
    # 로깅 설정
    log_file = setup_logging()
    logger = logging.getLogger(__name__)
    
    start_time = time.time()
    
    try:
        # 초기화
        logger.info("Initializing database and configuration...")
        config = ConfigManager()
        db_manager = DatabaseManager(config.get('database.db_path'))
        
        logger.info(f"Database: {config.get('database.db_path', 'nedis_synthetic.duckdb')}")
        logger.info(f"Config: {config}")
        
        # 전제 조건 확인
        if not check_prerequisites(db_manager):
            logger.error("Prerequisites check failed")
            return False
            
        # 실행 요약 준비
        execution_summary = {
            'phase': 1,
            'start_time': datetime.now().isoformat(),
            'config': config.get_summary(),
            'steps': {}
        }
        
        # Step 1: 인구학적 프로파일링
        success, profile_report = run_profiling_step(db_manager, config)
        execution_summary['steps']['profiling'] = {
            'success': success,
            'duration': time.time() - start_time if success else 0
        }
        
        if not success:
            logger.error("Phase 1 failed at profiling step")
            return False
        
        # Step 2: 조건부 확률 추출
        step_start_time = time.time()
        success = run_probability_extraction_step(db_manager, config)
        execution_summary['steps']['probability_extraction'] = {
            'success': success,
            'duration': time.time() - step_start_time if success else 0
        }
        
        if not success:
            logger.error("Phase 1 failed at probability extraction step")
            return False
        
        # 전체 실행 완료
        total_duration = time.time() - start_time
        execution_summary['end_time'] = datetime.now().isoformat()
        execution_summary['total_duration'] = total_duration
        execution_summary['success'] = True
        
        # 결과 저장
        save_results(profile_report, execution_summary, log_file)
        
        # 성공 메시지
        logger.info("=" * 60)
        logger.info("✓ Phase 1 completed successfully!")
        logger.info(f"Total execution time: {total_duration:.2f} seconds")
        logger.info("=" * 60)
        
        # 다음 단계 안내
        logger.info("Next steps:")
        logger.info("1. Review the generated profile report")
        logger.info("2. Verify the meta tables in the database")
        logger.info("3. Run Phase 2: Population & Temporal Pattern Generation")
        
        return True
        
    except Exception as e:
        logger.error(f"Phase 1 execution failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # 실패 요약 저장
        execution_summary = {
            'phase': 1,
            'success': False,
            'error': str(e),
            'end_time': datetime.now().isoformat()
        }
        
        try:
            outputs_dir = Path("outputs")
            outputs_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            error_path = outputs_dir / f"phase1_error_{timestamp}.json"
            
            with open(error_path, 'w', encoding='utf-8') as f:
                json.dump(execution_summary, f, indent=2, ensure_ascii=False, default=str)
                
        except:
            pass
            
        return False
        
    finally:
        # 데이터베이스 연결 정리
        try:
            if 'db_manager' in locals():
                db_manager.close()
        except:
            pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)