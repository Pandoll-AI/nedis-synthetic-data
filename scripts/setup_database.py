#!/usr/bin/env python3
"""
NEDIS 합성 데이터 생성 시스템 데이터베이스 초기화 스크립트

기존 sample 데이터베이스를 proper 스키마 구조로 마이그레이션하고
Phase 1-2 실행을 위한 환경을 준비합니다.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import duckdb
import logging
from datetime import datetime
from core.database import DatabaseManager
from core.config import ConfigManager

def setup_logging():
    """로깅 설정"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/database_setup.log'),
            logging.StreamHandler()
        ]
    )

def migrate_sample_data(source_db: str, target_db: str):
    """샘플 데이터를 적절한 스키마로 마이그레이션"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Migrating data from {source_db} to {target_db}")
    
    # 소스와 타겟 연결
    source_conn = duckdb.connect(source_db)
    target_conn = duckdb.connect(target_db)
    
    try:
        # 1. 타겟 데이터베이스에 스키마 생성
        schema_sql = Path("sql/create_schemas.sql").read_text()
        target_conn.execute(schema_sql)
        logger.info("Created schemas in target database")
        
        # 2. nedis2017 데이터를 nedis_original 스키마로 복사
        logger.info("Copying nedis2017 main table...")
        
        # 소스에서 데이터 조회
        source_data = source_conn.execute("SELECT * FROM nedis2017").fetchdf()
        logger.info(f"Source nedis2017 records: {len(source_data):,}")
        
        # 타겟에 삽입 (DuckDB register 방식 사용)
        target_conn.register("source_data", source_data)
        target_conn.execute("INSERT INTO nedis_original.nedis2017 SELECT * FROM source_data")
        
        # 검증
        target_count = target_conn.execute("SELECT COUNT(*) FROM nedis_original.nedis2017").fetchone()[0]
        logger.info(f"Migrated nedis2017 records: {target_count:,}")
        
        # 3. 진단 테이블 복사
        for table in ['diag_er', 'diag_adm']:
            logger.info(f"Copying {table} table...")
            
            source_diag = source_conn.execute(f"SELECT * FROM {table}").fetchdf()
            logger.info(f"Source {table} records: {len(source_diag):,}")
            
            target_conn.register(f"source_{table}", source_diag)
            target_conn.execute(f"INSERT INTO nedis_original.{table} SELECT * FROM source_{table}")
            
            target_diag_count = target_conn.execute(f"SELECT COUNT(*) FROM nedis_original.{table}").fetchone()[0]
            logger.info(f"Migrated {table} records: {target_diag_count:,}")
        
        logger.info("✓ Data migration completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"✗ Data migration failed: {e}")
        return False
    finally:
        source_conn.close()
        target_conn.close()

def validate_migration(db_path: str) -> bool:
    """마이그레이션 결과 검증"""
    logger = logging.getLogger(__name__)
    
    try:
        db = DatabaseManager(db_path)
        
        # 원본 데이터 테이블 확인
        if not db.table_exists("nedis_original.nedis2017"):
            logger.error("nedis_original.nedis2017 table not found")
            return False
            
        # 레코드 수 확인
        main_count = db.get_table_count("nedis_original.nedis2017")
        diag_er_count = db.get_table_count("nedis_original.diag_er")
        diag_adm_count = db.get_table_count("nedis_original.diag_adm")
        
        logger.info(f"Validation results:")
        logger.info(f"  nedis2017: {main_count:,} records")
        logger.info(f"  diag_er: {diag_er_count:,} records") 
        logger.info(f"  diag_adm: {diag_adm_count:,} records")
        
        # 기본 품질 체크
        sample_query = """
        SELECT 
            COUNT(*) as total,
            COUNT(DISTINCT pat_do_cd) as unique_regions,
            COUNT(DISTINCT emorg_cd) as unique_hospitals,
            COUNT(DISTINCT pat_age_gr) as unique_age_groups
        FROM nedis_original.nedis2017
        """
        
        quality_check = db.fetch_dataframe(sample_query).iloc[0]
        logger.info(f"Quality check:")
        logger.info(f"  Unique regions: {quality_check['unique_regions']}")
        logger.info(f"  Unique hospitals: {quality_check['unique_hospitals']}")
        logger.info(f"  Unique age groups: {quality_check['unique_age_groups']}")
        
        if main_count > 0 and quality_check['unique_regions'] > 5:
            logger.info("✓ Migration validation passed")
            return True
        else:
            logger.error("✗ Migration validation failed")
            return False
            
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def initialize_progress_tracking(db: DatabaseManager):
    """파이프라인 진행 상황 추적 초기화"""
    logger = logging.getLogger(__name__)
    
    phases = [
        'phase1_profiling',
        'phase1_population_margins', 
        'phase1_hospital_stats',
        'phase2_yearly_volumes',
        'phase2_daily_volumes',
        'phase3_hospital_allocation',
        'phase4_clinical_attributes',
        'phase5_temporal_patterns',
        'phase6_validation',
        'phase7_optimization'
    ]
    
    # 기존 진행 상황 초기화
    db.execute_query("DELETE FROM nedis_meta.pipeline_progress")
    
    # 새 단계들 초기화
    for phase in phases:
        db.execute_query("""
            INSERT INTO nedis_meta.pipeline_progress 
            (step_name, status, records_processed, start_time)
            VALUES (?, 'pending', 0, ?)
        """, [phase, datetime.now()])
    
    logger.info(f"Initialized progress tracking for {len(phases)} phases")

def main():
    """메인 실행 함수"""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== NEDIS Database Setup Started ===")
    
    # 설정 로드
    config = ConfigManager()
    
    source_db = "nedis_sample.duckdb"
    target_db = config.get('database.db_path', 'nedis_synthetic.duckdb')
    
    # 1. 데이터 마이그레이션
    if not migrate_sample_data(source_db, target_db):
        logger.error("Database migration failed")
        return False
        
    # 2. 마이그레이션 검증
    if not validate_migration(target_db):
        logger.error("Migration validation failed")
        return False
    
    # 3. 추가 설정
    db = DatabaseManager(target_db)
    
    # 진행 상황 추적 초기화
    initialize_progress_tracking(db)
    
    # 데이터베이스 최적화
    db.vacuum_analyze()
    
    logger.info("=== Database Setup Completed Successfully ===")
    logger.info(f"Ready to run Phase 1 with database: {target_db}")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)