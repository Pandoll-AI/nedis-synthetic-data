#!/usr/bin/env python3
"""
NEDIS 합성 데이터 생성 Phase 2: 인구 및 시간 패턴 생성

Phase 2 구현 내용:
1. Dirichlet-Multinomial 인구 볼륨 생성 (PopulationVolumeGenerator)
2. NHPP 비균질 포아송 과정 일별 분해 (NHPPTemporalGenerator)
3. 계절별, 요일별, 공휴일 패턴 반영
4. 생성 결과 검증 및 품질 확인
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import logging
import json
import time
from datetime import datetime
from typing import Dict, Any

from core.database import DatabaseManager
from core.config import ConfigManager
from population.generator import PopulationVolumeGenerator
from temporal.nhpp_generator import NHPPTemporalGenerator


def setup_logging():
    """로깅 설정"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"phase2_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def check_prerequisites(db_manager: DatabaseManager) -> bool:
    """Phase 2 전제 조건 확인"""
    logger = logging.getLogger(__name__)
    
    try:
        # Phase 1 완료 확인
        required_tables = [
            "nedis_meta.population_margins",
            "nedis_meta.hospital_capacity", 
            "nedis_meta.ktas_conditional_prob"
        ]
        
        for table in required_tables:
            if not db_manager.table_exists(table):
                logger.error(f"Required table '{table}' not found")
                logger.error("Please run Phase 1 first")
                return False
                
            count = db_manager.get_table_count(table)
            if count == 0:
                logger.error(f"Table '{table}' is empty")
                return False
                
            logger.info(f"Found {count:,} records in {table}")
        
        # population_margins 데이터 품질 확인
        margins_quality_query = """
        SELECT 
            COUNT(*) as total_combinations,
            COUNT(CASE WHEN yearly_visits > 0 THEN 1 END) as valid_combinations,
            AVG(yearly_visits) as avg_visits,
            MIN(yearly_visits) as min_visits,
            MAX(yearly_visits) as max_visits
        FROM nedis_meta.population_margins
        """
        
        quality_stats = db_manager.fetch_dataframe(margins_quality_query).iloc[0]
        
        if quality_stats['valid_combinations'] < quality_stats['total_combinations'] * 0.9:
            logger.warning(f"Only {quality_stats['valid_combinations']} / {quality_stats['total_combinations']} combinations have valid visit counts")
        
        logger.info(f"Population margins quality: avg_visits={quality_stats['avg_visits']:.1f}, range={quality_stats['min_visits']}-{quality_stats['max_visits']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Prerequisites check failed: {e}")
        return False


def update_progress(db_manager: DatabaseManager, step_name: str, status: str, 
                   records_processed: int = 0, error_message: str = None):
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
        logging.getLogger(__name__).warning(f"Failed to update progress for {step_name}: {e}")


def run_population_volume_generation(db_manager: DatabaseManager, config: ConfigManager) -> bool:
    """Step 1: 인구 볼륨 생성"""
    logger = logging.getLogger(__name__)
    logger.info("=== Step 1: Population Volume Generation ===")
    
    update_progress(db_manager, "phase2_yearly_volumes", "running")
    
    try:
        generator = PopulationVolumeGenerator(db_manager, config)
        target_records = config.get('population.target_total_records', 9_200_000)
        
        logger.info(f"Target total records: {target_records:,}")
        
        start_time = time.time()
        success = generator.generate_yearly_volumes(target_records)
        duration = time.time() - start_time
        
        if success:
            # 생성 결과 확인
            yearly_count = db_manager.get_table_count("nedis_synthetic.yearly_volumes")
            total_records_query = "SELECT SUM(synthetic_yearly_count) as total FROM nedis_synthetic.yearly_volumes"
            total_records = db_manager.fetch_dataframe(total_records_query)['total'][0]
            
            update_progress(db_manager, "phase2_yearly_volumes", "completed", int(yearly_count))
            
            logger.info(f"✓ Population volume generation completed:")
            logger.info(f"    Combinations: {yearly_count:,}")
            logger.info(f"    Total records: {total_records:,}")
            logger.info(f"    Duration: {duration:.2f}s")
            
            # 생성 요약 및 검증
            summary = generator.get_generation_summary()
            validation = generator.validate_generated_volumes()
            
            if not validation['passed']:
                logger.warning(f"Validation issues found: {validation['issues']}")
            
            return True
        else:
            update_progress(db_manager, "phase2_yearly_volumes", "failed", 0, "Generation failed")
            logger.error("✗ Population volume generation failed")
            return False
            
    except Exception as e:
        error_msg = str(e)
        update_progress(db_manager, "phase2_yearly_volumes", "failed", 0, error_msg)
        logger.error(f"✗ Population volume generation error: {error_msg}")
        return False


def run_temporal_pattern_generation(db_manager: DatabaseManager, config: ConfigManager) -> bool:
    """Step 2: 시간 패턴 생성"""
    logger = logging.getLogger(__name__)
    logger.info("=== Step 2: Temporal Pattern Generation ===")
    
    update_progress(db_manager, "phase2_daily_volumes", "running")
    
    try:
        generator = NHPPTemporalGenerator(db_manager, config)
        
        start_time = time.time()
        success = generator.generate_daily_events(2017)
        duration = time.time() - start_time
        
        if success:
            # 생성 결과 확인
            daily_count = db_manager.get_table_count("nedis_synthetic.daily_volumes")
            total_daily_records_query = "SELECT SUM(synthetic_daily_count) as total FROM nedis_synthetic.daily_volumes"
            total_daily_records = db_manager.fetch_dataframe(total_daily_records_query)['total'][0]
            
            update_progress(db_manager, "phase2_daily_volumes", "completed", int(daily_count))
            
            logger.info(f"✓ Temporal pattern generation completed:")
            logger.info(f"    Daily records: {daily_count:,}")
            logger.info(f"    Total visits: {total_daily_records:,}")
            logger.info(f"    Duration: {duration:.2f}s")
            
            # 시간 패턴 요약 및 검증
            summary = generator.get_temporal_summary()
            validation = generator.validate_temporal_patterns()
            
            if not validation['passed']:
                logger.warning(f"Temporal validation issues: {validation['issues']}")
            else:
                logger.info("✓ All temporal validation checks passed")
            
            # 패턴 분석 로그
            if 'weekday_patterns' in summary:
                logger.info("Weekday patterns generated:")
                for pattern in summary['weekday_patterns']:
                    logger.info(f"  {pattern['weekday']}: {pattern['avg_visits']:.1f} avg visits")
            
            return True
        else:
            update_progress(db_manager, "phase2_daily_volumes", "failed", 0, "Generation failed")
            logger.error("✗ Temporal pattern generation failed")
            return False
            
    except Exception as e:
        error_msg = str(e)
        update_progress(db_manager, "phase2_daily_volumes", "failed", 0, error_msg)
        logger.error(f"✗ Temporal pattern generation error: {error_msg}")
        return False


def generate_phase2_report(db_manager: DatabaseManager, config: ConfigManager, log_file: Path) -> bool:
    """Phase 2 완료 리포트 생성"""
    logger = logging.getLogger(__name__)
    logger.info("=== Step 3: Generating Phase 2 Report ===")
    
    try:
        # 전체 실행 요약
        report = {
            'phase': 2,
            'generation_timestamp': datetime.now().isoformat(),
            'config_summary': config.get_summary(),
        }
        
        # 인구 볼륨 생성 결과
        if db_manager.table_exists("nedis_synthetic.yearly_volumes"):
            pop_generator = PopulationVolumeGenerator(db_manager, config)
            report['population_volumes'] = pop_generator.get_generation_summary()
            report['population_validation'] = pop_generator.validate_generated_volumes()
        
        # 시간 패턴 생성 결과
        if db_manager.table_exists("nedis_synthetic.daily_volumes"):
            temporal_generator = NHPPTemporalGenerator(db_manager, config)
            report['temporal_patterns'] = temporal_generator.get_temporal_summary()
            report['temporal_validation'] = temporal_generator.validate_temporal_patterns()
        
        # 파이프라인 진행 상황
        progress_query = """
        SELECT step_name, status, records_processed, start_time, end_time
        FROM nedis_meta.pipeline_progress
        WHERE step_name LIKE 'phase2_%'
        ORDER BY step_name
        """
        progress_data = db_manager.fetch_dataframe(progress_query)
        report['pipeline_progress'] = progress_data.to_dict('records')
        
        # 전체 데이터 일관성 검증
        report['consistency_checks'] = perform_consistency_checks(db_manager)
        
        # 리포트 저장
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"phase2_complete_report_{timestamp}.json"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✓ Phase 2 report generated: {report_path}")
        
        # 요약 정보 로그
        log_report_summary(report, logger)
        
        return True
        
    except Exception as e:
        logger.error(f"✗ Phase 2 report generation failed: {e}")
        return False


def perform_consistency_checks(db_manager: DatabaseManager) -> Dict[str, Any]:
    """데이터 일관성 검증"""
    checks = {}
    
    try:
        # 1. 총 레코드 수 일치 확인
        yearly_total_query = "SELECT SUM(synthetic_yearly_count) as total FROM nedis_synthetic.yearly_volumes"
        daily_total_query = "SELECT SUM(synthetic_daily_count) as total FROM nedis_synthetic.daily_volumes"
        
        yearly_total = db_manager.fetch_dataframe(yearly_total_query)['total'][0] or 0
        daily_total = db_manager.fetch_dataframe(daily_total_query)['total'][0] or 0
        
        checks['totals'] = {
            'yearly_total': yearly_total,
            'daily_total': daily_total,
            'difference': abs(yearly_total - daily_total),
            'difference_pct': abs(yearly_total - daily_total) / yearly_total * 100 if yearly_total > 0 else 0
        }
        
        # 2. 조합 수 확인
        yearly_combinations = db_manager.get_table_count("nedis_synthetic.yearly_volumes")
        
        # 일별 데이터에서 고유 조합 수 (최대 365배)
        daily_unique_query = """
        SELECT COUNT(DISTINCT pat_do_cd || '|' || pat_age_gr || '|' || pat_sex) as unique_combinations
        FROM nedis_synthetic.daily_volumes
        """
        daily_unique_combinations = db_manager.fetch_dataframe(daily_unique_query)['unique_combinations'][0]
        
        checks['combinations'] = {
            'yearly_combinations': yearly_combinations,
            'daily_unique_combinations': daily_unique_combinations,
            'coverage_rate': daily_unique_combinations / yearly_combinations if yearly_combinations > 0 else 0
        }
        
        # 3. 날짜 범위 확인
        date_range_query = """
        SELECT 
            MIN(vst_dt) as min_date,
            MAX(vst_dt) as max_date,
            COUNT(DISTINCT vst_dt) as unique_dates
        FROM nedis_synthetic.daily_volumes
        """
        date_stats = db_manager.fetch_dataframe(date_range_query).iloc[0]
        checks['date_coverage'] = date_stats.to_dict()
        
        # 4. 데이터 품질 지표
        quality_query = """
        SELECT 
            COUNT(CASE WHEN synthetic_daily_count = 0 THEN 1 END) as zero_count_records,
            COUNT(CASE WHEN synthetic_daily_count < 0 THEN 1 END) as negative_count_records,
            COUNT(CASE WHEN lambda_value < 0 THEN 1 END) as negative_lambda_records,
            AVG(synthetic_daily_count) as avg_daily_count,
            STDDEV(synthetic_daily_count) as std_daily_count
        FROM nedis_synthetic.daily_volumes
        """
        quality_stats = db_manager.fetch_dataframe(quality_query).iloc[0]
        checks['data_quality'] = quality_stats.to_dict()
        
    except Exception as e:
        checks['error'] = str(e)
    
    return checks


def log_report_summary(report: Dict, logger):
    """리포트 요약 정보 로깅"""
    try:
        logger.info("\n" + "="*60)
        logger.info("PHASE 2 COMPLETION SUMMARY")
        logger.info("="*60)
        
        # 인구 볼륨 결과
        if 'population_volumes' in report and 'statistics' in report['population_volumes']:
            pop_stats = report['population_volumes']['statistics']
            logger.info(f"Population Volumes Generated:")
            logger.info(f"  Combinations: {pop_stats.get('total_combinations', 0):,}")
            logger.info(f"  Total Records: {pop_stats.get('total_records', 0):,}")
            logger.info(f"  Unique Regions: {pop_stats.get('unique_regions', 0)}")
            logger.info(f"  Average per Combination: {pop_stats.get('avg_records', 0):.1f}")
        
        # 시간 패턴 결과
        if 'temporal_patterns' in report and 'basic_statistics' in report['temporal_patterns']:
            temp_stats = report['temporal_patterns']['basic_statistics']
            logger.info(f"Temporal Patterns Generated:")
            logger.info(f"  Daily Records: {temp_stats.get('total_daily_records', 0):,}")
            logger.info(f"  Unique Dates: {temp_stats.get('unique_dates', 0)}")
            logger.info(f"  Total Visits: {temp_stats.get('total_visits', 0):,}")
            logger.info(f"  Average Daily Visits: {temp_stats.get('avg_daily_visits', 0):.1f}")
        
        # 일관성 검증 결과
        if 'consistency_checks' in report and 'totals' in report['consistency_checks']:
            consistency = report['consistency_checks']['totals']
            logger.info(f"Consistency Verification:")
            logger.info(f"  Yearly Total: {consistency.get('yearly_total', 0):,}")
            logger.info(f"  Daily Total: {consistency.get('daily_total', 0):,}")
            logger.info(f"  Difference: {consistency.get('difference_pct', 0):.3f}%")
        
        # 검증 결과
        validation_passed = True
        if 'population_validation' in report:
            pop_val = report['population_validation']
            if not pop_val.get('passed', False):
                logger.warning(f"Population validation issues: {len(pop_val.get('issues', []))}")
                validation_passed = False
        
        if 'temporal_validation' in report:
            temp_val = report['temporal_validation']
            if not temp_val.get('passed', False):
                logger.warning(f"Temporal validation issues: {len(temp_val.get('issues', []))}")
                validation_passed = False
        
        if validation_passed:
            logger.info("✓ All validation checks passed")
        
        logger.info("="*60)
        
    except Exception as e:
        logger.warning(f"Failed to log report summary: {e}")


def main():
    """메인 실행 함수"""
    print("=" * 60)
    print("NEDIS Synthetic Data Generation - Phase 2")
    print("Population & Temporal Pattern Generation")
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
        
        logger.info(f"Database: {config.get('database.db_path')}")
        logger.info(f"Target records: {config.get('population.target_total_records'):,}")
        
        # 전제 조건 확인
        if not check_prerequisites(db_manager):
            logger.error("Prerequisites check failed")
            return False
        
        # 실행 단계
        steps = [
            ("Population Volume Generation", run_population_volume_generation),
            ("Temporal Pattern Generation", run_temporal_pattern_generation),
        ]
        
        successful_steps = []
        
        for step_name, step_function in steps:
            logger.info(f"\nStarting: {step_name}")
            step_start = time.time()
            
            try:
                success = step_function(db_manager, config)
                step_duration = time.time() - step_start
                
                if success:
                    successful_steps.append(step_name)
                    logger.info(f"✓ {step_name} completed in {step_duration:.2f}s")
                else:
                    logger.error(f"✗ {step_name} failed after {step_duration:.2f}s")
                    break  # 중요한 단계 실패 시 중단
                    
            except Exception as e:
                step_duration = time.time() - step_start
                logger.error(f"✗ {step_name} error after {step_duration:.2f}s: {e}")
                break
        
        # 리포트 생성
        if len(successful_steps) == len(steps):
            logger.info(f"\nGenerating Phase 2 completion report...")
            generate_phase2_report(db_manager, config, log_file)
        
        total_duration = time.time() - start_time
        success_rate = len(successful_steps) / len(steps)
        
        # 최종 결과
        if success_rate == 1.0:
            logger.info(f"\n🎉 Phase 2 COMPLETED SUCCESSFULLY in {total_duration:.2f}s")
            logger.info(f"All {len(steps)} steps completed: {', '.join(successful_steps)}")
            
            # 데이터베이스 최적화
            db_manager.vacuum_analyze()
            logger.info("Database optimized")
            
            logger.info("\nNext steps:")
            logger.info("1. Review the generated population and temporal patterns")
            logger.info("2. Verify the synthetic data quality metrics")
            logger.info("3. Run Phase 3: Hospital Allocation & Capacity Constraints")
            
            return True
        else:
            logger.error(f"\n💥 Phase 2 PARTIALLY FAILED ({success_rate:.1%} success rate)")
            logger.error(f"Completed steps: {', '.join(successful_steps)}")
            return False
            
    except Exception as e:
        logger.error(f"Phase 2 execution error: {e}")
        return False
    finally:
        # 정리
        try:
            db_manager.close()
        except:
            pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)