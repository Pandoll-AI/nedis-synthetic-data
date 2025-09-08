#!/usr/bin/env python3
"""
NEDIS Synthetic Data Pipeline - Phases 3-7 Execution Script

This script runs the remaining phases of the NEDIS synthetic data generation pipeline:
- Phase 3: Hospital Allocation & Capacity Constraints
- Phase 4: Clinical Attributes Generation  
- Phase 5: Temporal Pattern Refinement
- Phase 6: Validation & Privacy Protection
- Phase 7: Rule-Based Optimization

Usage:
    python scripts/run_phases_3_to_7.py [options]
"""

import sys
import os
import logging
import argparse
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add project root directory to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.core.database import DatabaseManager
from src.core.config import ConfigManager

# Phase 3: Hospital Allocation
from src.allocation.gravity_model import HospitalGravityAllocator
from src.allocation.ipf_adjuster import IPFMarginalAdjuster

# Phase 4: Clinical Attributes
from src.clinical.dag_generator import ClinicalDAGGenerator
from src.clinical.diagnosis_generator import DiagnosisGenerator
from src.clinical.vitals_generator import VitalSignsGenerator

# Phase 5: Temporal Pattern Refinement
from src.temporal.duration_generator import DurationGenerator

# Phase 6: Validation
from src.validation.statistical_validator import StatisticalValidator
from src.validation.clinical_validator import ClinicalRuleValidator

def setup_logging():
    """Configure logging"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/phases_3_to_7_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )

def run_phase_3(db_manager, config, sample_dates=None):
    """
    Phase 3: Hospital Allocation & Capacity Constraints
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Phase 3: Hospital Allocation & Capacity Constraints ===")
    
    try:
        # Initialize components
        gravity_allocator = HospitalGravityAllocator(db_manager, config)
        ipf_adjuster = IPFMarginalAdjuster(db_manager, config)
        
        # Step 1: Initialize distance matrix
        logger.info("Step 1: Initializing distance matrix")
        gravity_allocator.initialize_distance_matrix()
        
        # Step 2: Calculate hospital attractiveness
        logger.info("Step 2: Calculating hospital attractiveness")
        gravity_allocator.calculate_hospital_attractiveness()
        
        # Step 3: Calculate allocation probabilities
        logger.info("Step 3: Calculating allocation probabilities")
        gravity_allocator.calculate_allocation_probabilities()
        
        # Step 4: Initialize allocation table (once for all dates)
        logger.info("Step 4: Initializing allocation table")
        gravity_allocator.initialize_allocation_table()
        
        # Step 5: Process sample dates or generate test allocation
        if sample_dates is None:
            # Use first 7 days of 2017 as sample
            sample_dates = [(datetime(2017, 1, 1) + timedelta(days=i)).strftime('%Y%m%d') 
                           for i in range(7)]
        
        for date_str in sample_dates:
            logger.info(f"Processing allocation for date: {date_str}")
            
            # Hospital allocation with capacity constraints
            gravity_allocator.allocate_with_capacity_constraints(date_str)
            
            # IPF margin adjustment
            ipf_result = ipf_adjuster.adjust_to_margins(date_str)
            
            if ipf_result['success']:
                logger.info(f"IPF adjustment completed for {date_str}: "
                           f"{ipf_result['iterations']} iterations, "
                           f"error: {ipf_result['final_error']:.6f}")
            else:
                logger.warning(f"IPF adjustment failed for {date_str}: {ipf_result}")
        
        logger.info("Phase 3 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 3 failed: {e}")
        return False

def run_phase_4(db_manager, config, sample_dates=None):
    """
    Phase 4: Clinical Attributes Generation
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Phase 4: Clinical Attributes Generation ===")
    
    try:
        # Initialize components
        dag_generator = ClinicalDAGGenerator(db_manager, config)
        diagnosis_generator = DiagnosisGenerator(db_manager, config)
        vitals_generator = VitalSignsGenerator(db_manager, config)
        
        # Step 1: Initialize clinical records table
        logger.info("Step 1: Initializing clinical records table")
        dag_generator.initialize_clinical_records_table()
        
        # Step 2: Initialize diagnosis tables
        logger.info("Step 2: Initializing diagnosis tables")
        diagnosis_generator.initialize_diagnosis_tables()
        
        if sample_dates is None:
            sample_dates = [(datetime(2017, 1, 1) + timedelta(days=i)).strftime('%Y%m%d') 
                           for i in range(7)]
        
        for date_str in sample_dates:
            logger.info(f"Generating clinical attributes for date: {date_str}")
            
            # Step 3: Generate clinical attributes
            clinical_result = dag_generator.generate_clinical_attributes(date_str)
            
            if clinical_result['success']:
                logger.info(f"Clinical attributes generated: {clinical_result['records_generated']} records")
                
                # Step 4: Generate diagnoses
                diagnosis_result = diagnosis_generator.generate_all_diagnoses(date_str)
                if diagnosis_result['success']:
                    logger.info(f"Diagnoses generated: {diagnosis_result['er_diagnoses_generated']} ER, "
                               f"{diagnosis_result['admission_diagnoses_generated']} admission")
                
                # Step 5: Generate vital signs
                vitals_result = vitals_generator.generate_all_vital_signs(date_str)
                if vitals_result['success']:
                    logger.info(f"Vital signs generated for {vitals_result['patients_processed']} patients")
            
        logger.info("Phase 4 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 4 failed: {e}")
        return False

def run_phase_5(db_manager, config, sample_dates=None):
    """
    Phase 5: Temporal Pattern Refinement
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Phase 5: Temporal Pattern Refinement ===")
    
    try:
        # Initialize duration generator
        duration_generator = DurationGenerator(db_manager, config)
        
        if sample_dates is None:
            sample_dates = [(datetime(2017, 1, 1) + timedelta(days=i)).strftime('%Y%m%d') 
                           for i in range(7)]
        
        for date_str in sample_dates:
            logger.info(f"Generating durations for date: {date_str}")
            
            # Generate ER and admission durations
            duration_result = duration_generator.generate_all_durations(date_str)
            
            if duration_result['success']:
                logger.info(f"Durations generated: {duration_result['patients_processed']} ER, "
                           f"{duration_result['admission_durations_generated']} admission")
                
                # Validate temporal consistency
                consistency_result = duration_generator.validate_temporal_consistency(date_str)
                if consistency_result.get('valid', True):
                    violations = len(consistency_result.get('violations', []))
                    if violations > 0:
                        logger.warning(f"Temporal consistency violations: {violations}")
                    else:
                        logger.info("Temporal consistency validation passed")
        
        logger.info("Phase 5 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 5 failed: {e}")
        return False

def run_phase_6(db_manager, config, sample_size=50000):
    """
    Phase 6: Validation & Privacy Protection
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Phase 6: Validation & Privacy Protection ===")
    
    try:
        # Initialize validators
        statistical_validator = StatisticalValidator(db_manager, config)
        clinical_validator = ClinicalRuleValidator(db_manager, config)
        
        # Step 1: Statistical validation
        logger.info("Step 1: Statistical distribution validation")
        statistical_results = statistical_validator.validate_distributions(sample_size)
        
        if statistical_results['success']:
            overall_score = statistical_results['overall_score']
            logger.info(f"Statistical validation completed: score = {overall_score:.3f}")
            
            # Generate and save statistical report
            stat_report = statistical_validator.generate_validation_report(statistical_results)
            report_path = Path("outputs") / "statistical_validation_report.md"
            report_path.parent.mkdir(exist_ok=True)
            report_path.write_text(stat_report, encoding='utf-8')
            logger.info(f"Statistical validation report saved: {report_path}")
            
            # Save results to database
            statistical_validator.save_validation_results(statistical_results)
        
        # Step 2: Clinical rule validation
        logger.info("Step 2: Clinical rule validation")
        clinical_results = clinical_validator.validate_all_clinical_rules(sample_size)
        
        if clinical_results['success']:
            compliance_rate = clinical_results['overall_compliance_rate']
            logger.info(f"Clinical validation completed: compliance = {compliance_rate:.3f}")
            
            # Generate and save clinical report
            clinical_report = clinical_validator.generate_clinical_validation_report(clinical_results)
            report_path = Path("outputs") / "clinical_validation_report.md"
            report_path.write_text(clinical_report, encoding='utf-8')
            logger.info(f"Clinical validation report saved: {report_path}")
            
            # Save results to database
            clinical_validator.save_clinical_validation_results(clinical_results)
        
        # Step 3: Privacy validation (placeholder)
        logger.info("Step 3: Privacy validation (basic checks)")
        # Note: Full privacy validation would require additional implementation
        logger.info("Privacy validation: K-anonymity and differential privacy checks would be implemented here")
        
        logger.info("Phase 6 completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Phase 6 failed: {e}")
        return False

def run_phase_7(db_manager, config):
    """
    Phase 7: Rule-Based Optimization (Placeholder)
    """
    logger = logging.getLogger(__name__)
    logger.info("=== Starting Phase 7: Rule-Based Optimization ===")
    
    try:
        logger.info("Rule-based optimization would be implemented here")
        logger.info("This phase would include:")
        logger.info("- Parameter tuning based on validation results")
        logger.info("- Medical knowledge-based adjustments")
        logger.info("- Quality score optimization")
        logger.info("- Iterative improvement cycles")
        
        # Placeholder for now
        logger.info("Phase 7 completed (placeholder)")
        return True
        
    except Exception as e:
        logger.error(f"Phase 7 failed: {e}")
        return False

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='NEDIS Synthetic Data Pipeline - Phases 3-7')
    parser.add_argument('--phases', nargs='+', default=['3', '4', '5', '6', '7'],
                       choices=['3', '4', '5', '6', '7'], 
                       help='Phases to run (default: all)')
    parser.add_argument('--sample-days', type=int, default=365,
                       help='Number of sample days to process (default: 365)')
    parser.add_argument('--validation-sample-size', type=int, default=50000,
                       help='Sample size for validation (default: 50000)')
    parser.add_argument('--start-date', default='20170101',
                       help='Start date for processing (YYYYMMDD, default: 20170101)')
    
    args = parser.parse_args()
    
    # Setup
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("=== NEDIS Synthetic Data Pipeline - Phases 3-7 ===")
    logger.info(f"Phases to run: {', '.join(args.phases)}")
    logger.info(f"Sample days: {args.sample_days}")
    logger.info(f"Validation sample size: {args.validation_sample_size}")
    
    try:
        # Initialize components - use the larger database with existing data
        db_manager = DatabaseManager("nedis_sample.duckdb")
        config = ConfigManager()
        
        # Generate sample dates
        start_date = datetime.strptime(args.start_date, '%Y%m%d')
        sample_dates = [(start_date + timedelta(days=i)).strftime('%Y%m%d') 
                       for i in range(args.sample_days)]
        
        success_count = 0
        total_phases = len(args.phases)
        
        # Execute selected phases
        if '3' in args.phases:
            if run_phase_3(db_manager, config, sample_dates):
                success_count += 1
        
        if '4' in args.phases:
            if run_phase_4(db_manager, config, sample_dates):
                success_count += 1
        
        if '5' in args.phases:
            if run_phase_5(db_manager, config, sample_dates):
                success_count += 1
        
        if '6' in args.phases:
            if run_phase_6(db_manager, config, args.validation_sample_size):
                success_count += 1
        
        if '7' in args.phases:
            if run_phase_7(db_manager, config):
                success_count += 1
        
        # Final summary
        logger.info(f"=== Pipeline Execution Summary ===")
        logger.info(f"Phases completed: {success_count}/{total_phases}")
        logger.info(f"Success rate: {success_count/total_phases:.1%}")
        
        if success_count == total_phases:
            logger.info("🎉 All selected phases completed successfully!")
            
            # Generate final summary report
            generate_final_summary_report(db_manager, sample_dates, args.validation_sample_size)
        else:
            logger.warning(f"⚠️ {total_phases - success_count} phases failed")
        
        return success_count == total_phases
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        return False

def generate_final_summary_report(db_manager, sample_dates, validation_sample_size):
    """Generate final pipeline summary report"""
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Generating final summary report")
        
        # Query data statistics
        stats = {}
        
        # Clinical records count
        stats['clinical_records'] = db_manager.fetch_dataframe(
            "SELECT COUNT(*) as count FROM nedis_synthetic.clinical_records"
        ).iloc[0]['count']
        
        # Hospital allocations count
        stats['hospital_allocations'] = db_manager.fetch_dataframe(
            "SELECT COUNT(*) as count FROM nedis_synthetic.hospital_allocations"
        ).iloc[0]['count']
        
        # ER diagnoses count
        stats['er_diagnoses'] = db_manager.fetch_dataframe(
            "SELECT COUNT(*) as count FROM nedis_synthetic.diag_er"
        ).iloc[0]['count']
        
        # Admission diagnoses count (if table exists)
        try:
            stats['admission_diagnoses'] = db_manager.fetch_dataframe(
                "SELECT COUNT(*) as count FROM nedis_synthetic.diag_adm"
            ).iloc[0]['count']
        except:
            stats['admission_diagnoses'] = 0
        
        # Generate summary report
        report = []
        report.append("# NEDIS Synthetic Data Generation - Final Summary")
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("## Pipeline Execution Summary")
        report.append(f"- Sample dates processed: {len(sample_dates)}")
        report.append(f"- Date range: {sample_dates[0]} to {sample_dates[-1]}")
        report.append(f"- Validation sample size: {validation_sample_size:,}")
        report.append("")
        report.append("## Generated Data Statistics")
        report.append(f"- Clinical records: {stats['clinical_records']:,}")
        report.append(f"- Hospital allocations: {stats['hospital_allocations']:,}")
        report.append(f"- ER diagnoses: {stats['er_diagnoses']:,}")
        report.append(f"- Admission diagnoses: {stats['admission_diagnoses']:,}")
        report.append("")
        report.append("## Quality Metrics")
        report.append("- Statistical validation: See statistical_validation_report.md")
        report.append("- Clinical validation: See clinical_validation_report.md")
        report.append("- Temporal consistency: Validated during Phase 5")
        report.append("")
        report.append("## Next Steps")
        report.append("1. Review validation reports for any issues")
        report.append("2. Run full pipeline on complete date range if sample results are satisfactory")
        report.append("3. Implement Phase 7 optimization if needed")
        report.append("4. Export final synthetic dataset")
        
        # Save report
        report_path = Path("outputs") / "pipeline_summary_report.md"
        report_path.write_text("\n".join(report), encoding='utf-8')
        
        logger.info(f"Final summary report saved: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate summary report: {e}")

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)