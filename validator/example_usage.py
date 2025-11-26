#!/usr/bin/env python3
"""
NEDIS Synthetic Data Validation Suite - Example Usage

This script demonstrates how to use the new validator system for comprehensive
synthetic data validation and analysis.
"""

import asyncio
import json
from pathlib import Path
from datetime import datetime

# Import the new validator system
from validator import (
    validate,
    validate_async,
    ValidationOrchestrator,
    ValidationConfig,
    get_logger,
    get_performance_tracker
)
from validator.utils.logging import setup_logging


async def example_comprehensive_validation():
    """Example of comprehensive validation"""
    print("🚀 Example 1: Comprehensive Validation")

    logger = get_logger(__name__)

    try:
        # Basic validation using global function
        result = await validate_async(
            original_db="nedis_data.duckdb",
            synthetic_db="nedis_synth_2017.duckdb",
            validation_type="comprehensive",
            sample_size=50000
        )

        print("✅ Validation completed!")
        print(f"   Overall Score: {result.overall_score:.2f}/100")
        print(f"   Duration: {result.duration:.2f} seconds")
        print(f"   Validation Type: {result.validation_type}")

        # Show detailed results
        for category, category_result in result.results.items():
            if isinstance(category_result, dict) and 'overall_score' in category_result:
                print(f"   {category.title()} Score: {category_result['overall_score']".2f"}")

        return result

    except Exception as e:
        logger.error(f"Comprehensive validation failed: {e}")
        return None


def example_statistical_validation():
    """Example of statistical validation only"""
    print("\n📊 Example 2: Statistical Validation")

    logger = get_logger(__name__)

    try:
        # Statistical validation using orchestrator
        orchestrator = ValidationOrchestrator()
        result = orchestrator.validate_statistical(
            original_db="../nedis_data.duckdb",
            synthetic_db="../nedis_synth_2017.duckdb",
            sample_size=30000
        )

        print("✅ Statistical validation completed!"        print(f"   Score: {result.overall_score".2f"}/100")
        print(f"   Duration: {result.duration".2f"} seconds")

        # Show detailed statistical results
        stats_result = result.results.get('statistical', {})
        if stats_result:
            continuous_tests = stats_result.get('continuous_tests', {})
            categorical_tests = stats_result.get('categorical_tests', {})

            print(f"   Continuous variables tested: {len(continuous_tests)}")
            print(f"   Categorical variables tested: {len(categorical_tests)}")

            # Show some specific results
            for var_name, var_result in list(continuous_tests.items())[:3]:
                if var_result.get('available', False):
                    score = var_result.get('score', 0.0)
                    print(f"   - {var_name}: {score".2f"}")

        return result

    except Exception as e:
        logger.error(f"Statistical validation failed: {e}")
        return None


def example_pattern_analysis():
    """Example of pattern analysis validation"""
    print("\n🔍 Example 3: Pattern Analysis")

    logger = get_logger(__name__)

    try:
        # Pattern analysis validation
        orchestrator = ValidationOrchestrator()
        result = orchestrator.validate_patterns(
            original_db="../nedis_data.duckdb",
            synthetic_db="../nedis_synth_2017.duckdb"
        )

        print("✅ Pattern analysis completed!"        print(f"   Score: {result.overall_score".2f"}/100")
        print(f"   Duration: {result.duration".2f"} seconds")

        # Show pattern analysis details
        pattern_result = result.results.get('patterns', {})
        if pattern_result:
            demographic = pattern_result.get('demographic_patterns', {})
            clinical = pattern_result.get('clinical_patterns', {})
            temporal = pattern_result.get('temporal_patterns', {})

            print(f"   Demographic patterns: {len(demographic)} categories")
            print(f"   Clinical patterns: {len(clinical)} categories")
            print(f"   Temporal patterns: {len(temporal)} categories")

        return result

    except Exception as e:
        logger.error(f"Pattern analysis failed: {e}")
        return None


def example_custom_configuration():
    """Example of using custom configuration"""
    print("\n⚙️ Example 4: Custom Configuration")

    try:
        # Load custom configuration
        config = ValidationConfig.from_yaml("validation_config.yaml")

        # Modify configuration
        config.sample_size = 25000
        config.significance_level = 0.01

        # Create orchestrator with custom config
        orchestrator = ValidationOrchestrator(config)

        print("✅ Custom configuration loaded"        print(f"   Sample size: {config.sample_size}")
        print(f"   Significance level: {config.significance_level}")
        print(f"   Statistical threshold: {config.get_statistical_config('ks_threshold')}")

        return orchestrator

    except Exception as e:
        print(f"❌ Configuration example failed: {e}")
        return None


def example_performance_monitoring():
    """Example of performance monitoring"""
    print("\n📈 Example 5: Performance Monitoring")

    try:
        performance_tracker = get_performance_tracker()

        # Get current performance stats
        stats = performance_tracker.get_stats()

        print("📊 Current Performance Statistics:"        print(f"   Total validations: {stats.get('total_validations', 0)}")
        print(f"   Average score: {stats.get('average_score', 0)".2f"}")
        print(f"   Query count: {stats.get('query_count', 0)}")
        print(f"   Average query duration: {stats.get('query_avg_duration', 0)".3f"}s")
        print(f"   Memory usage: {stats.get('memory_avg_mb', 0)".1f"} MB")

        # Get slow queries
        slow_queries = performance_tracker.get_slow_queries(threshold=1.0, limit=3)
        if slow_queries:
            print("🐌 Slow queries:")
            for i, query in enumerate(slow_queries, 1):
                print(f"   {i}. {query['query']} - {query['duration']".3f"}s")

        return stats

    except Exception as e:
        print(f"❌ Performance monitoring failed: {e}")
        return None


def example_validation_history():
    """Example of accessing validation history"""
    print("\n📋 Example 6: Validation History")

    try:
        orchestrator = ValidationOrchestrator()
        history = orchestrator.get_validation_history(limit=5)

        if history:
            print("📋 Recent validations:")
            for i, result in enumerate(history, 1):
                print(f"   {i}. {result.validation_type} - Score: {result.overall_score:.2f} - {result.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        else:
            print("   No validation history available")

        # Get overall statistics
        stats = orchestrator.get_validation_stats()
        print("
📊 Overall statistics:"
        print(f"   Total validations: {stats['total_validations']}")
        print(f"   Average score: {stats['average_score']:.2f}")
        print(f"   Validation types: {stats['validation_types']}")

        return history

    except Exception as e:
        print(f"❌ Validation history failed: {e}")
        return None


async def main():
    """Main example function"""
    print("🎯 NEDIS Synthetic Data Validation Suite - Examples")
    print("=" * 60)

    # Setup logging
    setup_logging(log_level='INFO', log_to_file=False)

    # Run examples
    results = {}

    # Example 1: Comprehensive validation
    results['comprehensive'] = await example_comprehensive_validation()

    # Example 2: Statistical validation
    results['statistical'] = example_statistical_validation()

    # Example 3: Pattern analysis
    results['patterns'] = example_pattern_analysis()

    # Example 4: Custom configuration
    results['config'] = example_custom_configuration()

    # Example 5: Performance monitoring
    results['performance'] = example_performance_monitoring()

    # Example 6: Validation history
    results['history'] = example_validation_history()

    print("\n" + "=" * 60)
    print("🎉 All examples completed!")

    # Save results summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'examples_run': len(results),
        'results_summary': {}
    }

    for example_name, result in results.items():
        if result and hasattr(result, 'overall_score'):
            summary['results_summary'][example_name] = {
                'score': result.overall_score,
                'duration': result.duration
            }

    # Save summary to file
    summary_file = Path("validation_examples_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"📄 Summary saved to: {summary_file}")

    return results


if __name__ == "__main__":
    # Run examples
    results = asyncio.run(main())

    print("\n💡 Tips for using the validator:")
    print("   • Use validate_async() for async validation")
    print("   • Use ValidationOrchestrator for advanced control")
    print("   • Configure with ValidationConfig for custom settings")
    print("   • Monitor performance with get_performance_tracker()")
    print("   • Check logs for detailed information")
