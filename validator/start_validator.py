#!/usr/bin/env python3
"""
NEDIS Synthetic Data Validator - Startup Script

This script initializes and starts the validator system with proper configuration.
"""

import argparse
import sys
import asyncio
from pathlib import Path

from validator import setup_logging, ValidationOrchestrator, ValidationConfig, get_logger
from validator.core.config import ConfigManager


def main():
    """Main startup function"""
    parser = argparse.ArgumentParser(description='Start NEDIS Validator System')
    parser.add_argument('--config', '-c', help='Configuration file path')
    parser.add_argument('--log-level', '-l', default='INFO', help='Logging level')
    parser.add_argument('--demo', '-d', action='store_true', help='Run demo validation')

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging(log_level=args.log_level, log_to_file=True)
        logger = get_logger(__name__)

        logger.info("🚀 Starting NEDIS Synthetic Data Validator System")

        # Load configuration
        if args.config:
            config_manager = ConfigManager(args.config)
        else:
            config_manager = ConfigManager()

        try:
            config = config_manager.get_config()
            logger.info("✅ Configuration loaded successfully")
        except Exception as e:
            logger.error(f"❌ Configuration loading failed: {e}")
            # Create default configuration
            config = ValidationConfig()
            config.save("validation_config.yaml")
            logger.info("📄 Default configuration created: validation_config.yaml")

        # Initialize validator
        logger.info("🔧 Initializing validation orchestrator...")
        orchestrator = ValidationOrchestrator(config)
        logger.info("✅ Validation orchestrator ready")

        if args.demo:
            logger.info("🎯 Running demo validation...")
            asyncio.run(run_demo_validation(orchestrator))
        else:
            logger.info("🎉 Validator system is ready!")
            logger.info("💡 Use the CLI or API to start validations:")
            logger.info("   - CLI: python -m validator.cli validate --help")
            logger.info("   - API: python -m validator.api (not implemented yet)")
            logger.info("   - Examples: python validator/example_usage.py")

        return 0

    except Exception as e:
        print(f"❌ Failed to start validator system: {e}")
        return 1


async def run_demo_validation(orchestrator: ValidationOrchestrator):
    """Run a demo validation"""
    logger = get_logger(__name__)

    try:
        logger.info("🎯 Starting demo validation...")

        # Run comprehensive validation
        result = await orchestrator.validate_comprehensive(
            original_db="nedis_data.duckdb",
            synthetic_db="nedis_synth_2017.duckdb",
            sample_size=10000  # Smaller sample for demo
        )

        logger.info("✅ Demo validation completed!")
        logger.info(f"📊 Overall Score: {result.overall_score:.2f}/100")
        logger.info(f"⏱️  Duration: {result.duration:.2f} seconds")

        # Show category scores
        for category, category_result in result.results.items():
            if isinstance(category_result, dict) and 'overall_score' in category_result:
                logger.info(f"   {category.title()}: {category_result['overall_score']:.2f}")

        if result.errors:
            logger.warning(f"⚠️  Errors encountered: {len(result.errors)}")
            for error in result.errors[:3]:  # Show first 3 errors
                logger.warning(f"   - {error['error']}")

        if result.warnings:
            logger.info(f"ℹ️  Warnings: {len(result.warnings)}")

        logger.info("🎉 Demo completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo validation failed: {e}")


if __name__ == "__main__":
    sys.exit(main())
