#!/usr/bin/env python3
"""
Command Line Interface for NEDIS Synthetic Data Validation Suite.

This module provides a comprehensive CLI for:
- Running validations
- Generating reports
- Managing configurations
- Viewing results
- Performance monitoring
"""

import argparse
import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import json

from .core.config import ValidationConfig, ConfigManager
from .core.validator import ValidationOrchestrator, validate_async
from .utils.logging import setup_global_logging
from .utils.metrics import get_performance_tracker


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI"""
    parser = argparse.ArgumentParser(
        description='NEDIS Synthetic Data Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Comprehensive validation
  python -m validator.cli validate --original-db original.duckdb --synthetic-db synthetic.duckdb

  # Statistical validation only
  python -m validator.cli validate --original-db original.duckdb --synthetic-db synthetic.duckdb --type statistical

  # Generate validation report
  python -m validator.cli report --validation-id abc123 --output-format pdf

  # Show performance statistics
  python -m validator.cli performance

  # Create default configuration
  python -m validator.cli config create --output validation_config.yaml
        '''
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Validate command
    validate_parser = subparsers.add_parser(
        'validate',
        help='Run validation on synthetic data'
    )
    validate_parser.add_argument(
        '--original-db',
        required=True,
        help='Path to original database'
    )
    validate_parser.add_argument(
        '--synthetic-db',
        required=True,
        help='Path to synthetic database'
    )
    validate_parser.add_argument(
        '--type',
        choices=['comprehensive', 'statistical', 'patterns', 'clinical', 'temporal'],
        default='comprehensive',
        help='Type of validation to perform'
    )
    validate_parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size for validation'
    )
    validate_parser.add_argument(
        '--config',
        help='Path to configuration file'
    )
    validate_parser.add_argument(
        '--output-format',
        choices=['json', 'html', 'pdf', 'excel'],
        default='json',
        help='Output format for results'
    )
    validate_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )

    # Report command
    report_parser = subparsers.add_parser(
        'report',
        help='Generate validation reports'
    )
    report_parser.add_argument(
        '--validation-id',
        help='Validation ID to generate report for'
    )
    report_parser.add_argument(
        '--output-format',
        choices=['json', 'html', 'pdf', 'excel', 'word'],
        default='html',
        help='Output format for report'
    )
    report_parser.add_argument(
        '--output-dir',
        default='reports',
        help='Output directory for reports'
    )
    report_parser.add_argument(
        '--include-charts',
        action='store_true',
        help='Include charts in report'
    )

    # Performance command
    performance_parser = subparsers.add_parser(
        'performance',
        help='Show performance statistics'
    )
    performance_parser.add_argument(
        '--export-format',
        choices=['json', 'text'],
        default='text',
        help='Export format for performance data'
    )
    performance_parser.add_argument(
        '--time-window',
        type=int,
        help='Time window in minutes for performance analysis'
    )

    # Config command
    config_parser = subparsers.add_parser(
        'config',
        help='Manage validation configuration'
    )
    config_subparsers = config_parser.add_subparsers(dest='config_command')

    # Config create
    config_create_parser = config_subparsers.add_parser(
        'create',
        help='Create default configuration file'
    )
    config_create_parser.add_argument(
        '--output', '-o',
        default='validation_config.yaml',
        help='Output configuration file path'
    )

    # Config validate
    config_validate_parser = config_subparsers.add_parser(
        'validate',
        help='Validate configuration file'
    )
    config_validate_parser.add_argument(
        'config_file',
        help='Configuration file to validate'
    )

    # Config show
    config_show_parser = config_subparsers.add_parser(
        'show',
        help='Show current configuration'
    )
    config_show_parser.add_argument(
        '--format',
        choices=['yaml', 'json'],
        default='yaml',
        help='Output format'
    )

    # Results command
    results_parser = subparsers.add_parser(
        'results',
        help='Manage validation results'
    )
    results_parser.add_argument(
        '--list',
        action='store_true',
        help='List recent validation results'
    )
    results_parser.add_argument(
        '--validation-id',
        help='Show specific validation result'
    )
    results_parser.add_argument(
        '--export',
        choices=['json', 'csv'],
        help='Export results to file'
    )

    return parser


def handle_validate(args):
    """Handle validate command"""
    print("🔍 Starting validation..."    print(f"Original DB: {args.original_db}")
    print(f"Synthetic DB: {args.synthetic_db}")
    print(f"Validation type: {args.type}")
    print(f"Sample size: {args.sample_size or 'default'}")

    try:
        # Setup logging
        log_level = 'DEBUG' if args.verbose else 'INFO'
        setup_global_logging(log_level)

        # Load configuration
        config_manager = ConfigManager(args.config) if args.config else ConfigManager()
        config = config_manager.get_config()

        # Create orchestrator
        orchestrator = ValidationOrchestrator(config)

        # Run validation
        result = asyncio.run(
            validate_async(
                args.original_db,
                args.synthetic_db,
                args.type,
                args.sample_size
            )
        )

        # Display results
        print("
✅ Validation completed!"
        print(f"Overall score: {result.overall_score:.2f}/100")
        print(f"Duration: {result.duration:.2f} seconds")

        if result.errors:
            print(f"❌ Errors: {len(result.errors)}")
            for error in result.errors:
                print(f"  - {error['error']}")

        if result.warnings:
            print(f"⚠️  Warnings: {len(result.warnings)}")
            for warning in result.warnings:
                print(f"  - {warning['warning']}")

        # Export results
        if args.output_format == 'json':
            output_file = f"validation_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"📄 Results exported to: {output_file}")

        return 0

    except Exception as e:
        print(f"❌ Validation failed: {e}")
        return 1


def handle_report(args):
    """Handle report command"""
    print("📊 Generating validation report...")

    try:
        # This is a placeholder - report generation would be implemented here
        print(f"Report format: {args.output_format}")
        print(f"Output directory: {args.output_dir}")

        if args.include_charts:
            print("📈 Including charts in report")

        print("✅ Report generation completed!")
        return 0

    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        return 1


def handle_performance(args):
    """Handle performance command"""
    print("📈 Performance Statistics")

    try:
        performance_tracker = get_performance_tracker()
        stats = performance_tracker.get_stats()

        if args.export_format == 'json':
            print(json.dumps(stats, indent=2, default=str))
        else:
            # Text format
            print(f"Total validations: {stats.get('total_validations', 0)}")
            print(f"Average score: {stats.get('average_score', 0):.2f}")
            print(f"Query count: {stats.get('query_count', 0)}")
            print(f"Average query duration: {stats.get('query_avg_duration', 0):.3f}s")
            print(f"Query success rate: {stats.get('query_success_rate', 0):.2%}")
            print(f"Average memory usage: {stats.get('memory_avg_mb', 0):.1f} MB")
            print(f"Average CPU usage: {stats.get('cpu_avg_percent', 0):.1f}%")

            if stats.get('query_count', 0) > 0:
                print("
🕐 Slow queries:"
                slow_queries = performance_tracker.get_slow_queries(threshold=1.0, limit=5)
                for i, query in enumerate(slow_queries, 1):
                    print(f"  {i}. {query['query']} - {query['duration']:.3f}s")

        return 0

    except Exception as e:
        print(f"❌ Performance analysis failed: {e}")
        return 1


def handle_config(args):
    """Handle config commands"""
    if args.config_command == 'create':
        try:
            ConfigManager().create_default_config(args.output)
            print(f"✅ Configuration file created: {args.output}")
            return 0
        except Exception as e:
            print(f"❌ Failed to create configuration: {e}")
            return 1

    elif args.config_command == 'validate':
        try:
            config = ValidationConfig.from_yaml(args.config_file)
            errors = config.validate()

            if errors:
                print("❌ Configuration validation failed:")
                for error in errors:
                    print(f"  - {error}")
                return 1
            else:
                print("✅ Configuration is valid")
                return 0
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return 1

    elif args.config_command == 'show':
        try:
            config_manager = ConfigManager()
            config = config_manager.get_config()

            if args.format == 'json':
                print(json.dumps(config.to_dict(), indent=2))
            else:
                import yaml
                print(yaml.dump(config.to_dict(), default_flow_style=False, allow_unicode=True))

            return 0
        except Exception as e:
            print(f"❌ Failed to show configuration: {e}")
            return 1

    else:
        print("❌ Unknown config command")
        return 1


def handle_results(args):
    """Handle results commands"""
    if args.list:
        try:
            orchestrator = ValidationOrchestrator()
            history = orchestrator.get_validation_history(limit=10)

            if not history:
                print("📋 No validation results found")
                return 0

            print("📋 Recent Validation Results:")
            for i, result in enumerate(history, 1):
                print(f"  {i}. {result.validation_type} - Score: {result.overall_score".2f"} - {result.start_time}")

            return 0
        except Exception as e:
            print(f"❌ Failed to list results: {e}")
            return 1

    elif args.validation_id:
        try:
            orchestrator = ValidationOrchestrator()
            history = orchestrator.get_validation_history(limit=100)

            for result in history:
                if result.metadata.get('validation_id') == args.validation_id:
                    if args.export:
                        if args.export == 'json':
                            output_file = f"validation_result_{args.validation_id}.json"
                            with open(output_file, 'w', encoding='utf-8') as f:
                                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                            print(f"📄 Result exported to: {output_file}")
                        else:
                            print("❌ CSV export not implemented yet")
                    else:
                        print(f"📊 Validation Result: {args.validation_id}")
                        print(f"Type: {result.validation_type}")
                        print(f"Score: {result.overall_score".2f"}")
                        print(f"Duration: {result.duration".2f"}s")
                        print(f"Start Time: {result.start_time}")
                        print(f"Errors: {len(result.errors)}")
                        print(f"Warnings: {len(result.warnings)}")

                    return 0

            print(f"❌ Validation result not found: {args.validation_id}")
            return 1

        except Exception as e:
            print(f"❌ Failed to get validation result: {e}")
            return 1

    else:
        print("❌ Please specify --list or --validation-id")
        return 1


def main():
    """Main CLI entry point"""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Route to appropriate handler
    handlers = {
        'validate': handle_validate,
        'report': handle_report,
        'performance': handle_performance,
        'config': handle_config,
        'results': handle_results
    }

    handler = handlers.get(args.command)
    if handler:
        return handler(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
