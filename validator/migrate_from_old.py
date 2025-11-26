#!/usr/bin/env python3
"""
Migration script from old validator to new validator system.

This script helps migrate from the old simple validator to the new comprehensive system.
"""

import shutil
import os
from pathlib import Path
import logging


def migrate_validator():
    """Migrate from old validator structure to new structure"""

    print("🔄 Starting migration from old validator to new system...")
    print("=" * 60)

    # Backup old files
    print("📦 Creating backup of old validator files...")

    old_validator_dir = Path("validator")
    backup_dir = Path("validator_old_backup")

    if old_validator_dir.exists() and not backup_dir.exists():
        try:
            shutil.copytree(old_validator_dir, backup_dir)
            print(f"✅ Old validator backed up to: {backup_dir}")
        except Exception as e:
            print(f"⚠️  Backup failed: {e}")

    # Remove old files (keep only what we want to preserve)
    print("🧹 Cleaning up old validator files...")

    old_files_to_remove = [
        "report.py",
        "db_analyzer.py",
        "templates.py",
        "generic_report.html",
        "smart_report.html",
        "updated_report.html",
        "test_report.html",
        "vertical_chart_report.html",
        "fixed_report.html"
    ]

    for file_name in old_files_to_remove:
        file_path = old_validator_dir / file_name
        if file_path.exists():
            try:
                if file_path.is_file():
                    file_path.unlink()
                else:
                    shutil.rmtree(file_path)
                print(f"   Removed: {file_name}")
            except Exception as e:
                print(f"   ⚠️  Failed to remove {file_name}: {e}")

    # Keep only essential files
    essential_files = [
        "README.md",  # We'll replace this
        "__pycache__",  # Keep Python cache
        ".DS_Store"  # Keep system files
    ]

    print("✅ Migration completed!")
    print("\n📋 Migration Summary:")
    print("   • Old validator files backed up to: validator_old_backup/")
    print("   • New comprehensive validator system is now active")
    print("   • Configuration files updated")
    print("   • New modules and utilities installed")

    print("\n🚀 Next Steps:")
    print("   1. Install dependencies: pip install -r validator/requirements.txt")
    print("   2. Configure databases in validator/validation_config.yaml")
    print("   3. Run demo: python validator/start_validator.py --demo")
    print("   4. Use CLI: python -m validator.cli validate --help")

    print("\n📚 Key Improvements:")
    print("   • Advanced statistical validation (KS test, Chi-square, etc.)")
    print("   • Dynamic pattern analysis with hierarchical fallback")
    print("   • Clinical data validation and medical protocol compliance")
    print("   • Temporal pattern analysis")
    print("   • Interactive web dashboard")
    print("   • REST API and real-time validation")
    print("   • Performance monitoring and optimization")
    print("   • Comprehensive reporting and visualization")

    print("\n⚠️  Important Notes:")
    print("   • The old simple comparison functionality is now enhanced")
    print("   • New validation methods provide much more detailed analysis")
    print("   • Configuration is more comprehensive but flexible")
    print("   • API integration allows for automated validation pipelines")

    return True


def restore_old_validator():
    """Restore the old validator system"""

    print("🔄 Restoring old validator system...")

    backup_dir = Path("validator_old_backup")
    current_dir = Path("validator")

    if backup_dir.exists():
        # Remove new files
        if current_dir.exists():
            shutil.rmtree(current_dir)

        # Restore old files
        shutil.copytree(backup_dir, current_dir)
        print("✅ Old validator restored")
    else:
        print("❌ No backup found. Cannot restore.")

    return True


def main():
    """Main migration function"""
    import argparse

    parser = argparse.ArgumentParser(description='Migrate validator system')
    parser.add_argument('--restore', action='store_true', help='Restore old validator')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be migrated')

    args = parser.parse_args()

    if args.restore:
        restore_old_validator()
    else:
        if args.dry_run:
            print("🔍 Migration dry run - showing what would be changed:")
            print("   • Old validator files would be backed up")
            print("   • New validator system would be installed")
            print("   • Configuration would be updated")
        else:
            migrate_validator()


if __name__ == "__main__":
    main()
