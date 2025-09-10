#!/usr/bin/env python3
"""
Database Comparison Report Generator

Simple tool to compare two DuckDB databases and generate an HTML report.
Usage: python report.py db1.duckdb db2.duckdb --output report.html
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime

from db_analyzer import compare_tables
from templates import (
    HTML_TEMPLATE, 
    create_table_overview_row, 
    create_table_section
)


def get_db_name(db_path):
    """Extract database name from path"""
    return Path(db_path).stem


def generate_report(comparison_data, output_path):
    """Generate HTML report from comparison data"""
    
    # Extract data
    db1_path = comparison_data['db1_path']
    db2_path = comparison_data['db2_path']
    db1_name = get_db_name(db1_path)
    db2_name = get_db_name(db2_path)
    
    tables_only_db1 = comparison_data['tables_only_db1']
    tables_only_db2 = comparison_data['tables_only_db2']
    common_tables = comparison_data['common_tables']
    table_comparisons = comparison_data['table_comparisons']
    
    # Create smart table mappings 
    def find_smart_table_mappings():
        """Find logical table mappings between databases"""
        mappings = []
        used_db1_tables = set()
        used_db2_tables = set()
        
        # Get all available tables
        available_db1 = set(tables_only_db1)
        available_db2 = set(table_comparisons.keys())
        
        # Priority mappings for common patterns
        priority_patterns = [
            ('nedis2017', 'clinical_records', 'Main Clinical Data'),
            ('diag_er', 'diag_er', 'ER Diagnosis Data'),
            ('diag_adm', 'diag_adm', 'Admission Diagnosis Data'),
        ]
        
        # Apply priority patterns first
        for pattern1, pattern2, description in priority_patterns:
            # Find best matches
            db1_match = None
            db2_match = None
            
            for table1 in available_db1:
                if pattern1.lower() in table1.lower() and table1 not in used_db1_tables:
                    db1_match = table1
                    break
                    
            for table2 in available_db2:
                if pattern2.lower() in table2.lower() and table2 not in used_db2_tables:
                    db2_match = table2
                    break
            
            if db1_match and db2_match:
                mappings.append((db1_match, db2_match, description))
                used_db1_tables.add(db1_match)
                used_db2_tables.add(db2_match)
        
        # Then find exact base name matches
        for table1 in available_db1:
            if table1 in used_db1_tables:
                continue
            table1_base = table1.split('.')[-1]
            
            for table2 in available_db2:
                if table2 in used_db2_tables:
                    continue
                table2_base = table2.split('.')[-1]
                
                if table1_base == table2_base:
                    mappings.append((table1, table2, f"{table1_base} comparison"))
                    used_db1_tables.add(table1)
                    used_db2_tables.add(table2)
                    break
        
        return mappings
    
    special_mappings = find_smart_table_mappings()
    
    # Process special mappings
    for db1_table, db2_table, display_name in special_mappings:
        print(f"Creating mapping: {db1_table} ↔ {db2_table}")
        try:
            from db_analyzer import DatabaseAnalyzer
            analyzer1 = DatabaseAnalyzer(db1_path)
            analyzer1.connect()
            
            db1_info, db1_stats = analyzer1.analyze_table(db1_table)
            
            # Use existing db2 data or analyze if needed
            if db2_table in table_comparisons:
                db2_comp = table_comparisons[db2_table]
            else:
                # Analyze db2 table if not already done
                analyzer2 = DatabaseAnalyzer(db2_path)
                analyzer2.connect()
                db2_info, db2_stats = analyzer2.analyze_table(db2_table)
                db2_comp = {'db2': {'info': db2_info, 'stats': db2_stats}}
                analyzer2.disconnect()
                
            # Create new comparison entry
            table_comparisons[display_name] = {
                'db1': {'info': db1_info, 'stats': db1_stats},
                'db2': db2_comp['db2']
            }
            
            # Update lists - remove original entries to avoid duplication
            if db1_table in tables_only_db1:
                tables_only_db1.remove(db1_table)
            if db2_table in table_comparisons and db2_table != display_name:
                del table_comparisons[db2_table]
            if db2_table in common_tables:
                common_tables.remove(db2_table)
                
            analyzer1.disconnect()
        except Exception as e:
            print(f"Warning: Could not create mapping {display_name}: {e}")
    
    # Summary statistics
    common_tables_count = len(common_tables)
    db1_only_count = len(tables_only_db1)
    db2_only_count = len(tables_only_db2)
    total_comparisons = common_tables_count + db1_only_count + db2_only_count
    
    # Create table overview rows
    table_overview_rows = []
    
    # All tables with comparisons (including special mappings)
    all_compared_tables = sorted(list(table_comparisons.keys()))
    
    for table_name in all_compared_tables:
        comp = table_comparisons[table_name]
        db1_info = {
            'rows': comp['db1']['info'].row_count,
            'cols': comp['db1']['info'].column_count
        }
        db2_info = {
            'rows': comp['db2']['info'].row_count,
            'cols': comp['db2']['info'].column_count
        }
        status = 'match'  # Could be more sophisticated
        
        table_overview_rows.append(
            create_table_overview_row(table_name, db1_info, db2_info, status)
        )
    
    # Tables only in DB1
    for table_name in sorted(tables_only_db1):
        db1_info = {'rows': 0, 'cols': 0}  # Could fetch actual info
        table_overview_rows.append(
            create_table_overview_row(table_name, db1_info, None, 'missing')
        )
    
    # Tables only in DB2
    for table_name in sorted(tables_only_db2):
        db2_info = {'rows': 0, 'cols': 0}  # Could fetch actual info
        table_overview_rows.append(
            create_table_overview_row(table_name, None, db2_info, 'missing')
        )
    
    # Create detailed table sections
    table_sections = []
    for table_name in all_compared_tables:
        comp = table_comparisons[table_name]
        db1_info = comp['db1']['info']
        db1_stats = comp['db1']['stats']
        db2_info = comp['db2']['info']
        db2_stats = comp['db2']['stats']
        
        section = create_table_section(
            table_name, db1_info, db1_stats, db2_info, db2_stats, db1_name, db2_name
        )
        table_sections.append(section)
    
    # Generate HTML
    html_content = HTML_TEMPLATE.replace('{{db1_name}}', db1_name)
    html_content = html_content.replace('{{db2_name}}', db2_name)
    html_content = html_content.replace('{{common_tables_count}}', str(common_tables_count))
    html_content = html_content.replace('{{db1_only_count}}', str(db1_only_count))
    html_content = html_content.replace('{{db2_only_count}}', str(db2_only_count))
    html_content = html_content.replace('{{total_comparisons}}', str(total_comparisons))
    html_content = html_content.replace('{{table_overview_rows}}', '\n'.join(table_overview_rows))
    html_content = html_content.replace('{{table_sections}}', '\n'.join(table_sections))
    
    # Write to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Report generated: {output_path}")
    
    # Print summary to console
    print(f"\nComparison Summary:")
    print(f"Database 1: {db1_name} ({db1_path})")
    print(f"Database 2: {db2_name} ({db2_path})")
    print(f"Common tables: {common_tables_count}")
    print(f"Tables only in DB1: {db1_only_count}")
    print(f"Tables only in DB2: {db2_only_count}")
    
    if tables_only_db1:
        print(f"\nTables only in {db1_name}:")
        for table in sorted(tables_only_db1):
            print(f"  - {table}")
    
    if tables_only_db2:
        print(f"\nTables only in {db2_name}:")
        for table in sorted(tables_only_db2):
            print(f"  - {table}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Compare two DuckDB databases and generate an HTML report'
    )
    parser.add_argument('db1', help='Path to first database')
    parser.add_argument('db2', help='Path to second database')
    parser.add_argument(
        '--output', '-o', 
        default='comparison_report.html',
        help='Output HTML file (default: comparison_report.html)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Check if database files exist
    if not os.path.exists(args.db1):
        print(f"Error: Database file not found: {args.db1}")
        sys.exit(1)
    
    if not os.path.exists(args.db2):
        print(f"Error: Database file not found: {args.db2}")
        sys.exit(1)
    
    print("Starting database comparison...")
    print(f"Database 1: {args.db1}")
    print(f"Database 2: {args.db2}")
    
    try:
        # Perform comparison
        comparison_data = compare_tables(args.db1, args.db2)
        
        # Generate report
        generate_report(comparison_data, args.output)
        
        print(f"\nComparison completed successfully!")
        print(f"Report saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error during comparison: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()