import duckdb
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class TableInfo:
    """Basic table information"""
    name: str
    row_count: int
    column_count: int
    columns: List[Tuple[str, str]]  # (column_name, data_type)


@dataclass
class ColumnStats:
    """Column statistics"""
    name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    total_count: int
    min_val: Any = None
    max_val: Any = None
    mean_val: Any = None
    std_val: Any = None
    top_values: List[Tuple[Any, int]] = None  # (value, count)


class DatabaseAnalyzer:
    """Simple database analyzer for DuckDB"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = None
    
    def connect(self):
        """Connect to database"""
        self.conn = duckdb.connect(self.db_path, read_only=True)
    
    def disconnect(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def get_table_list(self) -> List[str]:
        """Get list of all tables in database"""
        # Check for tables in all schemas
        schemas_query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('information_schema', 'pg_catalog')"
        schemas = [row[0] for row in self.conn.execute(schemas_query).fetchall()]
        
        all_tables = []
        for schema in schemas:
            try:
                tables_query = f"SELECT table_name FROM information_schema.tables WHERE table_schema = '{schema}'"
                tables = [f"{schema}.{row[0]}" if schema != 'main' else row[0] 
                         for row in self.conn.execute(tables_query).fetchall()]
                all_tables.extend(tables)
            except:
                continue
        
        # Also try SHOW TABLES for main schema
        try:
            main_tables = [row[0] for row in self.conn.execute("SHOW TABLES").fetchall()]
            for table in main_tables:
                if table not in all_tables:
                    all_tables.append(table)
        except:
            pass
            
        return sorted(list(set(all_tables)))
    
    def get_table_info(self, table_name: str) -> TableInfo:
        """Get basic table information"""
        # Get column info
        columns = self.conn.execute(f"DESCRIBE {table_name}").fetchall()
        column_info = [(col[0], col[1]) for col in columns]
        
        # Get row count
        row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        
        return TableInfo(
            name=table_name,
            row_count=row_count,
            column_count=len(column_info),
            columns=column_info
        )
    
    def get_column_stats(self, table_name: str, column_name: str, data_type: str, sample_size: int = 10000) -> ColumnStats:
        """Get statistics for a specific column"""
        # Basic stats query
        stats_query = f"""
        SELECT 
            COUNT(*) as total_count,
            COUNT({column_name}) as non_null_count,
            COUNT(DISTINCT {column_name}) as unique_count
        FROM {table_name}
        """
        
        basic_stats = self.conn.execute(stats_query).fetchone()
        total_count = basic_stats[0]
        non_null_count = basic_stats[1]
        unique_count = basic_stats[2]
        
        null_count = total_count - non_null_count
        null_percentage = (null_count / total_count * 100) if total_count > 0 else 0
        
        # Initialize optional stats
        min_val = max_val = mean_val = std_val = None
        top_values = []
        
        # Type-specific statistics
        if 'INT' in data_type.upper() or 'DOUBLE' in data_type.upper() or 'FLOAT' in data_type.upper() or 'DECIMAL' in data_type.upper():
            # Numeric statistics
            try:
                numeric_query = f"""
                SELECT 
                    MIN({column_name}) as min_val,
                    MAX({column_name}) as max_val,
                    AVG({column_name}) as mean_val,
                    STDDEV({column_name}) as std_val
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                """
                numeric_stats = self.conn.execute(numeric_query).fetchone()
                if numeric_stats:
                    min_val, max_val, mean_val, std_val = numeric_stats
            except:
                pass
        
        # Top values for categorical data (if reasonable cardinality)
        if unique_count <= 50 and unique_count > 1:
            try:
                top_values_query = f"""
                SELECT {column_name}, COUNT(*) as count
                FROM {table_name}
                WHERE {column_name} IS NOT NULL
                GROUP BY {column_name}
                ORDER BY count DESC
                LIMIT 10
                """
                top_values = self.conn.execute(top_values_query).fetchall()
            except:
                pass
        
        return ColumnStats(
            name=column_name,
            data_type=data_type,
            null_count=null_count,
            null_percentage=null_percentage,
            unique_count=unique_count,
            total_count=total_count,
            min_val=min_val,
            max_val=max_val,
            mean_val=mean_val,
            std_val=std_val,
            top_values=top_values
        )
    
    def analyze_table(self, table_name: str) -> Tuple[TableInfo, Dict[str, ColumnStats]]:
        """Analyze a complete table"""
        table_info = self.get_table_info(table_name)
        column_stats = {}
        
        for column_name, data_type in table_info.columns:
            try:
                stats = self.get_column_stats(table_name, column_name, data_type)
                column_stats[column_name] = stats
            except Exception as e:
                print(f"Warning: Could not analyze column {column_name}: {e}")
        
        return table_info, column_stats


def compare_tables(db1_path: str, db2_path: str) -> Dict[str, Any]:
    """Compare two databases and return comparison results"""
    analyzer1 = DatabaseAnalyzer(db1_path)
    analyzer2 = DatabaseAnalyzer(db2_path)
    
    try:
        analyzer1.connect()
        analyzer2.connect()
        
        # Get table lists
        tables1 = set(analyzer1.get_table_list())
        tables2 = set(analyzer2.get_table_list())
        
        # Find table mappings (simple name matching for now)
        common_tables = tables1.intersection(tables2)
        only_db1 = tables1 - tables2
        only_db2 = tables2 - tables1
        
        # Analyze common tables
        table_comparisons = {}
        for table_name in common_tables:
            try:
                info1, stats1 = analyzer1.analyze_table(table_name)
                info2, stats2 = analyzer2.analyze_table(table_name)
                
                table_comparisons[table_name] = {
                    'db1': {'info': info1, 'stats': stats1},
                    'db2': {'info': info2, 'stats': stats2}
                }
            except Exception as e:
                print(f"Warning: Could not compare table {table_name}: {e}")
        
        return {
            'db1_path': db1_path,
            'db2_path': db2_path,
            'tables_only_db1': list(only_db1),
            'tables_only_db2': list(only_db2),
            'common_tables': list(common_tables),
            'table_comparisons': table_comparisons
        }
    
    finally:
        analyzer1.disconnect()
        analyzer2.disconnect()