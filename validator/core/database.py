"""
Database connection management for the validation system.

This module handles:
- DuckDB connection management
- Connection pooling
- Query execution and optimization
- Schema management
- Data sampling and caching
"""

import duckdb
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time
import hashlib


@dataclass
class DatabaseConnection:
    """Database connection information"""
    path: str
    schema: str
    connection: Optional[duckdb.DuckDBPyConnection] = None
    last_used: Optional[datetime] = None
    is_connected: bool = False


class DatabaseManager:
    """Enhanced database manager with connection pooling and caching"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize database manager

        Args:
            config: Database configuration dictionary
        """
        self.logger = logging.getLogger(__name__)

        # Connection pool
        self._connections: Dict[str, DatabaseConnection] = {}
        self._pool_lock = threading.Lock()

        # Query cache
        self._query_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_lock = threading.Lock()
        self._cache_ttl = timedelta(minutes=30)

        # Performance tracking
        self._query_stats: Dict[str, int] = {}
        self._stats_lock = threading.Lock()

        # Default configuration
        self.config = config or {}

        # Set DuckDB configurations
        self._setup_duckdb_config()

    def _setup_duckdb_config(self):
        """Setup DuckDB configurations for optimal performance"""
        # These will be applied to new connections
        self.duckdb_config = {
            'threads': '4',  # Number of threads for parallel processing
            'max_memory': '2GB',  # Memory limit per query
            'temp_directory': '/tmp/duckdb_temp'  # Temporary directory
        }

    def connect(self, db_path: str, schema: str = 'main') -> DatabaseConnection:
        """
        Create or get existing database connection

        Args:
            db_path: Path to DuckDB database file
            schema: Database schema name

        Returns:
            Database connection object
        """
        connection_key = f"{db_path}:{schema}"

        with self._pool_lock:
            if connection_key not in self._connections:
                self._connections[connection_key] = DatabaseConnection(
                    path=db_path,
                    schema=schema
                )

            connection = self._connections[connection_key]

            if not connection.is_connected:
                try:
                    # Create new connection
                    conn = duckdb.connect(db_path, read_only=True)

                    # Apply configurations
                    for key, value in self.duckdb_config.items():
                        conn.execute(f"SET {key} = '{value}'")

                    connection.connection = conn
                    connection.is_connected = True
                    connection.last_used = datetime.now()

                    self.logger.info(f"Connected to database: {db_path} (schema: {schema})")

                except Exception as e:
                    self.logger.error(f"Failed to connect to database {db_path}: {e}")
                    raise

            return connection

    def disconnect(self, db_path: str = None, schema: str = None):
        """Disconnect from database(s)"""
        with self._pool_lock:
            if db_path is None:
                # Disconnect all connections
                for connection_key, connection in self._connections.items():
                    if connection.is_connected and connection.connection:
                        try:
                            connection.connection.close()
                            connection.is_connected = False
                            connection.connection = None
                        except Exception as e:
                            self.logger.warning(f"Error closing connection {connection_key}: {e}")

                self._connections.clear()
                self.logger.info("Disconnected from all databases")

            else:
                # Disconnect specific connection
                connection_key = f"{db_path}:{schema}"
                if connection_key in self._connections:
                    connection = self._connections[connection_key]
                    if connection.is_connected and connection.connection:
                        try:
                            connection.connection.close()
                            connection.is_connected = False
                            connection.connection = None
                        except Exception as e:
                            self.logger.warning(f"Error closing connection {connection_key}: {e}")

                    del self._connections[connection_key]
                    self.logger.info(f"Disconnected from database: {db_path}")

    def execute_query(self, query: str, db_path: str, schema: str = 'main',
                     params: Optional[Union[Dict[str, Any], List[Any]]] = None) -> Any:
        """
        Execute SQL query with caching and performance tracking

        Args:
            query: SQL query to execute
            db_path: Database path
            schema: Database schema
            params: Query parameters

        Returns:
            Query result
        """
        # Create cache key
        cache_key = self._create_cache_key(query, db_path, schema, params)

        # Check cache first
        with self._cache_lock:
            if cache_key in self._query_cache:
                result, timestamp = self._query_cache[cache_key]
                if datetime.now() - timestamp < self._cache_ttl:
                    self._track_query(query, 'cache_hit')
                    return result

        # Get connection
        connection = self.connect(db_path, schema)

        try:
            # Update last used time
            connection.last_used = datetime.now()

            # Execute query
            start_time = time.time()
            if params:
                if isinstance(params, dict):
                    # Convert dict to list of values for positional parameters
                    result = connection.connection.execute(query, list(params.values())).fetchall()
                else:
                    result = connection.connection.execute(query, params).fetchall()
            else:
                result = connection.connection.execute(query).fetchall()

            execution_time = time.time() - start_time

            # Cache result
            with self._cache_lock:
                self._query_cache[cache_key] = (result, datetime.now())

            # Track performance
            self._track_query(query, 'executed', execution_time)

            return result

        except Exception as e:
            self._track_query(query, 'failed')
            self.logger.error(f"Query execution failed: {query[:100]}... - {e}")
            raise

    def fetch_dataframe(self, query: str, db_path: str, schema: str = 'main',
                       params: Optional[Union[Dict[str, Any], List[Any]]] = None) -> pd.DataFrame:
        """
        Execute query and return pandas DataFrame

        Args:
            query: SQL query to execute
            db_path: Database path
            schema: Database schema
            params: Query parameters

        Returns:
            pandas DataFrame with query results
        """
        cache_key = self._create_cache_key(query, db_path, schema, params)

        # Check cache first
        with self._cache_lock:
            if cache_key in self._query_cache:
                df, timestamp = self._query_cache[cache_key]
                if isinstance(df, pd.DataFrame) and datetime.now() - timestamp < self._cache_ttl:
                    self._track_query(query, 'cache_hit')
                    return df

        # Create fresh connection for each dataframe query to avoid internal errors
        try:
            conn = duckdb.connect(db_path, read_only=True)

            # Apply configurations
            for key, value in self.duckdb_config.items():
                try:
                    conn.execute(f"SET {key} = '{value}'")
                except:
                    pass  # Some settings might not be available

            # Execute query with retry mechanism
            start_time = time.time()
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    if params:
                        if isinstance(params, dict):
                            result = conn.execute(query, list(params.values()))
                        else:
                            result = conn.execute(query, params)
                    else:
                        result = conn.execute(query)

                    # Convert to DataFrame safely
                    df = result.df()
                    break

                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    time.sleep(0.1 * (attempt + 1))  # Exponential backoff

            execution_time = time.time() - start_time

            # Cache result
            with self._cache_lock:
                self._query_cache[cache_key] = (df, datetime.now())

            # Track performance
            self._track_query(query, 'executed', execution_time)

            return df

        except Exception as e:
            self._track_query(query, 'failed')
            self.logger.error(f"DataFrame query failed: {query[:100]}... - {e}")
            return pd.DataFrame()  # Return empty DataFrame instead of raising
        finally:
            try:
                conn.close()
            except:
                pass

    def _create_cache_key(self, query: str, db_path: str, schema: str,
                         params: Optional[Union[Dict[str, Any], List[Any]]] = None) -> str:
        """Create cache key for query"""
        key_components = [query, db_path, schema]
        if params:
            if isinstance(params, dict):
                key_components.append(str(sorted(params.items())))
            else:
                key_components.append(str(params))

        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _track_query(self, query: str, status: str, execution_time: float = None):
        """Track query performance statistics"""
        with self._stats_lock:
            if query not in self._query_stats:
                self._query_stats[query] = {'executed': 0, 'failed': 0, 'cache_hit': 0, 'total_time': 0.0}

            self._query_stats[query][status] += 1
            if execution_time:
                self._query_stats[query]['total_time'] += execution_time

    def get_table_info(self, table_name: str, db_path: str, schema: str = 'main') -> Dict[str, Any]:
        """
        Get table information including schema, row count, columns

        Args:
            table_name: Table name
            db_path: Database path
            schema: Database schema

        Returns:
            Dictionary with table information
        """
        try:
            # Get table schema
            schema_query = f"DESCRIBE {table_name}"
            schema_df = self.fetch_dataframe(schema_query, db_path, schema)

            # Get row count
            count_query = f"SELECT COUNT(*) as row_count FROM {table_name}"
            count_df = self.fetch_dataframe(count_query, db_path, schema)
            row_count = count_df['row_count'].iloc[0]

            # Get sample data for type inference
            sample_query = f"SELECT * FROM {table_name} ORDER BY RANDOM() LIMIT 100"
            sample_df = self.fetch_dataframe(sample_query, db_path, schema)

            return {
                'table_name': table_name,
                'row_count': row_count,
                'column_count': len(schema_df),
                'columns': schema_df.to_dict('records'),
                'sample_data': sample_df,
                'schema': schema,
                'database_path': db_path
            }

        except Exception as e:
            self.logger.error(f"Failed to get table info for {table_name}: {e}")
            return {
                'table_name': table_name,
                'error': str(e),
                'row_count': 0,
                'column_count': 0,
                'columns': [],
                'sample_data': pd.DataFrame()
            }

    def get_table_list(self, db_path: str, schema: str = 'main') -> List[str]:
        """
        Get list of all tables in database

        Args:
            db_path: Database path
            schema: Database schema

        Returns:
            List of table names
        """
        try:
            # Try to get table list from information schema
            query = """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = ?
            ORDER BY table_name
            """
            df = self.fetch_dataframe(query, db_path, schema, [schema])

            if not df.empty:
                return df['table_name'].tolist()

            # Fallback: try SHOW TABLES
            try:
                query = "SHOW TABLES"
                df = self.fetch_dataframe(query, db_path, schema)
                return df.iloc[:, 0].tolist()
            except:
                return []

        except Exception as e:
            self.logger.error(f"Failed to get table list: {e}")
            return []

    def sample_data(self, table_name: str, sample_size: int, db_path: str,
                   schema: str = 'main', method: str = 'random') -> pd.DataFrame:
        """
        Sample data from table using different sampling methods

        Args:
            table_name: Table name
            sample_size: Number of samples to draw
            db_path: Database path
            schema: Database schema
            method: Sampling method ('random', 'stratified', 'systematic')

        Returns:
            pandas DataFrame with sampled data
        """
        try:
            if method == 'random':
                # Handle schema.table format
                full_table_name = f"{schema}.{table_name}" if schema != 'main' else table_name
                query = f"""
                SELECT * FROM {full_table_name}
                ORDER BY RANDOM()
                LIMIT ?
                """
                params = [sample_size]

            elif method == 'systematic':
                # Use reservoir sampling for systematic sampling
                full_table_name = f"{schema}.{table_name}" if schema != 'main' else table_name
                query = f"""
                SELECT * FROM (
                    SELECT *,
                           ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) as rn
                    FROM {full_table_name}
                ) t
                WHERE (rn - 1) % GREATEST(1, (SELECT COUNT(*) FROM {full_table_name}) / ?) = 0
                """
                params = [sample_size]

            else:
                # Default to random sampling
                full_table_name = f"{schema}.{table_name}" if schema != 'main' else table_name
                query = f"""
                SELECT * FROM {full_table_name}
                ORDER BY RANDOM()
                LIMIT ?
                """
                params = [sample_size]

            return self.fetch_dataframe(query, db_path, schema, params)

        except Exception as e:
            self.logger.error(f"Failed to sample data from {table_name}: {e}")
            return pd.DataFrame()

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        with self._pool_lock:
            total_connections = len(self._connections)
            active_connections = sum(1 for conn in self._connections.values() if conn.is_connected)

        with self._stats_lock:
            total_queries = sum(stats['executed'] + stats['failed'] for stats in self._query_stats.values())
            cache_hits = sum(stats.get('cache_hit', 0) for stats in self._query_stats.values())

        return {
            'total_connections': total_connections,
            'active_connections': active_connections,
            'total_queries': total_queries,
            'cache_hits': cache_hits,
            'cache_size': len(self._query_cache),
            'query_stats': dict(self._query_stats)
        }

    @contextmanager
    def transaction(self, db_path: str, schema: str = 'main'):
        """Context manager for database transactions"""
        connection = self.connect(db_path, schema)
        original_readonly = True  # Since we use read_only connections

        try:
            # For write operations, we'd need a separate connection
            # This is a placeholder for future write transaction support
            yield connection
        finally:
            # Restore original state
            pass

    def __del__(self):
        """Cleanup connections on destruction"""
        try:
            self.disconnect()
        except:
            pass


# Global database manager instance
_default_db_manager = DatabaseManager()


def get_database_manager() -> DatabaseManager:
    """Get global database manager instance"""
    return _default_db_manager


def set_database_config(config: Dict[str, Any]):
    """Set global database configuration"""
    global _default_db_manager
    _default_db_manager = DatabaseManager(config)
