"""
DuckDB 데이터베이스 연결 및 관리 모듈

NEDIS 합성 데이터 생성 시스템의 핵심 데이터베이스 인터페이스를 제공합니다.
안전한 연결 관리, 배치 처리, 트랜잭션 관리 등을 지원합니다.
"""

import duckdb
import pandas as pd
from pathlib import Path
from typing import Optional, Any, List, Dict, Generator
import logging
import contextlib
from datetime import datetime


class DatabaseManager:
    """DuckDB 연결 및 데이터베이스 작업 관리"""
    
    def __init__(self, db_path: str = "nedis_synthetic.duckdb"):
        """
        DuckDB 연결 관리자 초기화
        
        Args:
            db_path: DuckDB 데이터베이스 파일 경로
        """
        self.db_path = db_path
        self.conn = None
        self.logger = logging.getLogger(__name__)
        self._connect()
        self._setup_schemas()
        
    def _connect(self):
        """데이터베이스 연결"""
        try:
            self.conn = duckdb.connect(self.db_path)
            self.logger.info(f"Connected to database: {self.db_path}")
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            raise
            
    def _setup_schemas(self):
        """데이터베이스 스키마 초기화"""
        try:
            schema_path = Path("sql/create_schemas.sql")
            if schema_path.exists():
                schema_sql = schema_path.read_text(encoding='utf-8')
                self.conn.execute(schema_sql)
                self.logger.info("Database schemas created successfully")
            else:
                # 기본 스키마 생성
                self._create_basic_schemas()
        except Exception as e:
            self.logger.error(f"Failed to setup schemas: {e}")
            raise
            
    def _create_basic_schemas(self):
        """기본 스키마 생성 (fallback)"""
        schemas = [
            "CREATE SCHEMA IF NOT EXISTS nedis_original",
            "CREATE SCHEMA IF NOT EXISTS nedis_synthetic", 
            "CREATE SCHEMA IF NOT EXISTS nedis_meta"
        ]
        
        for schema_sql in schemas:
            self.conn.execute(schema_sql)
    
    def execute_query(self, query: str, params: Optional[List] = None) -> Any:
        """
        SQL 쿼리 실행
        
        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 파라미터 (옵션)
            
        Returns:
            쿼리 실행 결과
        """
        try:
            # 긴 쿼리는 일부만 로그에 출력
            log_query = query[:200] + "..." if len(query) > 200 else query
            self.logger.debug(f"Executing query: {log_query}")
            
            if params:
                result = self.conn.execute(query, params)
            else:
                result = self.conn.execute(query)
            return result
        except Exception as e:
            self.logger.error(f"Query execution failed: {e}")
            self.logger.error(f"Query: {query[:500]}")
            raise
            
    def fetch_dataframe(self, query: str, params: Optional[List] = None) -> pd.DataFrame:
        """
        쿼리 결과를 DataFrame으로 반환
        
        Args:
            query: 실행할 SQL 쿼리
            params: 쿼리 파라미터 (옵션)
            
        Returns:
            결과 DataFrame
        """
        result = self.execute_query(query, params)
        return result.fetchdf()
        
    def batch_insert(self, table: str, data: pd.DataFrame, batch_size: int = 10000):
        """
        대용량 데이터 배치 삽입
        
        Args:
            table: 대상 테이블명
            data: 삽입할 DataFrame
            batch_size: 배치 크기
        """
        total_rows = len(data)
        self.logger.info(f"Inserting {total_rows:,} rows into {table}")
        
        try:
            # 테이블의 컬럼 순서에 맞게 데이터 정렬
            self._align_dataframe_columns(table, data)
            
            for i in range(0, total_rows, batch_size):
                batch = data[i:i + batch_size]
                
                # DuckDB는 pandas DataFrame을 직접 INSERT 가능
                self.conn.execute(f"INSERT INTO {table} SELECT * FROM batch")
                
                if i % (batch_size * 10) == 0:  # 매 10 배치마다 로그
                    self.logger.debug(f"Inserted {i:,} / {total_rows:,} rows")
                    
            self.logger.info(f"Successfully inserted all {total_rows:,} rows into {table}")
            
        except Exception as e:
            self.logger.error(f"Batch insert failed: {e}")
            raise
            
    def _align_dataframe_columns(self, table: str, data: pd.DataFrame):
        """DataFrame 컬럼을 테이블 스키마에 맞게 정렬"""
        try:
            # 테이블 스키마 조회
            schema_query = f"DESCRIBE {table}"
            schema_df = self.fetch_dataframe(schema_query)
            table_columns = schema_df['column_name'].tolist()
            
            # DataFrame 컬럼 순서 조정
            missing_cols = set(table_columns) - set(data.columns)
            if missing_cols:
                self.logger.warning(f"Missing columns in data: {missing_cols}")
                
            # 공통 컬럼만 선택하고 순서 맞춤
            common_cols = [col for col in table_columns if col in data.columns]
            return data[common_cols]
            
        except Exception as e:
            self.logger.warning(f"Could not align columns for {table}: {e}")
            return data
    
    @contextlib.contextmanager
    def transaction(self):
        """트랜잭션 컨텍스트 매니저"""
        try:
            self.conn.execute("BEGIN TRANSACTION")
            self.logger.debug("Transaction started")
            yield
            self.conn.execute("COMMIT")
            self.logger.debug("Transaction committed")
        except Exception as e:
            self.conn.execute("ROLLBACK")
            self.logger.error(f"Transaction rolled back: {e}")
            raise
            
    def create_checkpoint(self, checkpoint_name: str):
        """체크포인트 생성"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_table = f"nedis_meta.checkpoint_{checkpoint_name}_{timestamp}"
        
        # 진행 상황을 별도 테이블에 저장
        query = f"""
        CREATE TABLE {checkpoint_table} AS 
        SELECT '{checkpoint_name}' as checkpoint_name, 
               '{timestamp}' as created_at,
               'completed' as status
        """
        
        self.execute_query(query)
        self.logger.info(f"Checkpoint created: {checkpoint_table}")
        
    def get_table_count(self, table: str) -> int:
        """테이블 레코드 수 조회"""
        query = f"SELECT COUNT(*) as count FROM {table}"
        result = self.fetch_dataframe(query)
        return result['count'][0]
        
    def table_exists(self, table: str) -> bool:
        """테이블 존재 여부 확인"""
        try:
            self.execute_query(f"SELECT 1 FROM {table} LIMIT 1")
            return True
        except:
            return False
            
    def get_table_info(self, table: str) -> Dict[str, Any]:
        """테이블 정보 조회"""
        info = {}
        
        try:
            # 테이블 스키마
            schema_df = self.fetch_dataframe(f"DESCRIBE {table}")
            info['schema'] = schema_df.to_dict('records')
            
            # 레코드 수
            info['row_count'] = self.get_table_count(table)
            
            # 테이블 크기 (근사치)
            size_query = f"SELECT pg_size_pretty(pg_total_relation_size('{table}')) as size"
            try:
                size_df = self.fetch_dataframe(size_query)
                info['size'] = size_df['size'][0]
            except:
                info['size'] = 'N/A'
                
        except Exception as e:
            self.logger.error(f"Failed to get table info for {table}: {e}")
            info['error'] = str(e)
            
        return info
    
    def vacuum_analyze(self):
        """데이터베이스 최적화"""
        try:
            self.conn.execute("VACUUM")
            self.conn.execute("ANALYZE")
            self.logger.info("Database optimized (VACUUM + ANALYZE)")
        except Exception as e:
            self.logger.warning(f"Database optimization failed: {e}")
    
    def close(self):
        """데이터베이스 연결 종료"""
        if self.conn:
            self.conn.close()
            self.logger.info("Database connection closed")
            
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class QueryBuilder:
    """SQL 쿼리 빌더 유틸리티"""
    
    @staticmethod
    def select_with_conditions(table: str, columns: List[str], 
                             conditions: Dict[str, Any], 
                             limit: Optional[int] = None) -> str:
        """조건부 SELECT 쿼리 생성"""
        cols = ", ".join(columns) if columns else "*"
        query = f"SELECT {cols} FROM {table}"
        
        if conditions:
            where_clauses = []
            for col, val in conditions.items():
                if isinstance(val, str):
                    where_clauses.append(f"{col} = '{val}'")
                elif isinstance(val, list):
                    val_str = "', '".join(str(v) for v in val)
                    where_clauses.append(f"{col} IN ('{val_str}')")
                else:
                    where_clauses.append(f"{col} = {val}")
            
            query += " WHERE " + " AND ".join(where_clauses)
            
        if limit:
            query += f" LIMIT {limit}"
            
        return query
        
    @staticmethod
    def insert_from_select(target_table: str, source_query: str) -> str:
        """SELECT 결과를 테이블에 INSERT하는 쿼리 생성"""
        return f"INSERT INTO {target_table} {source_query}"