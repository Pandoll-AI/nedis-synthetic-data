"""
Iterative Proportional Fitting (IPF) Marginal Adjuster

IPF 알고리즘을 사용하여 병원 할당 결과를 목표 마진(target margins)에 맞게 조정합니다.
행 제약(인구 그룹별 방문 수)과 열 제약(병원별 용량)을 동시에 만족하도록 반복 조정합니다.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import sparse
import warnings

from ..core.database import DatabaseManager
from ..core.config import ConfigManager


class IPFMarginalAdjuster:
    """IPF 기반 마진 조정기"""
    
    def __init__(self, db_manager: DatabaseManager, config: ConfigManager, 
                 max_iterations: int = 100, tolerance: float = 0.001):
        """
        IPF 마진 조정기 초기화
        
        Args:
            db_manager: 데이터베이스 관리자
            config: 설정 관리자
            max_iterations: 최대 반복 횟수
            tolerance: 수렴 허용 오차
        """
        self.db = db_manager
        self.config = config
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
        # IPF 수렴 이력
        self.convergence_history = []
        
    def adjust_to_margins(self, date_str: str) -> Dict[str, Any]:
        """
        IPF 알고리즘을 사용하여 할당을 목표 마진에 맞게 조정
        
        Args:
            date_str: 조정할 날짜 ('YYYYMMDD' 형식)
            
        Returns:
            조정 결과 딕셔너리
        """
        
        self.logger.info(f"Starting IPF adjustment for date: {date_str}")
        
        try:
            # 1. 현재 할당 매트릭스 로드
            current_allocation = self._load_allocation_matrix(date_str)
            
            if current_allocation.empty:
                self.logger.warning(f"No allocation data found for date: {date_str}")
                return {'success': False, 'reason': 'No data'}
            
            # 2. 목표 마진 로드
            target_margins = self._load_target_margins(date_str)
            
            # 3. IPF 알고리즘 실행
            adjusted_matrix, convergence_info = self._run_ipf_algorithm(
                current_allocation, target_margins
            )
            
            # 4. 정수화 (Controlled Rounding)
            integer_matrix = self._controlled_rounding(adjusted_matrix, target_margins)
            
            # 5. 결과를 데이터베이스에 업데이트
            self._update_allocation_table(date_str, integer_matrix)
            
            # 6. 조정 결과 검증
            validation_results = self._validate_adjustment(date_str, target_margins)
            
            result = {
                'success': True,
                'date': date_str,
                'iterations': convergence_info['iterations'],
                'final_error': convergence_info['final_error'],
                'converged': convergence_info['converged'],
                'validation': validation_results,
                'adjustment_summary': self._get_adjustment_summary(current_allocation, integer_matrix)
            }
            
            self.logger.info(
                f"IPF adjustment completed: {convergence_info['iterations']} iterations, "
                f"error: {convergence_info['final_error']:.6f}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"IPF adjustment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def _load_allocation_matrix(self, date_str: str) -> pd.DataFrame:
        """현재 할당 매트릭스 로드"""
        
        try:
            allocation_data = self.db.fetch_dataframe("""
                SELECT 
                    pat_do_cd, pat_age_gr, pat_sex, emorg_cd, allocated_count
                FROM nedis_synthetic.hospital_allocations
                WHERE vst_dt = ?
                ORDER BY pat_do_cd, pat_age_gr, pat_sex, emorg_cd
            """, [date_str])
            
            if len(allocation_data) == 0:
                return pd.DataFrame()
            
            # 피벗 테이블로 매트릭스 형태 변환
            matrix = allocation_data.pivot_table(
                index=['pat_do_cd', 'pat_age_gr', 'pat_sex'],
                columns='emorg_cd',
                values='allocated_count',
                fill_value=0
            )
            
            self.logger.info(f"Loaded allocation matrix: {matrix.shape[0]} groups × {matrix.shape[1]} hospitals")
            return matrix
            
        except Exception as e:
            self.logger.error(f"Failed to load allocation matrix: {e}")
            return pd.DataFrame()
    
    def _load_target_margins(self, date_str: str) -> Dict[str, pd.Series]:
        """목표 마진 로드 (행 합: 인구 그룹별 방문 수, 열 합: 병원별 용량)"""
        
        try:
            # 행 마진: 인구 그룹별 목표 방문 수
            row_margins = self.db.fetch_dataframe("""
                SELECT 
                    pat_do_cd, pat_age_gr, pat_sex, synthetic_daily_count
                FROM nedis_synthetic.daily_volumes
                WHERE vst_dt = ?
                ORDER BY pat_do_cd, pat_age_gr, pat_sex
            """, [date_str])
            
            if len(row_margins) == 0:
                raise ValueError(f"No target row margins found for date: {date_str}")
            
            # 행 마진을 MultiIndex Series로 변환
            row_margins_series = row_margins.set_index(['pat_do_cd', 'pat_age_gr', 'pat_sex'])['synthetic_daily_count']
            
            # 열 마진: 병원별 목표 용량 (약간의 여유 포함)
            col_margins = self.db.fetch_dataframe("""
                SELECT 
                    emorg_cd,
                    ROUND(daily_capacity_mean + 1.5 * daily_capacity_std) as target_capacity
                FROM nedis_meta.hospital_capacity
                ORDER BY emorg_cd
            """)
            
            col_margins_series = col_margins.set_index('emorg_cd')['target_capacity']
            
            self.logger.info(f"Target margins: {len(row_margins_series)} row margins, {len(col_margins_series)} col margins")
            
            return {
                'row_margins': row_margins_series,
                'col_margins': col_margins_series
            }
            
        except Exception as e:
            self.logger.error(f"Failed to load target margins: {e}")
            raise
    
    def _run_ipf_algorithm(self, allocation_matrix: pd.DataFrame, 
                          target_margins: Dict[str, pd.Series]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        IPF 알고리즘 실행
        
        Args:
            allocation_matrix: 현재 할당 매트릭스
            target_margins: 목표 마진 (행/열)
            
        Returns:
            (조정된 매트릭스, 수렴 정보)
        """
        
        self.logger.info("Running IPF algorithm")
        
        # 매트릭스를 NumPy 배열로 변환
        matrix = allocation_matrix.values.astype(float)
        row_targets = target_margins['row_margins'].reindex(allocation_matrix.index).values
        col_targets = target_margins['col_margins'].reindex(allocation_matrix.columns).values
        
        # 0으로 나누기 방지를 위한 작은 값 추가
        epsilon = 1e-10
        matrix = matrix + epsilon
        
        convergence_history = []
        
        for iteration in range(self.max_iterations):
            # 이전 매트릭스 저장 (수렴 검사용)
            prev_matrix = matrix.copy()
            
            # Step 1: 행 조정 (Row scaling)
            row_sums = matrix.sum(axis=1)
            row_factors = np.divide(row_targets, row_sums, 
                                  out=np.ones_like(row_targets), where=row_sums!=0)
            
            matrix = matrix * row_factors.reshape(-1, 1)
            
            # Step 2: 열 조정 (Column scaling)  
            col_sums = matrix.sum(axis=0)
            col_factors = np.divide(col_targets, col_sums,
                                  out=np.ones_like(col_targets), where=col_sums!=0)
            
            matrix = matrix * col_factors.reshape(1, -1)
            
            # 수렴 검사
            matrix_change = np.abs(matrix - prev_matrix).mean()
            
            # 마진 오차 계산
            current_row_sums = matrix.sum(axis=1)
            current_col_sums = matrix.sum(axis=0)
            
            row_error = np.abs(current_row_sums - row_targets).mean()
            col_error = np.abs(current_col_sums - col_targets).mean()
            total_error = (row_error + col_error) / 2
            
            convergence_history.append({
                'iteration': iteration + 1,
                'matrix_change': matrix_change,
                'row_error': row_error,
                'col_error': col_error,
                'total_error': total_error
            })
            
            # 수렴 조건 확인
            if total_error < self.tolerance:
                self.logger.info(f"IPF converged at iteration {iteration + 1}")
                break
                
            # 변화가 너무 작으면 조기 종료
            if matrix_change < self.tolerance / 10:
                self.logger.info(f"IPF stopped due to minimal change at iteration {iteration + 1}")
                break
        else:
            self.logger.warning(f"IPF did not converge within {self.max_iterations} iterations")
        
        # 결과를 DataFrame으로 변환
        adjusted_matrix = pd.DataFrame(
            matrix, 
            index=allocation_matrix.index,
            columns=allocation_matrix.columns
        )
        
        # epsilon 제거
        adjusted_matrix = np.maximum(adjusted_matrix - epsilon, 0)
        
        convergence_info = {
            'iterations': len(convergence_history),
            'converged': total_error < self.tolerance,
            'final_error': total_error,
            'history': convergence_history
        }
        
        self.convergence_history = convergence_history
        
        return adjusted_matrix, convergence_info
    
    def _controlled_rounding(self, float_matrix: pd.DataFrame, 
                           target_margins: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Controlled Rounding: 소수 매트릭스를 정수로 변환하면서 마진 보존
        
        Args:
            float_matrix: 실수 값 매트릭스
            target_margins: 목표 마진
            
        Returns:
            정수 매트릭스
        """
        
        self.logger.info("Applying controlled rounding")
        
        try:
            # 기본 정수 부분
            integer_matrix = np.floor(float_matrix.values)
            
            # 소수 부분
            fractional_matrix = float_matrix.values - integer_matrix
            
            # 각 행(인구 그룹)별로 정수화 조정
            for i, (group_idx, target) in enumerate(target_margins['row_margins'].items()):
                if group_idx not in float_matrix.index:
                    continue
                
                row_idx = float_matrix.index.get_loc(group_idx)
                current_sum = integer_matrix[row_idx].sum()
                target_int = int(round(target))
                needed = target_int - current_sum
                
                if needed > 0:
                    # 소수 부분이 큰 순서로 1씩 추가
                    fractional_row = fractional_matrix[row_idx]
                    top_indices = np.argsort(fractional_row)[::-1][:needed]
                    integer_matrix[row_idx, top_indices] += 1
                elif needed < 0:
                    # 정수 부분이 큰 셀에서 1씩 제거
                    integer_row = integer_matrix[row_idx]
                    removable_indices = np.where(integer_row > 0)[0]
                    if len(removable_indices) >= abs(needed):
                        # 정수 값이 큰 순서로 제거
                        sorted_indices = removable_indices[np.argsort(integer_row[removable_indices])[::-1]]
                        remove_indices = sorted_indices[:abs(needed)]
                        integer_matrix[row_idx, remove_indices] -= 1
            
            # DataFrame으로 변환
            result_matrix = pd.DataFrame(
                integer_matrix.astype(int),
                index=float_matrix.index,
                columns=float_matrix.columns
            )
            
            # 음수 값 제거
            result_matrix = result_matrix.clip(lower=0)
            
            self.logger.info("Controlled rounding completed")
            return result_matrix
            
        except Exception as e:
            self.logger.error(f"Controlled rounding failed: {e}")
            # 실패 시 단순 반올림 사용
            return float_matrix.round().astype(int).clip(lower=0)
    
    def _update_allocation_table(self, date_str: str, adjusted_matrix: pd.DataFrame):
        """조정된 매트릭스를 데이터베이스에 업데이트"""
        
        self.logger.info("Updating allocation table with IPF-adjusted values")
        
        try:
            # 기존 할당 데이터 삭제
            self.db.execute_query("""
                DELETE FROM nedis_synthetic.hospital_allocations 
                WHERE vst_dt = ?
            """, [date_str])
            
            # 조정된 할당 데이터 삽입
            update_records = []
            
            for (pat_do_cd, pat_age_gr, pat_sex), row in adjusted_matrix.iterrows():
                for emorg_cd, allocated_count in row.items():
                    if allocated_count > 0:  # 0인 값은 저장하지 않음
                        update_records.append({
                            'vst_dt': date_str,
                            'emorg_cd': emorg_cd,
                            'pat_do_cd': pat_do_cd,
                            'pat_age_gr': pat_age_gr,
                            'pat_sex': pat_sex,
                            'allocated_count': int(allocated_count),
                            'overflow_received': 0,  # IPF 조정 후이므로 초기화
                            'allocation_method': 'ipf_adjusted'
                        })
            
            # 배치 삽입
            for record in update_records:
                self.db.execute_query("""
                    INSERT INTO nedis_synthetic.hospital_allocations
                    (vst_dt, emorg_cd, pat_do_cd, pat_age_gr, pat_sex, 
                     allocated_count, overflow_received, allocation_method)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (record['vst_dt'], record['emorg_cd'], record['pat_do_cd'],
                      record['pat_age_gr'], record['pat_sex'], record['allocated_count'],
                      record['overflow_received'], record['allocation_method']))
            
            self.logger.info(f"Updated {len(update_records)} allocation records")
            
        except Exception as e:
            self.logger.error(f"Failed to update allocation table: {e}")
            raise
    
    def _validate_adjustment(self, date_str: str, target_margins: Dict[str, pd.Series]) -> Dict[str, Any]:
        """IPF 조정 결과 검증"""
        
        try:
            # 조정된 할당 결과 로드
            adjusted_data = self.db.fetch_dataframe("""
                SELECT pat_do_cd, pat_age_gr, pat_sex, emorg_cd, allocated_count
                FROM nedis_synthetic.hospital_allocations
                WHERE vst_dt = ?
            """, [date_str])
            
            if len(adjusted_data) == 0:
                return {'valid': False, 'reason': 'No adjusted data'}
            
            # 행 마진 검증 (인구 그룹별 총 방문 수)
            actual_row_sums = adjusted_data.groupby(['pat_do_cd', 'pat_age_gr', 'pat_sex'])['allocated_count'].sum()
            target_row_sums = target_margins['row_margins']
            
            common_groups = actual_row_sums.index.intersection(target_row_sums.index)
            row_errors = np.abs(actual_row_sums[common_groups] - target_row_sums[common_groups])
            max_row_error = row_errors.max() if len(row_errors) > 0 else 0
            mean_row_error = row_errors.mean() if len(row_errors) > 0 else 0
            
            # 열 마진 검증 (병원별 총 할당 수)
            actual_col_sums = adjusted_data.groupby('emorg_cd')['allocated_count'].sum()
            target_col_sums = target_margins['col_margins']
            
            common_hospitals = actual_col_sums.index.intersection(target_col_sums.index)
            col_errors = np.abs(actual_col_sums[common_hospitals] - target_col_sums[common_hospitals])
            max_col_error = col_errors.max() if len(col_errors) > 0 else 0
            mean_col_error = col_errors.mean() if len(col_errors) > 0 else 0
            
            # 전체 합 검증
            total_allocated = adjusted_data['allocated_count'].sum()
            total_target = target_margins['row_margins'].sum()
            total_error = abs(total_allocated - total_target)
            
            validation_result = {
                'valid': True,
                'total_allocated': int(total_allocated),
                'total_target': int(total_target),
                'total_error': int(total_error),
                'row_margin_validation': {
                    'max_error': float(max_row_error),
                    'mean_error': float(mean_row_error),
                    'groups_validated': len(common_groups),
                    'error_rate': float(mean_row_error / target_row_sums[common_groups].mean()) if len(common_groups) > 0 else 0
                },
                'col_margin_validation': {
                    'max_error': float(max_col_error),
                    'mean_error': float(mean_col_error),
                    'hospitals_validated': len(common_hospitals),
                    'error_rate': float(mean_col_error / target_col_sums[common_hospitals].mean()) if len(common_hospitals) > 0 else 0
                }
            }
            
            # 검증 결과 로깅
            self.logger.info(
                f"IPF validation: total_error={total_error}, "
                f"row_error={mean_row_error:.2f}, col_error={mean_col_error:.2f}"
            )
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"IPF validation failed: {e}")
            return {'valid': False, 'error': str(e)}
    
    def _get_adjustment_summary(self, original_matrix: pd.DataFrame, 
                              adjusted_matrix: pd.DataFrame) -> Dict[str, Any]:
        """조정 전후 비교 요약"""
        
        try:
            original_total = original_matrix.values.sum()
            adjusted_total = adjusted_matrix.values.sum()
            
            # 변화량 계산
            change_matrix = adjusted_matrix.values - original_matrix.values
            total_change = np.abs(change_matrix).sum()
            max_change = np.abs(change_matrix).max()
            
            # 변화율 계산
            change_rate = total_change / original_total if original_total > 0 else 0
            
            summary = {
                'original_total': float(original_total),
                'adjusted_total': float(adjusted_total),
                'total_change': float(total_change),
                'max_single_change': float(max_change),
                'change_rate': float(change_rate),
                'preservation_rate': 1.0 - change_rate
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate adjustment summary: {e}")
            return {}
    
    def adjust_multiple_dates(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        여러 날짜에 대해 배치 IPF 조정 수행
        
        Args:
            start_date: 시작 날짜 ('YYYYMMDD')
            end_date: 종료 날짜 ('YYYYMMDD')
            
        Returns:
            배치 조정 결과 요약
        """
        
        self.logger.info(f"Starting batch IPF adjustment: {start_date} to {end_date}")
        
        try:
            # 날짜 범위 생성
            date_range = pd.date_range(
                start=pd.to_datetime(start_date, format='%Y%m%d'),
                end=pd.to_datetime(end_date, format='%Y%m%d'),
                freq='D'
            )
            
            batch_results = []
            successful_dates = 0
            failed_dates = 0
            
            for date in date_range:
                date_str = date.strftime('%Y%m%d')
                
                try:
                    result = self.adjust_to_margins(date_str)
                    
                    if result['success']:
                        successful_dates += 1
                    else:
                        failed_dates += 1
                    
                    batch_results.append({
                        'date': date_str,
                        'success': result['success'],
                        'iterations': result.get('iterations', 0),
                        'final_error': result.get('final_error', float('inf'))
                    })
                    
                except Exception as e:
                    self.logger.error(f"IPF adjustment failed for date {date_str}: {e}")
                    failed_dates += 1
                    batch_results.append({
                        'date': date_str,
                        'success': False,
                        'error': str(e)
                    })
            
            # 배치 결과 요약
            summary = {
                'total_dates': len(date_range),
                'successful_dates': successful_dates,
                'failed_dates': failed_dates,
                'success_rate': successful_dates / len(date_range),
                'results': batch_results
            }
            
            if successful_dates > 0:
                successful_results = [r for r in batch_results if r['success']]
                avg_iterations = np.mean([r['iterations'] for r in successful_results])
                avg_error = np.mean([r['final_error'] for r in successful_results])
                
                summary.update({
                    'avg_iterations': avg_iterations,
                    'avg_final_error': avg_error
                })
            
            self.logger.info(
                f"Batch IPF completed: {successful_dates}/{len(date_range)} successful "
                f"({summary['success_rate']:.1%} success rate)"
            )
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Batch IPF adjustment failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_convergence_diagnostics(self) -> Dict[str, Any]:
        """IPF 수렴 진단 정보 반환"""
        
        if not self.convergence_history:
            return {'available': False, 'reason': 'No convergence history'}
        
        history_df = pd.DataFrame(self.convergence_history)
        
        diagnostics = {
            'available': True,
            'iterations': len(self.convergence_history),
            'final_error': history_df['total_error'].iloc[-1],
            'initial_error': history_df['total_error'].iloc[0],
            'error_reduction': history_df['total_error'].iloc[0] - history_df['total_error'].iloc[-1],
            'convergence_rate': self._calculate_convergence_rate(history_df),
            'history': self.convergence_history
        }
        
        return diagnostics
    
    def _calculate_convergence_rate(self, history_df: pd.DataFrame) -> float:
        """수렴 속도 계산 (error reduction per iteration)"""
        
        if len(history_df) < 2:
            return 0.0
        
        error_changes = []
        for i in range(1, len(history_df)):
            prev_error = history_df['total_error'].iloc[i-1]
            curr_error = history_df['total_error'].iloc[i]
            
            if prev_error > 0:
                change_rate = (prev_error - curr_error) / prev_error
                error_changes.append(change_rate)
        
        return np.mean(error_changes) if error_changes else 0.0