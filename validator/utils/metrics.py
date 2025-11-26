"""
Performance tracking and metrics for the validation system.

This module provides:
- Performance monitoring and profiling
- Metric collection and aggregation
- Performance visualization
- Bottleneck identification
"""

import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import logging
from contextlib import contextmanager


class PerformanceTracker:
    """Performance tracking and metrics collection"""

    def __init__(self, max_history: int = 1000):
        """
        Initialize performance tracker

        Args:
            max_history: Maximum number of performance records to keep
        """
        self.logger = logging.getLogger(__name__)

        # Performance metrics storage
        self._metrics = defaultdict(list)
        self._metrics_lock = threading.Lock()

        # Query performance tracking
        self._query_times = deque(maxlen=max_history)
        self._query_lock = threading.Lock()

        # Memory usage tracking
        self._memory_usage = deque(maxlen=100)
        self._memory_lock = threading.Lock()

        # System resource tracking
        self._cpu_usage = deque(maxlen=100)
        self._cpu_lock = threading.Lock()

        # Active timers
        self._active_timers: Dict[str, datetime] = {}
        self._timer_lock = threading.Lock()

        # Start background monitoring
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(target=self._background_monitoring, daemon=True)
        self._monitoring_thread.start()

    def start_timer(self, operation_name: str) -> str:
        """
        Start timing an operation

        Args:
            operation_name: Name of the operation being timed

        Returns:
            Timer ID for stopping the timer
        """
        timer_id = f"{operation_name}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        with self._timer_lock:
            self._active_timers[timer_id] = datetime.now()

        return timer_id

    def stop_timer(self, timer_id: str) -> Optional[float]:
        """
        Stop timing an operation

        Args:
            timer_id: Timer ID returned by start_timer

        Returns:
            Duration in seconds, or None if timer not found
        """
        with self._timer_lock:
            if timer_id not in self._active_timers:
                return None

            start_time = self._active_timers[timer_id]
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            del self._active_timers[timer_id]

        # Record the metric
        self.record_metric('operation_duration', duration, {'operation': timer_id.split('_')[0]})

        return duration

    def record_metric(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """
        Record a performance metric

        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional tags for the metric
        """
        timestamp = datetime.now()

        with self._metrics_lock:
            self._metrics[metric_name].append({
                'timestamp': timestamp,
                'value': value,
                'tags': tags or {}
            })

    def record_query_time(self, query: str, duration: float, success: bool = True):
        """
        Record query execution time

        Args:
            query: SQL query (truncated for storage)
            duration: Query execution time in seconds
            success: Whether the query was successful
        """
        query_key = query[:100] + "..." if len(query) > 100 else query

        with self._query_lock:
            self._query_times.append({
                'query': query_key,
                'duration': duration,
                'success': success,
                'timestamp': datetime.now()
            })

    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()

            with self._memory_lock:
                self._memory_usage.append({
                    'rss': memory_info.rss,  # Resident Set Size
                    'vms': memory_info.vms,  # Virtual Memory Size
                    'timestamp': datetime.now()
                })

        except Exception as e:
            self.logger.warning(f"Failed to record memory usage: {e}")

    def record_cpu_usage(self):
        """Record current CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)

            with self._cpu_lock:
                self._cpu_usage.append({
                    'cpu_percent': cpu_percent,
                    'timestamp': datetime.now()
                })

        except Exception as e:
            self.logger.warning(f"Failed to record CPU usage: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive performance statistics

        Returns:
            Dictionary with performance statistics
        """
        stats = {}

        # Operation duration statistics
        with self._metrics_lock:
            for metric_name, records in self._metrics.items():
                if records:
                    values = [r['value'] for r in records]
                    stats[f"{metric_name}_count"] = len(values)
                    stats[f"{metric_name}_avg"] = sum(values) / len(values)
                    stats[f"{metric_name}_min"] = min(values)
                    stats[f"{metric_name}_max"] = max(values)

        # Query performance statistics
        with self._query_lock:
            if self._query_times:
                query_durations = [q['duration'] for q in self._query_times]
                stats['query_count'] = len(self._query_times)
                stats['query_avg_duration'] = sum(query_durations) / len(query_durations)
                stats['query_min_duration'] = min(query_durations)
                stats['query_max_duration'] = max(query_durations)
                stats['query_success_rate'] = sum(1 for q in self._query_times if q['success']) / len(self._query_times)

        # Memory statistics
        with self._memory_lock:
            if self._memory_usage:
                memory_values = [m['rss'] for m in self._memory_usage]
                stats['memory_avg_mb'] = sum(memory_values) / len(memory_values) / (1024 * 1024)
                stats['memory_max_mb'] = max(memory_values) / (1024 * 1024)
                stats['memory_min_mb'] = min(memory_values) / (1024 * 1024)

        # CPU statistics
        with self._cpu_lock:
            if self._cpu_usage:
                cpu_values = [c['cpu_percent'] for c in self._cpu_usage]
                stats['cpu_avg_percent'] = sum(cpu_values) / len(cpu_values)
                stats['cpu_max_percent'] = max(cpu_values)

        # Active timer count
        with self._timer_lock:
            stats['active_timers'] = len(self._active_timers)

        return stats

    def get_recent_metrics(self, metric_name: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent metrics for a specific metric name

        Args:
            metric_name: Name of the metric
            limit: Maximum number of records to return

        Returns:
            List of recent metric records
        """
        with self._metrics_lock:
            if metric_name not in self._metrics:
                return []

            recent_metrics = self._metrics[metric_name][-limit:]
            return recent_metrics.copy()

    def get_slow_queries(self, threshold: float = 1.0, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get queries that took longer than threshold

        Args:
            threshold: Duration threshold in seconds
            limit: Maximum number of slow queries to return

        Returns:
            List of slow query records
        """
        with self._query_lock:
            slow_queries = [q for q in self._query_times if q['duration'] > threshold]
            slow_queries = sorted(slow_queries, key=lambda x: x['duration'], reverse=True)
            return slow_queries[:limit]

    def get_performance_summary(self, time_window: timedelta = None) -> Dict[str, Any]:
        """
        Get performance summary for a time window

        Args:
            time_window: Time window to analyze (default: all time)

        Returns:
            Dictionary with performance summary
        """
        cutoff_time = datetime.now() - time_window if time_window else None

        # Filter recent metrics
        recent_metrics = {}
        with self._metrics_lock:
            for metric_name, records in self._metrics.items():
                if cutoff_time:
                    filtered_records = [r for r in records if r['timestamp'] > cutoff_time]
                else:
                    filtered_records = records

                if filtered_records:
                    values = [r['value'] for r in filtered_records]
                    recent_metrics[metric_name] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'latest': values[-1] if values else 0
                    }

        return {
            'time_window': str(time_window) if time_window else 'all_time',
            'metrics': recent_metrics,
            'generated_at': datetime.now().isoformat()
        }

    def _background_monitoring(self):
        """Background monitoring thread"""
        while self._monitoring_active:
            try:
                # Record system metrics every 30 seconds
                self.record_memory_usage()
                self.record_cpu_usage()

                time.sleep(30)

            except Exception as e:
                self.logger.error(f"Background monitoring failed: {e}")
                time.sleep(60)  # Wait longer on error

    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring_active = False
        if hasattr(self, '_monitoring_thread'):
            self._monitoring_thread.join(timeout=5)

    def reset_metrics(self):
        """Reset all performance metrics"""
        with self._metrics_lock:
            self._metrics.clear()

        with self._query_lock:
            self._query_times.clear()

        with self._memory_lock:
            self._memory_usage.clear()

        with self._cpu_lock:
            self._cpu_usage.clear()

        self.logger.info("Performance metrics reset")

    def export_metrics(self, format: str = 'json') -> str:
        """
        Export metrics in specified format

        Args:
            format: Export format ('json' or 'dict')

        Returns:
            Metrics in requested format
        """
        stats = self.get_stats()

        if format == 'json':
            import json
            return json.dumps(stats, indent=2, default=str)
        else:
            return stats

    @contextmanager
    def time_operation(self, operation_name: str):
        """
        Context manager for timing operations

        Args:
            operation_name: Name of the operation

        Example:
            with performance_tracker.time_operation("data_loading"):
                # Your code here
                pass
        """
        timer_id = self.start_timer(operation_name)
        try:
            yield timer_id
        finally:
            self.stop_timer(timer_id)


# Global performance tracker instance
_default_performance_tracker = PerformanceTracker()


def get_performance_tracker() -> PerformanceTracker:
    """Get global performance tracker instance"""
    return _default_performance_tracker


def time_operation(operation_name: str):
    """
    Decorator for timing function execution

    Args:
        operation_name: Name of the operation

    Example:
        @time_operation("database_query")
        def my_query_function():
            # Your code here
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            tracker = get_performance_tracker()
            timer_id = tracker.start_timer(operation_name)

            try:
                result = func(*args, **kwargs)
                return result
            finally:
                tracker.stop_timer(timer_id)

        return wrapper
    return decorator
