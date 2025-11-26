"""
Logging configuration for the validation system.

This module provides:
- Centralized logging configuration
- Structured logging with JSON formatting
- Log rotation and management
- Performance logging integration
"""

import logging
import logging.handlers
import json
import sys
from typing import Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import threading


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging"""

    def format(self, record):
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'thread': record.thread,
            'thread_name': getattr(record, 'threadName', 'unknown')
        }

        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)

        # Add any extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'message']:
                log_entry[key] = value

        return json.dumps(log_entry)


class ValidationFilter(logging.Filter):
    """Filter for validation-related log messages"""

    def filter(self, record):
        return 'validator' in record.name.lower()


def setup_logging(config_path: Optional[str] = None, log_level: str = 'INFO',
                 log_to_file: bool = True, log_dir: str = 'logs') -> logging.Logger:
    """
    Setup logging configuration for the validation system

    Args:
        config_path: Optional path to logging configuration file
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_to_file: Whether to log to files
        log_dir: Directory for log files

    Returns:
        Root logger instance
    """

    # Create log directory if it doesn't exist
    if log_to_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

    # Get root logger
    logger = logging.getLogger('validator')
    logger.setLevel(getattr(logging, log_level.upper()))

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler with structured output
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, log_level.upper()))

    # Console formatter (human-readable)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)

    # Add validation filter to console handler
    validation_filter = ValidationFilter()
    console_handler.addFilter(validation_filter)

    logger.addHandler(console_handler)

    # File handler for all logs (if enabled)
    if log_to_file:
        # Main log file with rotation
        main_log_file = log_path / 'validation.log'
        file_handler = logging.handlers.RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # Always log debug to file

        # File formatter (human-readable)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # JSON log file for structured logging
        json_log_file = log_path / 'validation_structured.json'
        json_handler = logging.handlers.RotatingFileHandler(
            json_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=3
        )
        json_handler.setLevel(logging.INFO)

        json_formatter = JsonFormatter()
        json_handler.setFormatter(json_formatter)
        logger.addHandler(json_handler)

        # Error log file
        error_log_file = log_path / 'validation_errors.log'
        error_handler = logging.handlers.RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)

        error_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n'
            'Exception: %(exc_text)s\n',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        error_handler.setFormatter(error_formatter)
        logger.addHandler(error_handler)

    # Set specific levels for noisy loggers
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('duckdb').setLevel(logging.WARNING)

    # Add performance logging integration
    performance_logger = logging.getLogger('validator.performance')
    performance_handler = logging.StreamHandler(sys.stdout)
    performance_handler.setLevel(logging.INFO)
    performance_handler.setFormatter(console_formatter)
    performance_handler.addFilter(lambda record: 'performance' in record.getMessage().lower())
    performance_logger.addHandler(performance_handler)

    logger.info(f"Logging configured with level {log_level} (file logging: {log_to_file})")

    return logger


def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls

    Args:
        logger: Logger instance to use

    Example:
        @log_function_call(logger)
        def my_function():
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Entering function: {func_name}")

            try:
                start_time = datetime.now()
                result = func(*args, **kwargs)
                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                logger.debug(f"Function {func_name} completed in {duration:.3f}s")
                return result

            except Exception as e:
                logger.error(f"Function {func_name} failed: {e}", exc_info=True)
                raise

        return wrapper
    return decorator


def log_performance(logger: logging.Logger, operation_name: str):
    """
    Context manager for logging performance metrics

    Args:
        logger: Logger instance
        operation_name: Name of the operation

    Example:
        with log_performance(logger, "database_query"):
            # Your code here
            pass
    """
    class PerformanceContext:
        def __init__(self, logger, operation_name):
            self.logger = logger
            self.operation_name = operation_name
            self.start_time = None

        def __enter__(self):
            self.start_time = datetime.now()
            self.logger.debug(f"Starting operation: {self.operation_name}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                end_time = datetime.now()
                duration = (end_time - self.start_time).total_seconds()
                self.logger.info(f"Operation {self.operation_name} completed in {duration:.3f}s")

    return PerformanceContext(logger, operation_name)


def add_context_info(record: logging.LogRecord, **context):
    """
    Add context information to log records

    Args:
        record: Log record to modify
        **context: Context key-value pairs to add
    """
    for key, value in context.items():
        setattr(record, key, value)


class ValidationLoggerAdapter(logging.LoggerAdapter):
    """
    Logger adapter that adds validation context to log messages
    """

    def __init__(self, logger: logging.Logger, validation_id: Optional[str] = None):
        super().__init__(logger, {})
        self.validation_id = validation_id

    def process(self, msg, kwargs):
        if self.validation_id:
            kwargs['extra'] = kwargs.get('extra', {})
            kwargs['extra']['validation_id'] = self.validation_id
        return msg, kwargs


def get_logger(name: str, validation_id: Optional[str] = None) -> logging.Logger:
    """
    Get logger with optional validation context

    Args:
        name: Logger name
        validation_id: Optional validation ID for context

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    if validation_id:
        return ValidationLoggerAdapter(logger, validation_id)

    return logger


# Global logging lock for thread safety
_logging_lock = threading.Lock()


def setup_global_logging(log_level: str = 'INFO'):
    """
    Setup global logging configuration

    Args:
        log_level: Global logging level
    """
    with _logging_lock:
        if not logging.getLogger('validator').handlers:
            setup_logging(log_level=log_level, log_to_file=True)
