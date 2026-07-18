import time
from typing import Dict, Any, List
from contextlib import contextmanager
from custom_logging.logger import app_logger

# Memory cache for session metrics
class MetricsRegistry:
    """Central registry to collect latency and token metrics during execution."""
    _latency_records: Dict[str, List[float]] = {}
    _token_records: Dict[str, Dict[str, int]] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }

    @classmethod
    def record_latency(cls, operation: str, duration_ms: float):
        if operation not in cls._latency_records:
            cls._latency_records[operation] = []
        cls._latency_records[operation].append(duration_ms)
        app_logger.debug(f"Metrics: Operation '{operation}' completed in {duration_ms:.2f} ms")

    @classmethod
    def record_tokens(cls, prompt: int, completion: int):
        cls._token_records["prompt_tokens"] += prompt
        cls._token_records["completion_tokens"] += completion
        cls._token_records["total_tokens"] += (prompt + completion)
        app_logger.debug(f"Metrics: Added {prompt + completion} tokens (Prompt: {prompt}, Completion: {completion})")

    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        avg_latencies = {
            op: float(sum(times) / len(times)) if times else 0.0
            for op, times in cls._latency_records.items()
        }
        return {
            "avg_latency_ms": avg_latencies,
            "raw_latencies_ms": cls._latency_records,
            "tokens": cls._token_records
        }

    @classmethod
    def clear(cls):
        cls._latency_records.clear()
        cls._token_records = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0
        }


@contextmanager
def track_latency(operation: str):
    """Context manager to measure and record execution time."""
    start_time = time.perf_counter()
    try:
        yield
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        MetricsRegistry.record_latency(operation, duration_ms)
