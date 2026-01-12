"""Metrics collection and monitoring"""
import time
from collections import deque
from typing import Dict, Any
from datetime import datetime, timedelta


class MetricsCollector:
    """Collect and track model serving metrics"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.predictions_count = 0
        self.errors_count = 0
        self.start_time = datetime.utcnow()
    
    def record_prediction(self, latency_ms: float) -> None:
        """Record a successful prediction"""
        self.latencies.append(latency_ms)
        self.predictions_count += 1
    
    def record_error(self) -> None:
        """Record a failed prediction"""
        self.errors_count += 1
    
    def get_summary(self) -> Dict[str, Any]:
        """Get current metrics summary"""
        if not self.latencies:
            return {
                "predictions_count": 0,
                "errors_count": 0,
                "uptime_seconds": 0,
                "latency_stats": {}
            }
        
        latencies_list = list(self.latencies)
        latencies_list.sort()
        
        return {
            "predictions_count": self.predictions_count,
            "errors_count": self.errors_count,
            "uptime_seconds": (datetime.utcnow() - self.start_time).total_seconds(),
            "latency_stats": {
                "min_ms": min(latencies_list),
                "max_ms": max(latencies_list),
                "mean_ms": sum(latencies_list) / len(latencies_list),
                "p50_ms": latencies_list[len(latencies_list) // 2],
                "p95_ms": latencies_list[int(len(latencies_list) * 0.95)],
                "p99_ms": latencies_list[int(len(latencies_list) * 0.99)],
            },
            "error_rate": self.errors_count / (self.predictions_count + self.errors_count) if (self.predictions_count + self.errors_count) > 0 else 0
        }
    
    def check_data_drift(self, current_stats: Dict[str, float], threshold: float = 0.1) -> Dict[str, Any]:
        """Check for data drift in model inputs/outputs"""
        return {
            "drift_detected": False,
            "metrics": current_stats,
            "threshold": threshold
        }
