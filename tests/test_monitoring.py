"""Unit tests for monitoring modules"""
import pytest
from monitoring.metrics import MetricsCollector


class TestMetricsCollector:
    """Test metrics collection"""
    
    def test_metrics_initialization(self):
        """Test metrics collector initializes correctly"""
        collector = MetricsCollector()
        assert collector.predictions_count == 0
        assert collector.errors_count == 0
    
    def test_record_prediction(self):
        """Test recording predictions"""
        collector = MetricsCollector()
        
        collector.record_prediction(10.5)
        collector.record_prediction(12.3)
        
        assert collector.predictions_count == 2
    
    def test_metrics_summary(self):
        """Test getting metrics summary"""
        collector = MetricsCollector()
        
        collector.record_prediction(10.0)
        collector.record_prediction(20.0)
        collector.record_prediction(30.0)
        
        summary = collector.get_summary()
        
        assert summary["predictions_count"] == 3
        assert summary["latency_stats"]["min_ms"] == 10.0
        assert summary["latency_stats"]["max_ms"] == 30.0
        assert summary["error_rate"] == 0.0
    
    def test_error_recording(self):
        """Test recording errors"""
        collector = MetricsCollector()
        
        collector.record_prediction(10.0)
        collector.record_error()
        
        summary = collector.get_summary()
        assert summary["errors_count"] == 1
        assert summary["error_rate"] == 0.5
