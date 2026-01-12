# MLOps Implementation Guide

## 1. CI/CD Pipeline (GitHub Actions)

### What's Implemented
- **Automated Testing**: Runs pytest on every push/PR across Python 3.9, 3.10, 3.11
- **Code Quality**: Linting with flake8, coverage tracking
- **Docker Build**: Automatic Docker image building and pushing on main branch

### Files
- `.github/workflows/ci-cd.yml` - Main CI/CD pipeline
- `.github/workflows/model-validation.yml` - Daily model validation

### How to Use
1. Push to `main` or `develop` branches
2. Tests run automatically
3. On main branch, Docker image is built and pushed to GHCR

**Commands to run locally:**
```bash
pytest tests/ -v --cov=.
flake8 . --count --select=E9,F63,F7,F82
```

---

## 2. Model Serving API (FastAPI)

### What's Implemented
- **Production API**: FastAPI server with Pydantic validation
- **Health Check**: `/health` endpoint monitoring model status
- **Predictions**: `/predict` endpoint for translations
- **Metrics**: `/metrics` endpoint for performance monitoring
- **Model Versioning**: Track model version with predictions

### Files
- `api_server.py` - FastAPI application
- `monitoring/logger.py` - Structured logging
- `monitoring/metrics.py` - Metrics collection

### How to Use

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Start API server:**
```bash
python api_server.py
```

**Test endpoints:**
```bash
# Health check
curl http://localhost:8000/health

# Make prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "hello world"}'

# Get metrics
curl http://localhost:8000/metrics

# API documentation
http://localhost:8000/docs
```

---

## 3. Testing Framework (Pytest)

### What's Implemented
- **Unit Tests**: Model building, data loading, configuration
- **Integration Tests**: API endpoint testing
- **Metrics Tests**: Monitoring module validation
- **Artifact Tests**: Model checkpoint validation

### Files
- `tests/conftest.py` - Pytest fixtures
- `tests/test_model.py` - Model unit tests
- `tests/test_data.py` - Data pipeline tests
- `tests/test_api.py` - API integration tests
- `tests/test_monitoring.py` - Metrics tests
- `tests/test_model_artifacts.py` - Artifact validation

### How to Use

**Run all tests:**
```bash
pytest tests/ -v
```

**Run with coverage:**
```bash
pytest tests/ --cov=. --cov-report=html
```

**Run specific test file:**
```bash
pytest tests/test_model.py -v
```

**Run with parallel execution:**
```bash
pytest tests/ -n auto
```

---

## 4. Monitoring & Observability

### What's Implemented
- **Structured Logging**: JSON-formatted logs for analysis
- **Metrics Collection**: Latency tracking (p50, p95, p99)
- **Data Drift Detection**: Framework for monitoring input changes
- **Error Tracking**: Error rate monitoring

### Files
- `monitoring/logger.py` - Centralized logging
- `monitoring/metrics.py` - Metrics collection

### Features

**Prediction Logging:**
```python
from monitoring.logger import log_prediction

log_prediction(
    input_text="hello",
    output_text="नमस्ते",
    latency_ms=45.2,
    model_version="v1.0"
)
```

**Metrics Tracking:**
```python
from monitoring.metrics import MetricsCollector

metrics = MetricsCollector()
metrics.record_prediction(latency_ms=42.5)
summary = metrics.get_summary()
print(summary)
```

**Log Location:**
```
logs/
  app.log - Application logs
  predictions.log - Prediction logs (auto-created)
  training.log - Training logs (auto-created)
```

---

## Integration with Existing Components

### With Airflow DAG
```python
# In training_pipeline_dag.py
from monitoring.logger import log_training_event

log_training_event("training_started", {"epoch": 1})
```

### With MLflow
```python
# Logs are compatible with MLflow
# Predictions automatically tracked in metrics
```

### With Docker
```bash
# Build with API server
docker build -t transformer-api .

# Run API
docker run -p 8000:8000 transformer-api python api_server.py
```

---

## Next Steps

1. **Enable GitHub Actions**: Push to remote repository
2. **Configure Secrets**: Add GHCR credentials if needed
3. **Set up Monitoring Dashboard**: Use Prometheus + Grafana for `metrics` endpoint
4. **Add Alert Rules**: Configure alerts for error rates, latency spikes
5. **Database Logging**: Consider adding PostgreSQL for long-term log storage

---

## Performance Metrics

### API Latency
- P50: 50th percentile latency
- P95: 95th percentile (SLA target: < 200ms)
- P99: 99th percentile (SLA target: < 500ms)

### Error Rate
- Target: < 0.1% (1 error per 1000 requests)

### Model Availability
- Target: > 99.9% uptime
