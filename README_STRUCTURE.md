# Transformer Translation Project - Reorganized

## 🎯 Project Structure

The project has been reorganized for scalability and maintainability:

```
src/              ← All application code
├── core/         ← ML models and data pipelines
├── web/          ← Flask web application  
├── monitoring/   ← Logging and metrics
└── utils/        ← Helper utilities

config/           ← Configuration management
data/             ← Tokenizers and datasets
scripts/          ← Standalone scripts (training)
tests/            ← Test suite
dags/             ← Airflow orchestration
logs/             ← Application logs
```

## ✨ Key Improvements

- ✅ **Modular Architecture**: Clear separation of concerns
- ✅ **Scalable**: Easy to add new components
- ✅ **Maintainable**: Organized imports and structure
- ✅ **Production-Ready**: Following industry best practices
- ✅ **MLOps-Integrated**: Monitoring, logging, testing

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Web Application
```bash
python main.py
```
Visit: http://localhost:5000

### 3. Run Training
```bash
python scripts/train.py
```

### 4. Run Tests
```bash
pytest tests/ -v
```

### 5. Start Airflow
```bash
airflow webserver --port 8080
airflow scheduler
```

## 📁 File Organization

### Source Code (`src/`)

**src/core/model.py**
- Transformer model architecture
- Encoder and decoder layers

**src/core/dataset.py**
- Bilingual dataset loading
- Data preprocessing and tokenization
- Causal masking for training

**src/web/app.py**
- Flask application
- Web routes for model inference
- Template rendering

**src/monitoring/logger.py**
- Structured JSON logging
- Prediction tracking
- Training event logging

**src/monitoring/metrics.py**
- Performance metrics collection
- Latency tracking (p50, p95, p99)
- Error rate monitoring
- Data drift detection

### Configuration (`config/`)

**config/config.py**
- Centralized configuration
- Model hyperparameters
- Data paths updated for new structure

### Data (`data/`)

**data/tokenizers/**
- Tokenizer files (EN, HI)
- Used during training and inference

### Scripts (`scripts/`)

**scripts/train.py**
- Main training script
- Updated imports for new structure
- Can be run standalone

### Tests (`tests/`)

**test_model.py** - Model architecture tests
**test_data.py** - Data loading and validation
**test_monitoring.py** - Metrics collection tests
**test_model_artifacts.py** - Checkpoint validation

### Airflow DAGs (`dags/`)

**training_pipeline_dag.py**
- Daily model training pipeline
- Data validation and config checks

## 🔧 Configuration

All configuration is in `config/config.py`:

```python
config = {
    "batch_size": 2,
    "num_epochs": 3,
    "lr": 0.0001,
    "seq_length": 128,
    "d_model": 128,
    "nhead": 8,
    "model_folder": "models",
    "tokenizer_file": "data/tokenizers/tokenizer_{lang}.json",
    "experiment_name": "runs/en_hi_model",
}
```

## 📊 Monitoring

Logs are saved to `logs/` directory:
- `app.log` - General application logs
- `predictions.log` - Prediction tracking
- `training.log` - Training events

Access metrics via Flask app or check `src/monitoring/metrics.py`

## 🐳 Docker

### Build
```bash
docker build -t transformer:latest .
```

### Run
```bash
docker run -p 5000:5000 transformer:latest python main.py
```

## 📈 MLOps Features

✅ **CI/CD**: GitHub Actions workflows
✅ **Monitoring**: Structured logging and metrics
✅ **Testing**: Comprehensive test suite
✅ **Experiment Tracking**: MLflow integration
✅ **Data Versioning**: DVC pipeline
✅ **Orchestration**: Airflow DAGs

See [MLOPS_IMPLEMENTATION.md](MLOPS_IMPLEMENTATION.md) for details.

## 📚 Documentation

- [MLOPS_IMPLEMENTATION.md](MLOPS_IMPLEMENTATION.md) - MLOps setup guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Detailed structure overview
- [AIRFLOW_GUIDE.md](AIRFLOW_GUIDE.md) - Airflow configuration
- [DVC_GUIDE.md](DVC_GUIDE.md) - Data versioning guide
- [MLFLOW_GUIDE.md](MLFLOW_GUIDE.md) - Experiment tracking guide

## 🔄 Import Updates

### Old Structure
```python
from config import get_config
from model import build_transformer
from dataset import BilingualDataset
```

### New Structure
```python
from config.config import get_config
from src.core.model import build_transformer
from src.core.dataset import BilingualDataset
```

## 📋 Checklist

- [x] Modular code structure
- [x] Clear separation of concerns
- [x] MLOps infrastructure
- [x] Testing framework
- [x] Documentation
- [ ] Environment variables for secrets
- [ ] Database for logs (optional)
- [ ] Kubernetes deployment (optional)

## 🤝 Contributing

1. Create features in `src/` subdirectories
2. Write tests in `tests/`
3. Update logs in `logs/`
4. Document changes

## 📝 License

Project license here
