# Project Structure

## Overview
```
transformer/
├── src/                           # Source code
│   ├── core/                      # Core ML components
│   │   ├── model.py               # Transformer model architecture
│   │   ├── dataset.py             # Dataset loading & preprocessing
│   │   └── __init__.py
│   ├── web/                       # Web application (Flask)
│   │   ├── app.py                 # Flask app & routes
│   │   └── __init__.py
│   ├── monitoring/                # MLOps monitoring
│   │   ├── logger.py              # Structured logging
│   │   ├── metrics.py             # Metrics collection
│   │   └── __init__.py
│   ├── utils/                     # Utility modules
│   │   └── __init__.py
│   └── __init__.py
│
├── config/                        # Configuration management
│   ├── config.py                  # Main configuration
│   └── __init__.py
│
├── data/                          # Data directory
│   └── tokenizers/                # Tokenizer files
│       ├── tokenizer_en.json
│       └── tokenizer_hi.json
│
├── scripts/                       # Standalone scripts
│   └── train.py                   # Model training script
│
├── models/                        # Trained model checkpoints
├── runs/                          # TensorBoard logs
├── logs/                          # Application logs
│
├── tests/                         # Test suite
│   ├── conftest.py
│   ├── test_model.py
│   ├── test_data.py
│   ├── test_monitoring.py
│   ├── test_model_artifacts.py
│   └── __init__.py
│
├── dags/                          # Airflow DAGs
│   └── training_pipeline_dag.py
│
├── .github/                       # GitHub configuration
│   └── workflows/                 # CI/CD workflows
│       ├── ci-cd.yml
│       └── model-validation.yml
│
├── templates/                     # Flask HTML templates
│   ├── base.html
│   └── index.html
│
├── main.py                        # Application entry point
├── docker-compose.airflow.yml     # Airflow composition
├── Dockerfile                     # Container definition
├── requirements.txt               # Production dependencies
├── requirements_dev.txt           # Development dependencies
├── dvc.yaml                       # DVC pipeline
├── airflow.cfg                    # Airflow configuration
└── README.md                      # Project documentation
```

## Directory Purpose

### src/core
- **model.py**: Transformer model implementation
- **dataset.py**: Data loading, preprocessing, and masking

### src/web
- **app.py**: Flask web application with routes and UI

### src/monitoring
- **logger.py**: Centralized logging system
- **metrics.py**: Performance metrics collection

### config/
- Centralized configuration management
- Updated paths for new structure

### data/
- Tokenizer files in organized subdirectory
- Ready for dataset files

### scripts/
- Standalone training script
- Can be run independently

### tests/
- Comprehensive test suite
- Unit, integration, and artifact tests

### dags/
- Airflow orchestration
- Training pipeline definitions

### logs/
- Application logs (auto-created)
- Monitoring logs

## Import Changes

### Before
```python
from config import get_config
from model import build_transformer
from dataset import BilingualDataset
```

### After
```python
from config.config import get_config
from src.core.model import build_transformer
from src.core.dataset import BilingualDataset
```

## Running the Application

### Start Web App
```bash
python main.py
```

### Run Training
```bash
python scripts/train.py
```

### Run Tests
```bash
pytest tests/ -v
```

### Run Airflow DAG
```bash
airflow dags test training_pipeline_dag
```
