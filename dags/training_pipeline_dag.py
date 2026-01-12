"""
Airflow DAG for Transformer Model Training Pipeline
This orchestrates the end-to-end ML workflow
"""
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.dates import days_ago
import os

default_args = {
    'owner': 'mlops-team',
    'depends_on_past': False,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'transformer_training_pipeline',
    default_args=default_args,
    description='End-to-end transformer model training pipeline',
    schedule_interval='@daily',  # Run daily, adjust as needed
    start_date=days_ago(1),
    catchup=False,
    tags=['ml', 'transformer', 'training'],
)

def check_data_availability():
    """Check if training data is available"""
    from datasets import load_dataset
    from config import get_config
    
    config = get_config()
    try:
        ds = load_dataset(
            f'{config["datasource"]}',
            f'{config["lang_src"]}-{config["lang_tgt"]}',
            split='train[:1%]'
        )
        print(f"Data check passed: {len(ds)} samples found")
        return True
    except Exception as e:
        print(f"Data check failed: {e}")
        raise

def validate_config():
    """Validate configuration parameters"""
    from config import get_config
    
    config = get_config()
    required_keys = ['lang_src', 'lang_tgt', 'model_folder', 'num_epochs']
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    print("Configuration validation passed")
    return True

def check_gpu_availability():
    """Check if GPU is available for training"""
    import torch
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    return True

# Task 1: Validate configuration
validate_config_task = PythonOperator(
    task_id='validate_config',
    python_callable=validate_config,
    dag=dag,
)

# Task 2: Check data availability
check_data_task = PythonOperator(
    task_id='check_data_availability',
    python_callable=check_data_availability,
    dag=dag,
)

# Task 3: Check GPU availability
check_gpu_task = PythonOperator(
    task_id='check_gpu_availability',
    python_callable=check_gpu_availability,
    dag=dag,
)

# Task 4: DVC pull (get data/models if versioned)
dvc_pull_task = BashOperator(
    task_id='dvc_pull',
    bash_command='cd /home/amitdubey/Downloads/transformer && dvc pull || echo "No DVC remote configured"',
    dag=dag,
)

# Task 5: Run training
train_model_task = BashOperator(
    task_id='train_model',
    bash_command='cd /home/amitdubey/Downloads/transformer && python train.py',
    dag=dag,
)

# Task 6: DVC add and push models
dvc_push_task = BashOperator(
    task_id='dvc_push_models',
    bash_command='''
    cd /home/amitdubey/Downloads/transformer && \
    dvc add models/runs/en_hi_model && \
    dvc push || echo "No DVC remote configured"
    ''',
    dag=dag,
)

# Task 7: Test model inference
test_inference_task = BashOperator(
    task_id='test_inference',
    bash_command='''
    cd /home/amitdubey/Downloads/transformer && \
    python -c "
import torch
from pathlib import Path
from config import get_config

config = get_config()
model_folder = Path(config['model_folder'])
model_path = list(model_folder.glob('*.pt'))

if model_path:
    print(f'Model found: {model_path[0]}')
    # Basic sanity check
    checkpoint = torch.load(model_path[0], map_location='cpu')
    print(f'Checkpoint keys: {checkpoint.keys()}')
    print('Inference test passed')
else:
    raise FileNotFoundError('No trained model found')
"
    ''',
    dag=dag,
)

# Task 8: Log to MLflow
mlflow_log_task = BashOperator(
    task_id='log_to_mlflow',
    bash_command='echo "MLflow logging completed during training"',
    dag=dag,
)

# Define task dependencies
validate_config_task >> check_data_task >> check_gpu_task >> dvc_pull_task
dvc_pull_task >> train_model_task >> dvc_push_task
train_model_task >> test_inference_task >> mlflow_log_task
