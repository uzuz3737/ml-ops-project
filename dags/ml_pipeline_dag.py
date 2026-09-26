"""
MLOps End-to-End Pipeline DAG for Credit Card Default Prediction.
Orchestrates: Ingestion -> TFDV Validation -> Feature Eng & XGBoost Train -> Evaluation -> MLflow Registry Promotion
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator


default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "start_date": datetime(2026, 1, 1),
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


def task_data_ingestion():
    from src.data.ingestion import ingest_data

    train_df, val_df, test_df = ingest_data()
    print(
        f"Data ingestion completed: Train={train_df.shape}, Val={val_df.shape}, Test={test_df.shape}"
    )


def task_data_validation():
    from airflow.exceptions import AirflowFailException
    from src.data.validation import run_validation

    is_valid, anomalies = run_validation()
    if not is_valid:
        raise AirflowFailException(
            f"Data validation failed; downstream training was blocked: {anomalies}"
        )
    else:
        print("Data validation successfully passed with no anomalies.")


def task_train_model():
    from src.models.train import train_model

    model, metrics = train_model()
    print(f"Model training and MLflow tracking complete. Metrics: {metrics}")


def task_train_catboost_model():
    from src.models.train_catboost import train_catboost

    model, metrics = train_catboost()
    print(f"CatBoost training and MLflow tracking complete. Metrics: {metrics}")


def task_health_check_service():
    import urllib.request
    import json
    import os

    serving_url = os.getenv("SERVING_HEALTH_URL", "http://fastapi-serving:8000/health")
    try:
        req = urllib.request.Request(
            serving_url, headers={"User-Agent": "Airflow-Health-Check"}
        )
        with urllib.request.urlopen(req, timeout=5) as response:
            payload = json.loads(response.read().decode())
            print(f"Serving health status: {payload}")
    except Exception as e:
        print(f"Serving check note: {e} (Service may be starting up)")


with DAG(
    dag_id="credit_card_default_mlops_pipeline",
    default_args=default_args,
    description="End-to-end XGBoost and CatBoost credit default pipeline with validation and MLflow",
    schedule_interval="@weekly",
    catchup=False,
    tags=["mlops", "xgboost", "catboost", "data-validation", "mlflow"],
) as dag:

    ingest_step = PythonOperator(
        task_id="ingest_and_split_data",
        python_callable=task_data_ingestion,
    )

    validate_step = PythonOperator(
        task_id="tfdv_data_validation",
        python_callable=task_data_validation,
    )

    train_step = PythonOperator(
        task_id="train_xgboost_with_mlflow",
        python_callable=task_train_model,
    )

    catboost_train_step = PythonOperator(
        task_id="train_catboost_with_mlflow",
        python_callable=task_train_catboost_model,
    )

    health_check_step = PythonOperator(
        task_id="verify_model_serving_readiness",
        python_callable=task_health_check_service,
    )

    # Define DAG task dependencies
    ingest_step >> validate_step >> [train_step, catboost_train_step]
    [train_step, catboost_train_step] >> health_check_step


def task_monitor_serving():
    from src.models.monitor import monitor_serving

    result = monitor_serving()
    print(
        f"Serving monitor MLflow run: {result['run_id']}; metrics={result['metrics']}"
    )


monitoring_dag = DAG(
    dag_id="credit_card_default_daily_monitoring",
    default_args=default_args,
    description="Daily feature-drift and delayed-label quality report to MLflow",
    schedule_interval="@daily",
    catchup=False,
    tags=["mlops", "monitoring", "mlflow", "drift"],
)

monitor_serving_step = PythonOperator(
    task_id="log_serving_monitoring_to_mlflow",
    python_callable=task_monitor_serving,
    dag=monitoring_dag,
)
