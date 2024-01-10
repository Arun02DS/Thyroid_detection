from asyncio import tasks
import json
from textwrap import dedent
import pendulum
import os
from airflow import DAG 
from airflow.operators.python import PythonOperator

with DAG(
    'thyroid_training',
    default_args={'retries': 2},
    description='thyroid disease detection',
    schedule_interval="@weekly",
    start_date=pendulum.datetime(2024,9,1,tz="UTC"),
    catchup=False,
    tags=['training'],
) as dag:
    
    def training(**kwargs):
        from src.pipeline.training_pipeline import start_training_pipeline
        start_training_pipeline()
    
    def sync_artifacts_to_s3_bucket(**kwargs):
        bucket_name=os.getenv("BUCKET_NAME")
        os.system(f"aws s3 sync /app/artifact s3://{bucket_name}/artifact")
        os.system(f"aws s3 sync /app/saved_models s3://{bucket_name}/saved_models")

    training_pipeline=PythonOperator(
        task_id="train_pipeline",
        python_callable=training
    )

    sync_data_to_s3=PythonOperator(
        task_id="sync_data_to_s3",
        python_callable=sync_artifacts_to_s3_bucket
    )

    training_pipeline >> sync_data_to_s3

