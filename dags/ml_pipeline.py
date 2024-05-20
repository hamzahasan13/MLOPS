from __future__ import annotations
import json
import os
import sys
import subprocess

from textwrap import dedent
import pendulum
import pandas as pd
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator

from src.training_pipeline import TrainPipeline

## Pipeline object
tp_obj = TrainPipeline()

with DAG(
    dag_id = 'Price_Predictor',
    default_args = {'retries': 2},
    description = "Training-Pipeline",
    schedule = "@daily",
    start_date = pendulum.datetime(2024, 5, 16, tz = "UTC"),
    catchup = False,
    tags = ['MLOps', 'Prediction'],
    
) as dag:
    dag.doc_md = __doc__
    
    def initialize_dvc_command():
        return '''
        if [ ! -d .dvc ]; then
            dvc init
            echo "DVC initialized successfully."
        else
            echo ".dvc already exists. DVC Init is skipped."
        fi
        '''
    
    initialize_dvc_task = BashOperator(
        task_id='initialize_dvc',
        bash_command=initialize_dvc_command()
    )
    
    def data_cleaning(**kwargs):
        ## Task Instance
        ## Cross Communication: xcom
        #ti = kwargs['ti']
        tp_obj.init_data_cleaning()
        #ti.xcom_push('data_cleaning_artifact')
    
    def data_ingestion(**kwargs):
        ti = kwargs['ti']
        #data_cleaning_artifact = ti.xcom_pull(task_ids = 'data_cleaning', key = 'data_cleaning_artifact')
        train_data_path, test_data_path = tp_obj.init_data_ingestion()
        ti.xcom_push('data_ingestion_artifact', {"train_data_path":train_data_path, "test_data_path": test_data_path})
        
    def data_transformation(**kwargs):
        ti = kwargs['ti']
        data_ingestion_artifact = ti.xcom_pull(task_ids = 'data_ingestion', key = 'data_ingestion_artifact')
        train_arr, test_arr, input_feature_train_df, input_feature_test_df = tp_obj.init_data_transformation(data_ingestion_artifact['train_data_path'], data_ingestion_artifact['test_data_path'])
        ti.xcom_push("data_transformation_artifact", {'train_arr':train_arr, 'test_arr': test_arr, 
                            'input_feature_train_df': input_feature_train_df , 'input_feature_test_df': input_feature_test_df})
    
    def model_trainer(**kwargs):
        ti = kwargs['ti']
        data_transformation_artifact = ti.xcom_pull(task_ids = 'data_transformation', key = 'data_transformation_artifact')
        tp_obj.init_model_trainer(data_transformation_artifact['train_arr'], data_transformation_artifact['test_arr'],
                data_transformation_artifact['input_feature_train_df'], data_transformation_artifact['input_feature_test_df'])
    
    data_cleaning_task = PythonOperator(
        task_id = 'data_cleaning',
        python_callable = data_cleaning,
    )
    data_cleaning.doc_md = dedent(
        """
        This file performs data cleaning
        """
    )
    
    data_ingestion_task = PythonOperator(
        task_id = 'data_ingestion',
        python_callable = data_ingestion,
    )
    data_ingestion.doc_md = dedent(
        """
        This file creates training and testing datasets.
        """
    )
    
    data_transformation_task = PythonOperator(
        task_id = 'data_transformation',
        python_callable = data_transformation,
    )
    data_transformation.doc_md = dedent(
        """
        Data Transformation
        """
    )
    
    model_trainer_task = PythonOperator(
        task_id = 'model_trainer',
        python_callable = model_trainer,
    )
    model_trainer.doc_md = dedent(
        """
        Trains ML models and find the best model
        """
    )
    
initialize_dvc_task >> data_cleaning_task >> data_ingestion_task >> data_transformation_task >> model_trainer_task