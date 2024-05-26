from src.components.data_cleaning import DataCleaning
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
    

try:
    logging.info("Entered the data cleaning process")
    
    ## Initialize DataCleaning object
    dc_obj = DataCleaning()

    # Load data
    data_path = 'notebook/data/data.csv'
    df = pd.read_csv(data_path)
    logging.info('Read the data into dataframe')
    
    # Clean the data using multiple cleaning functions
    cleaned_df = dc_obj.clean_data(df)
    
    # Specify the path to save the cleaned data
    cleaned_data_path = os.path.join('artifacts', 'cleaned_data.csv')
    os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
    
    # Save cleaned data to a new CSV file in the artifacts folder
    cleaned_df.to_csv(cleaned_data_path, index=False)
    #data_path = 'artifacts/cleaned_data.csv'
    
    dc_obj.initialize_dvc()
    dc_obj.run_dvc_command(f"{data_path}")
    dc_obj.run_dvc_command(f"{cleaned_data_path}")
    
    logging.info('Data cleaning process completed')
    
    di_obj = DataIngestion()
    train_data, test_data = di_obj.initiate_data_ingestion()
    
    dt_obj = DataTransformation()
    train_arr, test_arr, _, input_feature_train_df, input_feature_test_df = dt_obj.initiate_data_transformation(train_data, test_data)

    mt_obj = ModelTrainer()
    print(mt_obj.initiate_model_trainer(train_arr, test_arr, input_feature_train_df, input_feature_test_df))
    
except Exception as e:
    raise CustomException(e, sys)

