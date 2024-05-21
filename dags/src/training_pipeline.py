from src.data_cleaning import DataCleaning
from src.data_ingestion import DataIngestion
from src.data_transformation import DataTransformation
from src.model_trainer import ModelTrainer
from src.logger import logging
from src.exception import CustomException
import sys
import os
import pandas as pd

class TrainPipeline:
    
    def init_data_cleaning(self):
        try:
            logging.info("Entered the data cleaning process")
            
            ## Initialize DataCleaning object
            dc_obj = DataCleaning()

            # Load data
            data_path = 'dags/data_files/data.csv'
            df = pd.read_csv(data_path)
            logging.info('Read the data into dataframe')
            
            # Clean the data using multiple cleaning functions
            cleaned_df = dc_obj.clean_data(df)

            cleaned_df = cleaned_df.sample(n=15300, random_state=42)
            # Specify the path to save the cleaned data
            cleaned_data_path = os.path.join('data_files', 'cleaned_data.csv')
            os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
            
            # Save cleaned data to a new CSV file in the artifacts folder
            cleaned_df.to_csv(cleaned_data_path, index=False)
            #data_path = 'artifacts/cleaned_data.csv'
            """
            dc_obj.initialize_dvc()
            dc_obj.run_dvc_command(f"{data_path}")
            dc_obj.run_dvc_command(f"{cleaned_data_path}")
            """
    
            logging.info('Data cleaning process completed')
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def init_data_ingestion(self):
        try:
            
            di_obj = DataIngestion()
            train_data, test_data = di_obj.initiate_data_ingestion()
        
            return (train_data, test_data)
        
        except Exception as e:
            raise CustomException(e, sys)

    def init_data_transformation(self, train_data_path, test_data_path):
        try:
            dt_obj = DataTransformation()
            train_arr, test_arr, _, input_feature_train_df, input_feature_test_df = dt_obj.initiate_data_transformation(train_data_path, test_data_path)
            
            return (train_arr, test_arr, input_feature_train_df, input_feature_test_df)
            
        except Exception as e:
            raise CustomException(e, sys)

    def init_model_trainer(self, train_arr, test_arr, input_feature_train_df, input_feature_test_df):
        try:
            
            mt_obj = ModelTrainer()
            print(mt_obj.initiate_model_trainer(train_arr, test_arr, input_feature_train_df, input_feature_test_df))
            
        except Exception as e:
            raise CustomException(e, sys)

    
    def starting_pipeline(self):
        try:
            self.init_data_cleaning()
            train_data_path, test_data_path = self.init_data_ingestion()
            train_arr, test_arr, _, input_feature_train_df, input_feature_test_df = self.init_data_transformation(train_data_path, test_data_path)
            self.init_model_trainer(train_arr, test_arr, input_feature_train_df, input_feature_test_df)
        
        except Exception as e:
            raise CustomException(e, sys)
