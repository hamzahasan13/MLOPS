import pytest
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.pipeline.predict_pipeline import PredictPipeline, CustomData

import pandas as pd
from src.utils import load_object

@pytest.fixture
def sample_custom_data():
    # Create a sample CustomData object for testing
    return CustomData(HorsePower=600,
                      kilometer=50000,
                      RiskLevel="High",
                      fuelType="Gasoline",
                      vehicleType="SUV",
                      gearbox="Automatic",
                      Seller="Private",
                      NotRepairedDamaged="Yes",
                      offerType="Customer Offer")

def test_data_ingestion():
    # Test the data ingestion process
    data_ingestion = DataIngestion()
    train_data, test_data = data_ingestion.initiate_data_ingestion()
    assert isinstance(train_data, str)
    assert isinstance(test_data, str)

def test_data_transformation():
    # Test the data transformation process
    data_transformation = DataTransformation()
    train_path = 'artifacts/train.csv'
    test_path = 'artifacts/test.csv'
    
    global input_feature_train_df
    global input_feature_test_df
    
    train_arr, test_arr, _, input_feature_train_df, input_feature_test_df = data_transformation.initiate_data_transformation(train_path, test_path)
    assert train_arr.shape[0] > 0
    assert test_arr.shape[0] > 0


def test_model_trainer():
    # Test the model training process
    model_trainer = ModelTrainer()
    
    preprocessor_path = 'artifacts/preprocessor.pkl'
    preprocessor = load_object(file_path=preprocessor_path)
    
    numerical_columns = ['HorsePower', 'kilometer']
    categorical_columns = ['Seller', 'offerType', 'vehicleType', 'gearbox', 'fuelType', 'NotRepairedDamaged','RiskLevel']
    
    train_data = [[150, 50000, "High", "Gasoline", "SUV", "Automatic", "Private", "Yes", "Customer Offer"],
                  [200, 80000, "Low", "Diesel", "Compact Car", "Manual", "Private", "No", "Customer Offer"]]
    test_data = [[180, 60000, "Low", "Diesel", "Bus", "Automatic", "Private", "Yes", "Customer Offer"],
                 [220, 70000, "High", "Gasoline", "SUV", "Automatic", "Private", "No", "Customer Offer"]]

    train_df = pd.DataFrame(train_data, columns=["HorsePower", "kilometer", "RiskLevel", "fuelType", "vehicleType",
                                                 "gearbox", "Seller", "NotRepairedDamaged", "offerType"])
    test_df = pd.DataFrame(test_data, columns=["HorsePower", "kilometer", "RiskLevel", "fuelType", "vehicleType",
                                               "gearbox", "Seller", "NotRepairedDamaged", "offerType"])
    
    input_feature_train_arr = preprocessor.transform(train_df)
    input_feature_test_arr = preprocessor.transform(test_df)
    
    transformed_columns = (
            preprocessor.named_transformers_['num_pipeline'].named_steps['scaler'].get_feature_names_out(numerical_columns).tolist()
            + preprocessor.named_transformers_['cat_pipeline'].named_steps['one_hot_encoder'].get_feature_names_out(categorical_columns).tolist()
            )

    dense_input_feature_train_arr = input_feature_train_arr
    dense_input_feature_test_arr = input_feature_test_arr
            
    # Create DataFrame using dense matrix and column names
    input_feature_train_arr = pd.DataFrame(dense_input_feature_train_arr, columns=transformed_columns)
    input_feature_test_arr = pd.DataFrame(dense_input_feature_test_arr, columns=transformed_columns)

    r2_score = model_trainer.initiate_model_trainer(input_feature_train_arr, input_feature_test_arr, input_feature_train_df, input_feature_test_df)
    assert isinstance(r2_score, float)

