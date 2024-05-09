import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessed.pkl'
        
            numeric_columns = features.select_dtypes(include=['number']).columns
            categorical_columns = features.select_dtypes(include=['object', 'category']).columns
            
            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                     ('scaler', StandardScaler())
                ]
            )
            
            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )
            logging.info("Preprocessing Serving Request Data");
            
            preprocessor_serv = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numeric_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            
            preprocessed_serving = preprocessor_serv.fit_transform(features)
            
            model = load_object(file_path = model_path);
            
            #preprocessor = load_object(file_path=preprocessor_path)
            
            #data_scaled = preprocessor.transform(features)
            preds = model.predict(preprocessed_serving)
            return (preds)

        except Exception as e:
            raise CustomException(e, sys)
        
    
class CustomData:
    def __init__(self, HorsePower: int, kilometer: int, Risk_Level_Low: str, Risk_Level_High: str, fuelType_Diesel: str, 
                 vehicleType_Convertible: str, gearbox_Automatic: str):
        
        self.HorsePower = HorsePower
        self.kilometer = kilometer
        self.Risk_Level_Low = Risk_Level_Low
        self.Risk_Level_High = Risk_Level_High
        self.fuelType_Diesel = fuelType_Diesel
        self.vehicleType_Convertible = vehicleType_Convertible
        self.gearbox_Automatic = gearbox_Automatic
    
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict = {
                "HorsePower": [self.HorsePower],
                "kilometer": [self.kilometer],
                "Risk_Level_Low": [self.Risk_Level_Low],
                "Risk_Level_High": [self.Risk_Level_High],
                "fuelType_Diesel": [self.fuelType_Diesel],
                "vehicleType_Convertible": [self.vehicleType_Convertible],
                "gearbox_Automatic": [self.gearbox_Automatic]
                
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)