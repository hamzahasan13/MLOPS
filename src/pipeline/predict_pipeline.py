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
            preprocessor_path = 'artifacts/preprocessor.pkl'

            model = load_object(file_path = model_path);
            preprocessor = load_object(file_path=preprocessor_path)
            
            original_columns = set(feature.split('_', 1)[0] for feature in features)
            print('X0')
            print(features)
            print('X0')
            print(original_columns)
            print('X1')
            data_scaled = preprocessor.transform(features[list[original_columns]])
            preds = model.predict(data_scaled)
            return (preds)

        except Exception as e:
            raise CustomException(e, sys)
        
    
class CustomData:
    def __init__(self, HorsePower: int, kilometer: int, RiskLevel_Low: str, RiskLevel_High: str, fuelType_Diesel: str, 
                 vehicleType_Convertible: str, gearbox_Automatic: str):
        
        self.HorsePower = HorsePower
        self.kilometer = kilometer
        self.RiskLevel_Low = RiskLevel_Low
        self.RiskLevel_High = RiskLevel_High
        self.fuelType_Diesel = fuelType_Diesel
        self.vehicleType_Convertible = vehicleType_Convertible
        self.gearbox_Automatic = gearbox_Automatic
    
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict = {
                "HorsePower": [self.HorsePower],
                "kilometer": [self.kilometer],
                "RiskLevel_Low": [self.RiskLevel_Low],
                "RiskLevel_High": [self.RiskLevel_High],
                "fuelType_Diesel": [self.fuelType_Diesel],
                "vehicleType_Convertible": [self.vehicleType_Convertible],
                "gearbox_Automatic": [self.gearbox_Automatic]
                
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)