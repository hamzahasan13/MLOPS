import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            model_path = 'artifacts/model.pkl'
            preprocessor_path = 'artifacts/preprocessed.pkl'
            model = load_object(file_path = model_path);
            print('H1')
            preprocessor = load_object(file_path=preprocessor_path)
            print(features)
            
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return (preds)

        except Exception as e:
            raise CustomException(e, sys)
        
    
class CustomData:
    def __init__(self, HorsePower: str, kilometer: str, Risk_Level_Low: str, Risk_Level_High: str, fuelType_Diesel: str, 
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