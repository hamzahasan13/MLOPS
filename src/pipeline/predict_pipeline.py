import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import numpy as np

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
            
            data_scaled = preprocessor.transform(features)
            preds = np.round(model.predict(data_scaled),2)

            return (preds)

        except Exception as e:
            raise CustomException(e, sys)
        
    
class CustomData:
    def __init__(self, HorsePower: float, kilometer: float, RiskLevel: str, fuelType: str, vehicleType: str, gearbox: str,
                 Seller: str, NotRepairedDamaged: str, offerType: str):
        
        self.HorsePower = HorsePower
        self.kilometer = kilometer
        self.RiskLevel = RiskLevel
        self.fuelType = fuelType
        self.vehicleType = vehicleType
        self.gearbox = gearbox
        self.Seller = Seller
        self.NotRepairedDamaged = NotRepairedDamaged
        self.offerType = offerType
        
    
    def get_data_as_data_frame(self):
        
        try:
            custom_data_input_dict = {
                "HorsePower": [self.HorsePower],
                "kilometer": [self.kilometer],
                "RiskLevel": [self.RiskLevel],
                "fuelType": [self.fuelType],
                "vehicleType": [self.vehicleType],
                "gearbox": [self.gearbox],
                "Seller": [self.Seller],
                "NotRepairedDamaged": [self.NotRepairedDamaged],
                "offerType": [self.offerType]
                
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise CustomException(e, sys)