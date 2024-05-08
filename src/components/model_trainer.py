import os
import sys
import pandas as pd
from dataclasses import dataclass
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models
    
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self, train_array, test_array):
        try:
            

            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array.iloc[:, :-1],
                train_array.iloc[:, -1],
                test_array.iloc[:, :-1],
                test_array.iloc[:, -1]
            )
            
            models = {
                'Linear Regression': LinearRegression(),
                'Elastic Net': ElasticNet(),
                'Decision Tree Regressor': DecisionTreeRegressor(),
                'RandomForestRegressor': RandomForestRegressor(),
                'GradientBoostingRegressor': GradientBoostingRegressor(),
                'ExtraTreesRegressor': ExtraTreesRegressor(),

            }
            """
            hyperparameters = {
                'Linear Regression': {},  # Linear Regression doesn't have any hyperparameters
                'Elastic Net': {
                    'alpha': [0.001, 0.01, 0.1, 1, 10, 100],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'max_iter': [100, 1000, 10000],
                    'tol': [0.0001, 0.001, 0.01, 0.1]
                },
                'Decision Tree Regressor': {
                    'criterion': ['mse', 'friedman_mse', 'mae'],
                    'splitter': ['best', 'random'],
                    'max_depth': [None, 10, 50, 100, 200],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                'RandomForestRegressor': {
                    'n_estimators': [10, 50, 100, 200],
                    'criterion': ['mse', 'mae'],
                    'max_depth': [None, 10, 50, 100, 200],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                },
                'GradientBoostingRegressor': {
                    'loss': ['ls', 'lad', 'huber', 'quantile'],
                    'learning_rate': [0.001, 0.01, 0.1],
                    'n_estimators': [10, 50, 100, 200],
                    'subsample': [0.5, 0.75, 1.0],
                    'max_depth': [3, 4, 5, 6],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'alpha': [0.9, 0.99]
                },
                'ExtraTreesRegressor': {
                    'n_estimators': [10, 50, 100, 200],
                    'criterion': ['mse', 'mae'],
                    'max_depth': [None, 10, 50, 100, 200],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'bootstrap': [True, False]
                }
            }
            """
            
            model_report:dict=evaluate_models(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models= models)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            feature_importances = best_model.feature_importances_
            feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

            # Sorting features by importance
            feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

            threshold = 0.01

            # Selecting features above the threshold
            selected_features = feature_importance_df[feature_importance_df['Importance'] >= threshold]['Feature'].tolist()
            # Dropping columns below the threshold from the dataset
            
            X_train_selected = X_train[selected_features]
            X_test_selected = X_test[selected_features]
            
            # Specify the path to save the cleaned data
            final_X_train_path = os.path.join('artifacts', 'final_X_train.csv')
            final_X_test_path = os.path.join('artifacts', 'final_X_test.csv')
            os.makedirs(os.path.dirname(final_X_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(final_X_test_path), exist_ok=True)
            
            # Save cleaned data to a new CSV file in the artifacts folder
            X_train_selected.to_csv(final_X_train_path, index=False)
            X_test_selected.to_csv(final_X_test_path, index=False)
            
            predicted=best_model.predict(X_test)
            
            r2_square = r2_score(y_test, predicted)
            return r2_square
            
        except Exception as e:
            raise CustomException(e, sys)

