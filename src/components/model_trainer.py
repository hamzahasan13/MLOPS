import os
import sys
import pandas as pd
from dataclasses import dataclass
import mlflow
from hyperopt import hp
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.utils import save_obj, evaluate_models
from src.components.data_transformation import DataTransformationConfig

    
@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        
        self.model_trainer_config=ModelTrainerConfig()
        self.data_transformation_config = DataTransformationConfig()


    def initiate_model_trainer(self, train_array, test_array, input_feature_train_df, input_feature_test_df):
        try:
            

            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array.iloc[:, :-1],
                train_array.iloc[:, -1],
                test_array.iloc[:, :-1],
                test_array.iloc[:, -1]
            )
            
            
            X_orig = pd.concat([input_feature_train_df, input_feature_test_df]).reset_index(drop=True)
            
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

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")
            
            # Log parameters
            mlflow.set_tag("Model", best_model_name)
            mlflow.sklearn.log_model(best_model_name, "best_model")

            # Evaluate best model
            predicted = best_model.predict(X_test)
            r2_square = round(r2_score(y_test, predicted), 2)
            rmse = round(np.sqrt(mean_squared_error(y_test, predicted)), 2)
            mae = round(mean_absolute_error(y_test, predicted), 2)

            # Log metrics
            mlflow.log_metric("r2", r2_square)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.end_run()

            predicted=best_model.predict(X_test)
            
            original_columns = set(feature.split('_', 1)[0] for feature in X_train)
            
            X_orig_selected = X_orig[list(original_columns)]
            
            r2_square = round(r2_score(y_test, predicted), 2)
            
            ## Hyperparameter Tuning with MLflow
            """
            with mlflow.start_run():
                mlflow.set_tag("Model", best_model)
                
                if best_model_name == 'ElasticNet':
                    search_space = {'alpha': hp.loguniform('alpha', 0.01, 1),
                                    'l1_ratio': hp.loguniform('l1_ratio', 0, 1)}
                    
                    mlflow.set_tag("model", 'ElasticNet')
                    for alpha in search_space['alpha']:
                        for l1_ratio in search_space['l1_ratio']:
                            ml_model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X_train, y_train)
                            pred = ml_model.predict(X_test)
                            r2_sq = r2_score(pred, y_test)
                            
                            print("alpha:", alpha, "l1_ratio:", l1_ratio, "R2 Score:", r2_sq)
                            
                    mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
                    mlflow.log_metric("r2", r2_square)

                
                elif best_model_name == 'DecisionTreeRegressor':
                    search_space = {'min_samples_split': hp.loguniform('min_samples_split', 2, 10),
                                    'min_samples_leaf': hp.loguniform('min_samples_leaf', 1, 4)}
                    
                    mlflow.set_tag("model", 'DecisionTreeRegressor')
                    for min_samples_split in search_space['min_samples_split']:
                        for min_samples_leaf in search_space['min_samples_leaf']:
                            ml_model = DecisionTreeRegressor(min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf).fit(X_train, y_train)
                            pred = ml_model.predict(X_test)
                            r2_sq = r2_score(pred, y_test)
                            
                            print("min_samples_split:", min_samples_split, "min_samples_leaf:", min_samples_leaf, "R2 Score:", r2_sq)
                            
                    mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
                    mlflow.log_metric("r2", r2_square)

                elif best_model_name == 'RandomForestRegressor':
                    search_space = {'n_estimators': hp.loguniform('n_estimators', 10, 200),
                                    'max_depth': hp.loguniform('max_depth', 10, 50)}
                    
                    mlflow.set_tag("model", 'RandomForestRegressor')
                    for n_estimators in search_space['n_estimators']:
                        for max_depth in search_space['max_depth']:
                            ml_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth).fit(X_train, y_train)
                            pred = ml_model.predict(X_test)
                            r2_sq = r2_score(pred, y_test)
                            
                            print("n_estimators:", n_estimators, "Max Depth:", max_depth, "R2 Score:", r2_sq)
                            
                    mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
                    mlflow.log_metric("r2", r2_square)

                elif best_model_name == 'GradientBoostingRegressor':
                    search_space = {'learning_rate': [0.001, 0.01, 0.1],
                                    'max_depth': [3, 4, 5, 6]}
                    
                    mlflow.set_tag("model", 'GradientBoostingRegressor')
                    for learning_rate in search_space['learning_rate']:
                        for max_depth in search_space['max_depth']:
                            ml_model = GradientBoostingRegressor(learning_rate=learning_rate, max_depth=max_depth).fit(X_train, y_train)
                            pred = ml_model.predict(X_test)
                            r2_sq = r2_score(pred, y_test)
                            
                            print("Learning Rate:", learning_rate, "Max Depth:", max_depth, "R2 Score:", r2_sq)
                            
                    mlflow.log_params({'learning_rate': learning_rate, 'max_depth': max_depth})
                    mlflow.log_metric("r2", r2_square)
                """
                
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            return r2_square
        
        except Exception as e:
            raise CustomException(e, sys)
        
            
