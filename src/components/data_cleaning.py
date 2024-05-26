import pandas as pd
from sklearn.impute import SimpleImputer
from src.exception import CustomException
import sys
import subprocess
import os
class DataCleaning:
    def __init__(self):
        pass
    
    
    def data_analysis(self, df):
        """
        This function performs basic data analysis.
        """
        
        ## Rows and columns
        print('Total Rows:', df.shape[0]);
        print('Total Columns:', df.shape[1], '\n');
        
        ## Column names
        print('Column Names:', '\n', df.columns, '\n')
        
        ## Column types
        print('Column Types:', '\n', df.dtypes)
        
        ## Missing values
        print('Percentage of missing values in each column:', '\n', 100*df.isnull().sum()/df.shape[0])
        
        return (df)
        
    def col_names_replace(self, df):
        
        """
        This function changes the name of columns for better readability.
        """
        df.rename(columns={'yearOfRegistration':'Registration_Year', 'powerPS':'HorsePower', 'monthOfRegistration':'Registration_Month', 
                           'notRepairedDamage':'NotRepairedDamaged','price':'Price', 'seller':'Seller'}, inplace=True)
        
        return (df)
        
    def fill_missing(self, df):
        """
        This function fills the missing values in categorical columns with 'Other'.
        """
        df.loc[:, 'vehicleType'] = df['vehicleType'].fillna("Other")
        df.loc[:, 'gearbox'] = df['gearbox'].fillna("Other")
        df.loc[:, 'model'] = df['model'].fillna("Other")
        df.loc[:, 'fuelType'] = df['fuelType'].fillna("Other")
        df.loc[:, 'NotRepairedDamaged'] = df['NotRepairedDamaged'].fillna("Other")
        
        return df


    def capitalize_letters(self, df):
        """
        This function capitalizes the first letter of each observation in the respective column.
        """
        if isinstance(df, pd.Series):
            cap_words = df.str[0].str.upper() + df.str[1:]
        elif isinstance(df, pd.DataFrame):
            cap_words = df.apply(lambda x: x.str[0].str.upper() + x.str[1:] if x.dtype == 'object' else x)
        else:
            raise ValueError("Input must be a Pandas Series or DataFrame.")
        
        return cap_words
    
    def obs_mapping(self, df):
        """
        This function translates and maps German words to English words.
        """
    
        df['Seller'] = df['Seller'].replace({'Privat': 'Private', 'Gewerblich': 'Commercial'});
        df['offerType'] = df['offerType'].replace({'Angebot': 'Customer Offer', 'Gesuch': 'Company Offer'});
        df['gearbox'] = df['gearbox'].replace({'Manuell': 'Manual', 'Automatik': 'Automatic'});
        df['vehicleType'] = df['vehicleType'].replace({'Limousine': 'Limousine', 'Kleinwagen': 'Compact Car',
                                                'Kombi':'Station Wagon', 'Bus':'Bus', 'Cabrio': 'Convertible', 
                                                'Coupe': 'Coupe', 'Suv': 'SUV', 'Andere': 'Other'});
        
        df['model'] = df['model'].replace({'Golf': 'Golf', 'E_klasse': 'E-Class','3er':'3-series', 
                                    'Polo': 'Polo', 'A4': 'A4', 'Corsa':'Corsa', 
                                    'Astra':'Astra', 'Passat': 'Passat', 'A3': 'A3',
                                    'C_klasse': 'C-Class','5er': '5-Series', 'Andere': 'Other',
                                    'A6': 'A6', 'Focus': 'Focus', 'Fiesta': 'Fiesta',
                                    'Twingo': 'Twingo'});
        
        df['fuelType'] = df['fuelType'].replace({'Benzin':'Gasoline', 'Diesel':'Diesel', 'Lpg':'LPG', 
                                                        'Cng':'CNG', 'Hybrid':'Hybrid', 'Andere':'Other', 
                                                        'Elektro':'Electric'});
        
        df['NotRepairedDamaged'] = df['NotRepairedDamaged'].replace({'Nein':'No', 'Ja':'Yes'});
        
        columns_to_check = ['vehicleType', 'gearbox', 'fuelType', 'NotRepairedDamaged'];
        for col in columns_to_check:
            df = df[df[col] != 'Other']
        
        return df
    
    def cat_imputing(self, df):
        """
        This code imputes missing values in the categorical variables
        """
        
        cols_to_impute = ['vehicleType', 'gearbox', 'model', 'fuelType', 'NotRepairedDamaged']

        imputer = SimpleImputer(strategy='most_frequent')
        imputed_data = imputer.fit_transform(df[cols_to_impute])

        df[cols_to_impute] = imputed_data
        return (df)
    
    def data_filtering(self, df):
        ## Filtering the dataset for years between 1950 and 2022. 
        ## Filtering for Price range between $500 and $100000.
        ## Filtering for horsepower between 10 and 2000.

        df = df[(df['Registration_Year']>=1950) & (df['Registration_Year']<=2022) &
                            (df['Price']>=500) & (df['Price']<=100000) &
                            (df['HorsePower']>10) & (df['HorsePower']<=2000)];
        
        ## Feature engineering a new column using the registeration year.
        risk_levels = df['Registration_Year'].apply(lambda x: 'High' if 1950 <= x <= 2000 else 'Medium' if 2000 < x <= 2010 else 'Low')

        # Assign the Series to the 'Risk_Level' column using .loc
        df.loc[:, 'RiskLevel'] = risk_levels
        return (df)
    
    def drop_cols(self, df):
        """
        This function drops irrelevant columns.
        """
        cols_to_drop = ['index', 'nrOfPictures', 'postalCode', 'dateCrawled', 'name', 'dateCreated', 'lastSeen', 'Registration_Month',
                        'model', 'brand', 'abtest'];
        df.drop(cols_to_drop, axis=1, inplace=True);
        
        return (df)
    
    def sample_with_all_categories(self, df, n_samples=15300, random_state=0):
        categorical_columns = df.select_dtypes(include=['object', 'category']).columns
        samples_per_category = {}

        for col in categorical_columns:
            # Find unique values and their counts
            unique_values = df[col].value_counts().to_dict()
            samples_per_category[col] = {
                value: max(1, n_samples // len(unique_values)) for value in unique_values
            }

        sampled_dfs = []

        for col in categorical_columns:
            for value, count in samples_per_category[col].items():
                sampled_dfs.append(df[df[col] == value].sample(count, replace=True))

        # Concatenate all the sampled DataFrames
        sampled_df = pd.concat(sampled_dfs)

        # Adjust the final sample size to n_samples
        sampled_df = sampled_df.sample(n=n_samples, random_state=random_state)

        # Reset index to avoid duplicate index issues
        sampled_df = sampled_df.reset_index(drop=True)
        
        return sampled_df
        
    def clean_data(self, df):
        # Apply multiple cleaning functions sequentially
        #df = self.data_analysis(df)
        df = self.col_names_replace(df)
        df = self.fill_missing(df)
        df = self.capitalize_letters(df)
        df = self.obs_mapping(df)
        df = self.data_filtering(df)
        df = self.drop_cols(df)
        df = self.sample_with_all_categories(df)
        
        return df
    
    def initialize_dvc(self):
        if not os.path.exists(".dvc"):
            try:
                
                subprocess.run(['dvc', 'init'])
                print("DVC initialized successfully.")
            except CustomException as e:
                print(e, sys)
                
        else:
            print(".dvc already exists. DVC Init is skipped.")
    
    def run_dvc_command(self, command):
        try:
            subprocess.run(f"dvc add {command}", shell=True, check=True)
    
        except CustomException as e:
            print(e, sys)