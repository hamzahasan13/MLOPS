import pandas as pd
from sklearn.impute import SimpleImputer
import os
import sys
from src.exception import CustomException
from src.logger import logging

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
                           'notRepairedDamage':'NotRepaired/Damaged','price':'Price', 'seller':'Seller'}, inplace=True)
        
        return (df)
        
    def fill_missing(self, df):
        """
        This function fills the missing values in categorical columns with 'Unknown'.
        """
        df.loc[:, 'vehicleType'] = df['vehicleType'].fillna("Unknown")
        df.loc[:, 'gearbox'] = df['gearbox'].fillna("Unknown")
        df.loc[:, 'model'] = df['model'].fillna("Unknown")
        df.loc[:, 'fuelType'] = df['fuelType'].fillna("Unknown")
        df.loc[:, 'NotRepaired/Damaged'] = df['NotRepaired/Damaged'].fillna("Unknown")
        
        return df
        
        return (df)

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
    
        df['Seller'] = df['Seller'].replace({'privat': 'Private', 'gewerblich': 'Commercial'});
        df['offerType'] = df['offerType'].replace({'Angebot': 'Customer Offer', 'Gesuch': 'Company Offer'});
        df['gearbox'] = df['gearbox'].replace({'manuell': 'Manual', 'automatik': 'Automatic'});
        df['vehicleType'] = df['vehicleType'].replace({'limousine': 'Limousine', 'kleinwagen': 'Compact Car',
                                                'kombi':'Station Wagon', 'bus':'Bus', 'cabrio': 'Convertible', 
                                                'coupe': 'Coupe', 'suv': 'SUV', 'andere': 'Other'});
        
        df['model'] = df['model'].replace({'golf': 'Golf', 'e_klasse': 'E-Class','3er':'3-series', 
                                    'polo': 'Polo', 'a4': 'A4', 'corsa':'Corsa', 
                                    'astra':'Astra', 'passat': 'Passat', 'a3': 'A3',
                                    'c_klasse': 'C-Class','5er': '5-Series', 'andere': 'Other',
                                    'a6': 'A6', 'focus': 'Focus', 'fiesta': 'Fiesta',
                                    'twingo': 'Twingo'});
        
        df['fuelType'] = df['fuelType'].replace({'benzin':'Gasoline', 'diesel':'Diesel', 'lpg':'LPG', 
                                                        'cng':'CNG', 'hybrid':'Hybrid', 'andere':'Other', 
                                                        'elektro':'Electric'});
        
        df['NotRepaired/Damaged'] = df['NotRepaired/Damaged'].replace({'nein':'No', 'ja':'Yes'});
        
        return df
    
    def cat_imputing(self, df):
        """
        This code imputes missing values in the categorical variables
        """
        
        cols_to_impute = ['vehicleType', 'gearbox', 'model', 'fuelType', 'NotRepaired/Damaged']

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
        df.loc[:, 'Risk_Level'] = risk_levels
        return (df)
    
    def drop_cols(self, df):
        """
        This function drops irrelevant columns.
        """
        cols_to_drop = ['index', 'nrOfPictures', 'postalCode', 'dateCrawled', 'name', 'dateCreated', 'lastSeen', 'Registration_Month',
                        'model', 'brand'];
        df.drop(cols_to_drop, axis=1, inplace=True);
        
        return (df)
        
    def clean_data(self, df):
        # Apply multiple cleaning functions sequentially
        df = self.data_analysis(df)
        df = self.col_names_replace(df)
        df = self.fill_missing(df)
        df = self.capitalize_letters(df)
        df = self.obs_mapping(df)
        #print(df.columns)
        #df = self.cat_imputing(df)
        df = self.data_filtering(df)
        df = self.drop_cols(df)
        
        return df
    
if __name__ == "__main__":
    try:
        logging.info("Entered the data cleaning process")
        
        # Load data
        df = pd.read_csv('notebook/data/data.csv')
        logging.info('Read the data into dataframe')
        
        # Initialize DataCleaning object
        cleaner = DataCleaning()
        
        # Clean the data using multiple cleaning functions
        cleaned_df = cleaner.clean_data(df)
        
        # Specify the path to save the cleaned data
        cleaned_data_path = os.path.join('artifacts', 'cleaned_data.csv')
        os.makedirs(os.path.dirname(cleaned_data_path), exist_ok=True)
        
        # Save cleaned data to a new CSV file in the artifacts folder
        cleaned_df.to_csv(cleaned_data_path, index=False)
        
        logging.info('Data cleaning process completed')
        
    except Exception as e:
        raise CustomException(e, sys)