import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Example dataset
data = {
    'age': [30, 40, None, 50, 35, 45],
    'income': [50000, 60000, 70000, None, 55000, None],
    'gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'education': ['High School', 'College', 'College', 'Graduate', 'College', 'Graduate'],
    'target': [0, 1, 1, 0, 0, 1]
}

# Convert data dictionary to pandas DataFrame
df = pd.DataFrame(data)

# Split data into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Define numerical columns
numerical_columns = ['age', 'income']

# Define categorical columns
categorical_columns = ['gender', 'education']

# Define preprocessing steps for numerical columns
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Define preprocessing steps for categorical columns
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Create ColumnTransformer
preprocessor = ColumnTransformer(
    [
        ("num_pipeline", num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]
)

# Fit and transform on train data
transformed_train_data = preprocessor.fit_transform(train_df)

# Get the column names after transformation
transformed_columns = (
    preprocessor.named_transformers_['num_pipeline']
    .named_steps['scaler']
    .get_feature_names_out(numerical_columns)
    .tolist()
    + preprocessor.named_transformers_['cat_pipeline']
    .named_steps['encoder']
    .get_feature_names_out(categorical_columns)
    .tolist()
)

# Create a DataFrame from the transformed train data
transformed_train_df = pd.DataFrame(transformed_train_data, columns=transformed_columns)

# Concatenate the transformed train DataFrame with any remaining columns from the original train DataFrame
remaining_columns = train_df.columns.difference(numerical_columns + categorical_columns)
final_train_df = pd.concat([train_df[remaining_columns].reset_index(drop=True), transformed_train_df.reset_index(drop=True)], axis=1)

# Transform test data
transformed_test_data = preprocessor.transform(test_df)

# Create a DataFrame from the transformed test data
transformed_test_df = pd.DataFrame(transformed_test_data, columns=transformed_columns)

# Concatenate the transformed test DataFrame with any remaining columns from the original test DataFrame
final_test_df = pd.concat([test_df[remaining_columns].reset_index(drop=True), transformed_test_df.reset_index(drop=True)], axis=1)

# Now, final_train_df and final_test_df contain the original columns along with the transformed ones
print("Final Train DataFrame:")
print(final_train_df)
print("\nFinal Test DataFrame:")
print(final_test_df)
