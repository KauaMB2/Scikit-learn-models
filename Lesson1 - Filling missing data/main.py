# Importing necessary libraries
import numpy as np  # For numerical operations
import pandas as pd  # For data manipulation and analysis
import os  # For working with file paths
from sklearn.impute import SimpleImputer  # For handling missing values in the dataset

# Set the directory where the script is located to make file path management easier
DIR = os.path.dirname(os.path.realpath(__file__))
# Move one folder back by getting the parent directory
parent_dir = os.path.dirname(DIR)

# Load the dataset from the CSV file. The delimiter is set to ';' since it is typically used in CSV files from some regions
dataset = pd.read_csv(fr'{parent_dir}/datas/science_vlogs_brazil.csv', delimiter=';')

# Get all values from the dataset (rows and columns) and ignore the column names
X = dataset.iloc[:,:].values

# Create an instance of the SimpleImputer class that will be used to fill missing values
# Here, we choose to fill missing values with the mean of the column (this is the 'mean' strategy)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer to the selected columns (columns 1 and 2) to learn how to replace missing values
# The notation X[:, 1:3] selects all rows, but only columns 1 and 2 (Python indexing starts at 0)
imputer = imputer.fit(X[:,1:3])

# Transform the data in columns 1 and 2, filling in missing values with the column mean
# The transform method replaces NaN values in the dataset with the calculated mean of each column
# After transformation, we convert the resulting data to string type
X = imputer.transform(X[:,1:3]).astype(str)

# Insert the first column (original data in column 0) back into the transformed data
# This ensures that the first column (e.g., identifiers or non-numeric data) is preserved in the final dataset
X = np.insert(X, 0, dataset.iloc[:,0].values, axis=1)

# Print the final result to show the modified dataset with missing values filled
print(X)
