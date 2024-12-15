# Importing the necessary libraries for numerical operations and data manipulation
import os
import numpy as np  # For handling arrays and numerical operations
import pandas as pd  # For reading and manipulating datasets (DataFrames)
from sklearn.preprocessing import StandardScaler

# Function to load the dataset from a CSV file
def loadDataset(fileName, delimiter):
    # Set the directory where the script is located to make file path management easier
    DIR = os.path.dirname(os.path.realpath(__file__))
    # Move one folder back by getting the parent directory
    parent_dir = os.path.dirname(DIR)
    # Load the dataset from the CSV file. The delimiter is set to "delimiter"
    dataset = pd.read_csv(f'{parent_dir}/datas/{fileName}', delimiter=delimiter)
    # X represents the feature matrix, which contains all columns except the last one (target)
    X = dataset.iloc[:,:-1].values  # Select all rows and all columns except the last
    # y represents the target variable, which is the last column of the dataset
    y = dataset.iloc[:,-1].values  # Select all rows and only the last column (target)
    
    # Return the feature matrix X and the target variable y
    return X, y, dataset


# Function to handle missing data in specific columns
def fillMissingData(X, inicioColuna, fimColuna):
    # SimpleImputer from scikit-learn is used to fill missing data
    from sklearn.impute import SimpleImputer
    # Create an imputer instance that replaces missing values with the median of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    # Apply the imputer to the specified columns (from 'inicioColuna' to 'fimColuna')
    X[:,inicioColuna:fimColuna + 1] = imputer.fit_transform(X[:,inicioColuna:fimColuna + 1])
    # Return the dataset with missing values replaced by the median
    return X


def computeCategorization(X):
    # Importing the necessary class for label encoding
    from sklearn.preprocessing import LabelEncoder
    # Create an instance of the LabelEncoder to convert labels into numerical format
    labelencoder_X = LabelEncoder()
    
    # Applying label encoding on the first column (assuming it's categorical)
    # The fit_transform method encodes the unique categories into numeric values
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])

    # One-hot encoding the first column (after label encoding)
    # The pd.get_dummies method creates binary (0 or 1) columns for each unique category
    D = pd.get_dummies(X[:,0]).values
    
    # Now, remove the first column (the one we just encoded) from the original data
    X = X[:,1:]
    
    # Append the one-hot encoded columns to the remaining data
    # We iterate through each column in D and insert it at the end of X
    for ii in range(0, D.shape[1]):
        X = np.insert(X, X.shape[1], D[:,ii], axis=1)
    
    # Remove the last column after inserting all one-hot encoded columns(Solve the Dummy Variable Trap)
    # This is done to avoid having redundant columns from the one-hot encoding process
    X = X[:,:X.shape[1] - 1]

    # Return the transformed matrix X with one-hot encoding applied
    return X


# Function to split the dataset into training and test sets
def splitTrainTestSets(X, y, testSize):
    # train_test_split from scikit-learn is used to split the data
    from sklearn.model_selection import train_test_split
    # Split the data into training and test sets
    # test_size determines the proportion of the dataset to include in the test split
    XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size = testSize)
    
    # Return the training and testing sets for both features (X) and target variable (y)
    return XTrain, XTest, yTrain, yTest


# Function to apply feature scaling to the training and test datasets
def computeScaling(train, test):
    # StandardScaler from scikit-learn is used to standardize the data
    from sklearn.preprocessing import StandardScaler
    # Create a StandardScaler instance
    scaleX = StandardScaler()

    # Fit the scaler to the training data and then transform it
    # Standardizing means to scale the data to have zero mean and unit variance
    train = scaleX.fit_transform(train)

    # Scale the test data using the same scaler fitted to the training data
    test = scaleX.transform(test)

    # Return the scaled training and test data
    return train, test

def computeScaling(X):
    scale = StandardScaler()
    X = scale.fit_transform(X)
    return X, scale