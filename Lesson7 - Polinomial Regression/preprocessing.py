# Importing the necessary libraries for numerical operations and data manipulation
import os
import numpy as np  # For handling arrays and numerical operations
import pandas as pd  # For reading and manipulating datasets (DataFrames)

# Function to load the dataset from a CSV file
def loadDataset(fileName):
    # Set the directory where the script is located to make file path management easier
    DIR = os.path.dirname(os.path.realpath(__file__))
    # Move one folder back by getting the parent directory
    parent_dir = os.path.dirname(DIR)
    # Load the dataset from the CSV file. The delimiter is set to ';' since it's commonly used in certain regions.
    dataset = pd.read_csv(f'{parent_dir}/datas/{fileName}', delimiter=';')
    # X represents the feature matrix, which contains all columns except the last one (target)
    X = dataset.iloc[:,:-1].values  # Select all rows and all columns except the last
    # y represents the target variable, which is the last column of the dataset
    y = dataset.iloc[:,-1].values  # Select all rows and only the last column (target)
    
    # Return the feature matrix X and the target variable y
    return X, y


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


# Function to perform categorization (Label Encoding and One-Hot Encoding)
# The argument 'i' is the column index where categorization is performed
def computeCategorization(X, i):
    # LabelEncoder from scikit-learn is used to convert categorical labels into numeric labels
    from sklearn.preprocessing import LabelEncoder
    labelencoder_X = LabelEncoder()

    # Apply LabelEncoder to the 'i' column of the feature matrix (X)
    X[:, i] = labelencoder_X.fit_transform(X[:, i])

    # One-Hot Encoding is used to convert the categorical values into binary columns (0 or 1)
    D = pd.get_dummies(X[:,i]).values  # pd.get_dummies() returns the one-hot encoded matrix

    #Inser the binary columns of the one-hot encoder in the matrix
    if(i == 0):# If we are working with the first column (i == 0)
        X = X[:,1:] # Remove the first column (because it was label-encoded)
        X = np.insert(X, 0, D, axis=1)# Insert the one-hot encoded columns at the beginning of the matrix
        X = X[:,1:]# Remove the first column again to avoid the dummy variable trap (multicollinearity)
    else:# If we are working with any other column (i != 0)
        X = X[:,:i]# Slice the matrix to keep columns up to index 'i' (exclude the 'i' column)
        for j in range(0, D.shape[1]):# Insert the one-hot encoded columns into the feature matrix (at the 'i' column index)
            X = np.insert(X, i, D[:,j], axis=1)
        X = X[:,:-1]# Remove the last column to avoid the dummy variable trap (to avoid multicollinearity)
    return X# Return the updated feature matrix X with one-hot encoded columns


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
