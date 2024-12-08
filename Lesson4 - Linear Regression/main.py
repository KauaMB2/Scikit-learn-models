# Importing necessary libraries
from matplotlib import pyplot as plt
import numpy as np  # For numerical operations (e.g., handling arrays)
import pandas as pd  # For data manipulation and analysis (e.g., reading CSV files, handling data)
import os  # For working with file and directory paths
from sklearn.impute import SimpleImputer  # For handling missing values in the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder  # For encoding categorical variables into numeric labels
from sklearn.model_selection import train_test_split  # For splitting the data into training and testing sets
from sklearn.preprocessing import StandardScaler  # For scaling the data to have zero mean and unit variance



def loadDataset(fileName):
    # Set the directory where the script is located to make file path management easier
    DIR = os.path.dirname(os.path.realpath(__file__))
    # Move one folder back by getting the parent directory
    parent_dir = os.path.dirname(DIR)
    # Load the dataset from the CSV file. The delimiter is set to ';' since it's commonly used in certain regions.
    dataset = pd.read_csv(f'{parent_dir}/datas/{fileName}', delimiter=';')
    # Load all rows and all columns of the dataset except the last column (which is the target variable 'Y')
    # The `iloc` selects the data by position: `:-1` excludes the last column.
    X = dataset.iloc[:, :-1].values
    # Load only the last column (target variable 'Y') of the dataset
    # The `iloc` selects the last column using `-1`.
    Y = dataset.iloc[:, -1].values
    return X, Y
def fillMissingData(X):
    # Create an instance of the SimpleImputer class that will be used to fill missing values in the dataset
    # We will fill missing values with the mean of each column ('mean' strategy)
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    # Fit the imputer on the dataset excluding the last column (X[:, 1:]), then transform the dataset to fill missing values
    # The notation `X[:, 1:]` selects all rows and all columns starting from index 1 (i.e., excluding the first column).
    # After transformation, the missing values in columns 1 and beyond are replaced by their respective column means.
    X[:, 1:] = imputer.fit_transform(X[:, 1:])
    return X
def computeCategorization(X):
    # Apply label encoding to the first column (categorical data), transforming it into numeric labels.
    labelencoder_X = LabelEncoder()
    X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
    # One-hot encoding: Convert the values in the first column (GRE scores, for example) into binary columns
    # Each unique value will get its own column with binary values (0 or 1), making the data suitable for machine learning models.
    X = X[:, 1:]  # Remove the first column, which was just label-encoded
    # Use pandas' get_dummies function to create one-hot encoded columns from the first column of `X`
    D = pd.get_dummies(X[:, 0])  # One-hot encoding the categorical values in the first column (e.g., GRE scores)
    # Insert the one-hot encoded columns into the dataset by adding them at the beginning of `X`
    # `np.insert` adds the one-hot encoded columns as new columns at the start of the dataset.
    X = np.insert(X, 0, D.values, axis=1)
    return X

def splitTrainTestSets(X, Y, testSize):
    # Split the dataset into training and testing sets with 85% for training and 15% for testing
    XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=testSize)
    return XTrain, XTest, YTrain, YTest

def computeScaling(train, test):
    scaleX = StandardScaler() # Create a StandardScaler instance
    XTrain = scaleX.fit_transform(train) # Fit and transform the training data
    XTest = scaleX.fit_transform(test) # Transform the test data using the same scaler
    return XTrain, XTest

def computeLineararRegressionModel(XTrain, YTrain, XTest, YTest):
    regressor = LinearRegression()
    regressor.fit(XTrain, YTrain)
    YPredicted = regressor.predict(XTest)
    print(YTest)
    print("=========================")
    print(YPredicted)
    plt.scatter(XTest[:,-1], YTest, color = 'red')
    plt.plot(XTest[:,-1], regressor.predict(XTest), color='blue')
    plt.title("Inscritos x Visualizações")
    plt.xlabel("Inscritos")
    plt.ylabel("Visualizações")
    plt.show()

def runLinearRegressionExample(fileName):
    X, Y = loadDataset(fileName)
    X = fillMissingData(X)
    X = computeCategorization(X)
    XTrain, XTest, YTrain, YTest = splitTrainTestSets(X, Y, 0.2)
    computeLineararRegressionModel(XTrain, YTrain, XTest, YTest)

if __name__ == "__main__":
    runLinearRegressionExample("svbr.csv")
