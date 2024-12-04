import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer

DIR = os.path.dirname(os.path.realpath(__file__))

dataset = pd.read_csv(f'{DIR}/science_vlogs_brazil.csv', delimiter=';')
X = dataset.iloc[:,:].values #Get all the values of all the rows and columns of the dataset and ignore the title of the columns
imputer = SimpleImputer(missing_values=np.nan, strategy= 'mean') #Create a instance of the SimpleImputer class responsable for fill missing values with the mean of the selected strategy
imputer = imputer.fit(X[:,1:3])
X = imputer.transform(X[:,1:3]).astype(str)
X = np.insert(X, 0, dataset.iloc[:,0].values, axis=1)

print(X)