from matplotlib import pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import preprocessing as pre
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor

#timer
import time

def showPlot(XPoints, YPoints, XLine, YLine):
    plt.scatter(XPoints, YPoints, color='red')
    plt.plot(XLine, YLine, color='blue')
    plt.title("Comparing real dots with the line built by the Random Forest.")
    plt.xlabel('Experience in years')
    plt.ylabel('Salary')
    plt.show()

def computeRandomForestRegressionModel(X, Y, numberOfTrees):
    regressor = RandomForestRegressor(n_estimators=numberOfTrees)
    regressor.fit(X, Y)
    return regressor

def runRandomForestRegressionRegressionExample(filename):
    start_time = time.time()
    X, y, csv = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")
    
    start_time = time.time()
    X, scaleX = pre.computeScaling(X)
    y, scaleY = pre.computeScaling(np.reshape(y, (-1, 1)))
    elapsed_time = time.time() - start_time
    print("Compute Scaling: %.2f" % elapsed_time, "segundos.")
    
    start_time = time.time()
    RandomForestModel = computeRandomForestRegressionModel(X, y, 50)
    elapsed_time = time.time() - start_time
    print("Compute decision tree regression model: %.2f" % elapsed_time, "segundos.")
    
    # Reshape predictions before using inverse_transform
    XGrid = np.arange(X.min(), X.max(), 0.01).reshape(-1, 1)
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), 
             scaleX.inverse_transform(X), scaleY.inverse_transform(RandomForestModel.predict(X).reshape(-1, 1)))
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), 
             scaleX.inverse_transform(XGrid), scaleY.inverse_transform(RandomForestModel.predict(XGrid).reshape(-1, 1)))

if __name__ == "__main__":
    runRandomForestRegressionRegressionExample("salary2.csv")
