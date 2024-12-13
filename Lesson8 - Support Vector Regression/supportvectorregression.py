from matplotlib import pyplot as plt
import numpy as np
import preprocessing as pre
from sklearn.svm import SVR

#timer
import time



def showPlot(XPoints, YPoints, XLine, YLine):
    plt.scatter(XPoints, YPoints, color='red')
    plt.plot(XLine, YLine, color='blue')
    plt.title("Comparing real dots with the line built by the SVR.")
    plt.xlabel('Experience in years')
    plt.ylabel('Salary')
    plt.show()

def computeSupportVectorRegressionModel(X, Y, Kernel, D):
    regressor = SVR()
    if (Kernel == "poly"):
        regressor = SVR(kernel=Kernel, degree=D)
    else:
        regressor = SVR(kernel=Kernel)
    regressor.fit(X, Y)
    return regressor

def runSupportVectorRegressionExample(filename):
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
    svrModel = computeSupportVectorRegressionModel(X, y, "linear", 2)
    elapsed_time = time.time() - start_time
    print("Compute SVR with kernel Linear: %.2f" % elapsed_time, "segundos.")
    
    # Reshape predictions before using inverse_transform
    predictions = svrModel.predict(X).reshape(-1, 1)
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), 
             scaleX.inverse_transform(X), scaleY.inverse_transform(predictions))
    
    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "poly", 3)
    elapsed_time = time.time() - start_time
    print("Compute SVR with kernel Poly: %.2f" % elapsed_time, "segundos.")
    
    # Reshape predictions before using inverse_transform
    predictions = svrModel.predict(X).reshape(-1, 1)
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), 
             scaleX.inverse_transform(X), scaleY.inverse_transform(predictions))
    
    start_time = time.time()
    svrModel = computeSupportVectorRegressionModel(X, y, "rbf", 2)
    elapsed_time = time.time() - start_time
    print("Compute SVR with kernel RBF: %.2f" % elapsed_time, "segundos.")
    
    # Reshape predictions before using inverse_transform
    predictions = svrModel.predict(X).reshape(-1, 1)
    showPlot(scaleX.inverse_transform(X), scaleY.inverse_transform(y), 
             scaleX.inverse_transform(X), scaleY.inverse_transform(predictions))

if __name__ == "__main__":
    runSupportVectorRegressionExample("salary2.csv")
