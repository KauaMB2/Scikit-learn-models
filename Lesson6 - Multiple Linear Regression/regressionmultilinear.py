import numpy as np
import preprocessing as pre
import statsmodels.formula.api as sm

#timer
import time

def computeAutomaticBackwardElimination(XTrain, yTrain, XTest, sl):
    import statsmodels.api as sm

    # Add a column of ones to X for the intercept term
    XTrain = np.insert(XTrain, 0, 1, axis=1)
    XTest = np.insert(XTest, 0, 1, axis=1)

    numVars = len(XTrain[0])

    while True:
        regressor_OLS = sm.OLS(yTrain, XTrain.astype(float)).fit()  # Use OLS instead of ols
        maxVar = max(regressor_OLS.pvalues)  # Get the max p-value
        print(regressor_OLS.summary())

        if maxVar > sl:
            # Find the feature with the max p-value
            max_pval_index = np.argmax(regressor_OLS.pvalues)
            # Remove that feature
            XTrain = np.delete(XTrain, max_pval_index, 1)
            XTest = np.delete(XTest, max_pval_index, 1)
        else:
            break
    return XTrain, XTest


#https://medium.com/@manjabogicevic/multiple-linear-regression-using-python-b99754591ac0
def computeMultipleLinearRegressionModel(XTrain, yTrain, XTest, yTest):
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor.fit(XTrain, yTrain)

    yPred = regressor.predict(XTest)
    # for i in range(0, yPred.shape[0]):
    #     print(yPred[i], yTest[i], abs(yPred[i] - yTest[i]))
    #     time.sleep(0.5)

def runMultipleLinearRegressionExample(filename):
    start_time = time.time()
    X, y = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.fillMissingData(X, 0, 2)
    elapsed_time = time.time() - start_time
    print("Fill Missing Data: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    X = pre.computeCategorization(X, 3)
    elapsed_time = time.time() - start_time
    print("Compute Categorization: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.8)
    elapsed_time = time.time() - start_time
    print("Split Train Test sets: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    XTrain, XTest = computeAutomaticBackwardElimination(XTrain, yTrain, XTest, 0.05)
    elapsed_time = time.time() - start_time
    print("Compute Automatic Backward Elimination: %.2f" % elapsed_time, "segundos.")

    start_time = time.time()
    computeMultipleLinearRegressionModel(XTrain, yTrain, XTest, yTest)
    elapsed_time = time.time() - start_time
    print("Compute Multiple Linear Regression: %.2f" % elapsed_time, "segundos.")

if __name__ == "__main__":
    runMultipleLinearRegressionExample("insurance.csv")
