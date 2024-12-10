from matplotlib import pyplot as plt
import preprocessing as pre
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

#timer
import time

def showPlot(XPoints, YPoints, XLine, YLine):
    plt.scatter(XPoints, YPoints, color='red')
    plt.plot(XLine, YLine, color='blue')
    plt.title("Comparing real dots with the line built by the polinomial.")
    plt.xlabel('Experience in years')
    plt.ylabel('Salary')
    plt.show()

def computePolynomialLinearRegressionModel(X, Y, degree):
    polynomialFeatures = PolynomialFeatures(degree=degree)
    XPoly = polynomialFeatures.fit_transform(X)
    polynomialRegressionModel = LinearRegression()
    polynomialRegressionModel.fit(XPoly, Y)
    return XPoly, polynomialRegressionModel

def runPolynomialLinearRegressionExample(filename):
    start_time = time.time()
    X, y = pre.loadDataset(filename)
    elapsed_time = time.time() - start_time
    print("Load Dataset: %.2f" % elapsed_time, "segundos.")
    start_time = time.time()
    XPoly, polyLinearRegressor = computePolynomialLinearRegressionModel(X, y, 20)
    elapsed_time = time.time() - start_time
    print("Compute Polynomial Linear Regression: %.2f" % elapsed_time, "segundos.")
    showPlot(X, y, X, polyLinearRegressor.predict(XPoly))

if __name__ == "__main__":
    runPolynomialLinearRegressionExample("salary2.csv")
