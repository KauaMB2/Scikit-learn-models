from matplotlib import pyplot as plt
import preprocessing as pre
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def computeLogisticRegressionModel(XTrain, yTrain, XTest):
    from sklearn.linear_model import LogisticRegression

    classifier = LogisticRegression(solver='lbfgs')
    classifier.fit(XTrain[0], yTrain)

    return classifier

def predictModel(classifier, XTest):
    return classifier.predict(XTest[0])

def evaluateModel(classifier, yPred, yTest):
    # Generate confusion matrix
    confusionMatrix = confusion_matrix(yTest, yPred)

    # Plot confusion matrix using seaborn's heatmap
    plt.figure(figsize=(6, 5))  # You can adjust the figure size
    sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])

    # Adding labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    # Show the plot
    plt.show()

    return confusionMatrix
    
def computeLogisticRegressionExample(filename):
    X, y, csv = pre.loadDataset(filename, ",")
    X = pre.fillMissingData(X, 2, 3)

    #sex
    X = pre.computeCategorization(X)
    #embark
    X = pre.computeCategorization(X)

    XTrain, XTest, yTrain, yTest = pre.splitTrainTestSets(X, y, 0.15)
    XTrain = pre.computeScaling(XTrain)
    XTest = pre.computeScaling(XTest)

    classifier = computeLogisticRegressionModel(XTrain, yTrain, XTest)
    yPred = predictModel(classifier, XTest)
    return evaluateModel(classifier, yPred, yTest)

if __name__ == "__main__":
    print(computeLogisticRegressionExample("titanic.csv"))
