
# Lessons 1 - 3:
### Categorial variables and numerical variables
**1. Categorical Variables (Qualitative Data)**
</br>
Categorical variables represent categories or labels. They are used to describe characteristics that can be grouped into distinct categories, where each category represents a different value. These values typically do not have a meaningful order or ranking.
 * Examples:
    * Gender: Male, Female
    * Country: USA, Canada, Germany, Japan
    * Color: Red, Blue, Green
    * Product type: Electronics, Clothing, Food
    * Marital status: Single, Married, Divorced
    * Types of Categorical Variables:
    * Nominal: Categories with no inherent order (e.g., color, country, or product type). No ranking is possible.
    * Ordinal: Categories with a meaningful order or ranking, but the distances between categories are not meaningful (e.g., education level: High School, Bachelor's, Master's, PhD; or satisfaction level: Poor, Average, Good).

**2. Numerical Variables (Quantitative Data)**
</br>
Numerical variables represent quantities and are measured in terms of numbers. They can be further divided into two types based on whether the values are discrete or continuous:

**Types of Numerical Variables:**
Discrete: These are countable values, often integers, that represent whole numbers. Discrete variables typically arise from counting something (e.g., number of children, number of products sold).
 * Examples:
    * Number of students in a class
    * Number of cars in a parking lot
    * Number of goals scored in a game

Continuous: These are values that can take any real number within a range. Continuous variables arise from measurements and can have infinite possible values between any two values.
 * Examples:
    * Height (e.g., 5.8 feet, 6.2 feet)
    * Weight (e.g., 150.5 pounds, 200.1 pounds)
    * Temperature (e.g., 32.5°C, 100.2°C)

### Fit concept
<u>In the context of the training of machine learning models</u>, the expression **"Fit"** refers to the process of training a model on a dataset. This involves the model learning the underlying patterns or relationships in the data. For instance, when you call the fit method on a machine learning model (e.g., LinearRegression, RandomForestClassifier), the model adjusts its parameters (like coefficients, weights, etc.) based on the data it is exposed to, <u>but in the context of preprocessing</u> and transformations (like scaling, normalization, or encoding), "fit" still refers to learning something from the data, but not in terms of model parameters. Instead, it means learning the necessary statistics or properties needed for the transformation. Example:
 - For scaling(with `StandardScaler`), "fit" learns the mean and standard deviation of the features in the training data.
 - For normalization (with `MinMaxScaler`), "fit" learns the minimum and maximum values of each feature.
1. Training a model
```python
from sklearn.linear_model import LinearRegression
# Example data
X_train = [[1, 2], [3, 4], [5, 6]]
y_train = [5, 7, 9]
model = LinearRegression()
model.fit(X_train, y_train)  # Fit learns the best-fit line
```
2. Preprocessing
```python
from sklearn.preprocessing import StandardScaler
data = [[1, 2], [3, 4], [5, 6]]
scaler = StandardScaler()
# Fit learns the mean and standard deviation
scaler.fit(data)
```
**Key distintions:**
- **Model fitting** (e.g., `LinearRegression.fit()`) train the data.
- **Preprocessing fitting** (e.g., `StandardScaler.fit()`) learns statistics from the
data to apply transformations.

### Transformation concept
In scikit-learn, the **transform() method** is used to apply a transformation to data using parameters that were learned previously (usually during the fit step).

**General Purpose of transform:**
The transform method modifies the input data based on the parameters learned during the fit step.

**Common use cases of transform:**
 - Scaling: In the case of feature scaling (e.g., using `StandardScaler`), transform applies the scaling formula (using the mean and standard deviation learned during fit) to the data.
 - Encoding: For categorical encoding (e.g., using `OneHotEncoder`), transform converts the input features into the encoded format based on the learned mapping.
 - Dimensionality Reduction: In algorithms like PCA (Principal Component Analysis), transform reduces the dimensions of the data using the principal components learned during fit.

### One-hot encoder concept
**One-hot encoding** is a technique used in machine learning to convert categorical variables into numerical variables that can be processed by machine learning algorithms. It works by creating a new binary feature for each category in the categorical variable. For example, if we have a categorical variable with three categories A, B, and C, the one-hot encoded version would be a binary vector [1, 0, 0] for category A, [0, 1, 0] for category B, and [0, 0, 1] for category C.
```python
from sklearn.preprocessing import OneHotEncoder
# Example data
data = [[1, 'A'], [2, 'B'], [3, 'C']
encoder = OneHotEncoder()
# Fit learns the categories
encoder.fit(data)
```
```python
### Example of using transform
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
data = [[1, 'A'], [2, 'B'], [3, 'C']]
encoder = OneHotEncoder()
# Fit learns the categories
encoder.fit(data)
# Transform applies the one-hot encoding
encoded_data = encoder.transform(data)
```