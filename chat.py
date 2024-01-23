# Import necessary libraries
import numpy as np  # NumPy numerical computations
import pandas as pd   # Create pandas Dataframes from csv data (for structured data)
import sklearn.datasets  # to import Breast cancer dataset
from sklearn.model_selection import train_test_split  # to test and train data
from sklearn.linear_model import LogisticRegression  # provides logistic Regression model.
from sklearn.metrics import accuracy_score   # supports prediction
import joblib  # for saving and loading the model

# Loading data from sklearn (sklearn.dataset)
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

# Creating pandas dataframe
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target

# Splitting data into features (X) and labels (Y)
X = data_frame.drop(columns="label", axis=1)
Y = data_frame['label']

# Splitting data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Creating a Logistic Regression model
model = LogisticRegression()

# Training the model
model.fit(X_train, Y_train)

# Save the trained model to a file
joblib.dump(model, 'breast_cancer_model.joblib')

# Make predictions on the test set
X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test, X_test_prediction)
print("Accuracy on Testing data:  ", testing_data_accuracy)

# Load the saved model for predictions
loaded_model = joblib.load('breast_cancer_model.joblib')

# Example prediction using the loaded model
input_data = np.array([13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886,
                       2.058, 23.56, 0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144,
                       0.1773, 0.239, 0.1288, 0.2977, 0.07259]).reshape(1, -1)

prediction = loaded_model.predict(input_data)

if prediction[0] == 0:
    print("The Breast Cancer is Malignant")
else:
    print("The Breast Cancer is Benign")
