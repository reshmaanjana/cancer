import numpy as np  # NumPy numerical computations
import pandas as pd   # Create pandas Dataframes from csv data (for structed data)
import sklearn.datasets  # to import Breast cancer dataset
from sklearn.model_selection import train_test_split  # to test and train data
from sklearn.linear_model import LogisticRegression  # provides logistic Regression model.
from sklearn.metrics import accuracy_score   # supports prediction
#Loading data from sklearn (sklearn.dataset)
breast_cancer_dataset= sklearn.datasets.load_breast_cancer()
#print(breast_cancer_dataset)
#creatinf pandas dataframe
data_frame= pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame.head() # converted data into a structed form using the pandas DataFrame.
#Adding target column
data_frame['label']= breast_cancer_dataset.target
data_frame.tail()
# Number of rows and columns in our dataset.
data_frame.shape  # As the result of .shape is a tuple object we need not use "()" parentheses here.
#getting information about our data
data_frame.info()
data_frame.isnull().sum()
# statistical information about data
data_frame.describe()
data_frame['label'].value_counts()
data_frame.groupby('label').mean()
X= data_frame.drop(columns="label", axis=1) # Whenever dropping a column mention axis value as 1, and if dropping row axis=0
Y= data_frame['label']
print(X)
print(Y)
X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size=0.2, random_state=2) # created 4 arrays to store train and test data
print(X.shape, X_train.shape, X_test.shape) # You can see 20% of data is now split into test and 80% in train
model= LogisticRegression()
model.fit(X_train, Y_train)
X_train_prediction=model.predict(X_train)
training_data_accuracy=accuracy_score(Y_train, X_train_prediction)
print("Accuracy on Training data: ",training_data_accuracy)
X_test_prediction=model.predict(X_test)
testing_data_accuracy=accuracy_score(Y_test, X_test_prediction)
print("Accuracy on Testing data:  ",testing_data_accuracy )
input_data=(13.54,14.36,87.46,566.3,0.09779,0.08129,0.06664,0.04781,0.1885,0.05766,0.2699,0.7886,2.058,23.56,0.008462,0.0146,0.02387,0.01315,0.0198,0.0023,15.11,19.26,99.7,711.2,0.144,0.1773,0.239,0.1288,0.2977,0.07259)
input_data_numpy_array= np.asarray(input_data)
#reshaping numpy array as we are predicting for one data point
input_data_reshaped= input_data_numpy_array.reshape(1,-1)
prediction=model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
    print("The Breast Cancer is Malignant")
else:
    print("The Breast Cancer is Benign")
    

