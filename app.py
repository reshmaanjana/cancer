import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Assuming your dataset is stored in a CSV file named 'cancer_data.csv'
df = pd.read_csv('data.txt')

# Assuming the target variable is named 'target' (1 for malignant, 0 for benign)
X = df.drop('target', axis=1)
y = df['target']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but recommended for some algorithms)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Save the trained model to a file
joblib.dump(model, 'cancer_model.joblib')

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_rep}')

# Load the saved model for predictions
loaded_model = joblib.load('cancer_model.joblib')

# Example prediction using the loaded model
new_data = [[15.0, 13.0, 95.0, 700.0, 0.1, 0.2, 0.15, 0.07, 0.2, 0.05, 0.8, 0.7, 6.0, 120.0, 0.005, 0.02, 0.03, 0.01, 0.02, 0.003, 18.0, 15.0, 110.0, 900.0, 0.12, 0.3, 0.25, 0.1, 0.3, 0.08]]
scaled_new_data = scaler.transform(new_data)
prediction = loaded_model.predict(scaled_new_data)

print(f'Prediction for new data: {prediction}')
