# -*- coding: utf-8 -*-
"""
#######
Diabetes Disease Prediction
########
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import RFE
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle  # Import pickle for saving the model
import os # To handle file path

## Part 1: Data Loading ##


column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv('D:/SEMESTER-4/377-AI/Assignment/diabetes.csv')
print(df.head())
print(df.shape)
print(df.dtypes)

## Part 2: Data Pre-processing ##
print(df.isnull().sum())

# No need to replace '?' as this dataset typically has no such entries, but confirm the dataset first
# df.replace('?', np.nan, inplace=True)
# df = df.apply(pd.to_numeric) # convert all values to numeric type

# Handle missing values 
# df.dropna(inplace=True) # Adjust this step based on actual data exploration

# Features and Labels
X = df.drop(columns='Outcome')
y = df['Outcome']  # Outcome as a binary target is already in the correct format

## Part 3: Model Training ###
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = LogisticRegression(max_iter=5000)

model.fit(X_train_scaled, y_train)

# assign the path
destination_folder = 'D:/SEMESTER-4/377-AI/Assignment'
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)  # Create the folder if it does not exist
model_filename = os.path.join(destination_folder, 'diabetes_model.pkl')

# Save the model to disk using pickle
model_filename = 'diabetes_model.pkl'
with open(model_filename, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved to {model_filename}")

### Part 4: Model Evaluation ###

y_pred = model.predict(X_test_scaled)
### Part 5: Model Accuracy ###
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Output results
print(f"Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n", classification_rep)

### Visualization & Display ###
## Print Results to the Screen ###
# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap to visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
