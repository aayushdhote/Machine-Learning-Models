#Logistic Regression -Pass or Fail

#Import libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

#Created a sample dataset
data = {
    'Study Hours': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Sleep Hours': [8, 7, 6, 6, 5, 5, 4, 4, 3, 3],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1, 1, 1]  # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)
print("📘 Dataset:\n", df, "\n")

#Split features and target
X = df[['Study Hours', 'Sleep Hours']]
y = df['Pass']

#Split data for training/testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

#Predict on test data
y_pred = model.predict(X_test)
print("Predictions:", y_pred)
print("Actual:", np.array(y_test))

#Evaluate performance
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)

print("\n Accuracy:", acc)
print(" Confusion Matrix:\n", cm)
print(" Classification Report:\n", cr)

#Predict for a new student
new_data = [[int(input("Enter the hours of Study")), int(input("Enter the hours of sleep"))]]  # 7 hours study, 4 hours sleep
prediction = model.predict(new_data)
result = "Pass" if prediction[0] == 1 else "Fail"
print("\n🎯 Predicted Result for new student:", result)
