from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Attendance': [30, 40, 50, 60, 65, 80, 85, 90],
    'Pass': [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied', 'Attendance']]
y = df['Pass']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = RandomForestClassifier(n_estimators=10, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(" Accuracy:", accuracy_score(y_test, y_pred))
print(" Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print(" Classification Report:\n", classification_report(y_test, y_pred))
