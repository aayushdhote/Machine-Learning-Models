from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

data = pd.DataFrame({
    "Hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "Marks": [30, 40, 50, 55, 65, 70, 75, 85]})

X = data[["Hours"]]   # input feature
y = data["Marks"]     # output label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

pred = model.predict(X_test)
print("Predictions:", pred)

print("Error:", mean_squared_error(y_test, pred))
