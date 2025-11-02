#To run this Model You Can Use jypter Notebook or colab for better Understanding the visuals and insights 

#Importing libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

#Created a basic dataset for trainig the model
data = {
    'Size (sqft)': [1000, 1500, 1800, 2400, 3000, 3500, 4000, 4200, 5000, 6000],
    'Bedrooms': [2, 3, 3, 4, 4, 5, 5, 5, 6, 7],
    'Age (years)': [10, 8, 5, 4, 3, 2, 2, 1, 1, 1],
    'Price (Lakh ₹)': [50, 65, 80, 100, 120, 140, 160, 170, 190, 210]
}

df = pd.DataFrame(data)
print("House Dataset:\n", df, "\n")

#Giving the Spliting features and target
X = df[['Size (sqft)', 'Bedrooms', 'Age (years)']]
y = df['Price (Lakh ₹)']

#Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Training the model
model = LinearRegression()
model.fit(X_train, y_train)

#Making predictions
y_pred = model.predict(X_test)
print("Predicted Prices:", y_pred)
print("Actual Prices:", np.array(y_test))

#Now Evaluating the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n Model Evaluation:")
print("Mean Squared Error:", mse)
print("R² Score:", r2)

#Lets Predict a new house price
new_house = [[2800, 4, 3]]  # Size=2800 sqft, 4 Bedrooms, 3 years old
predicted_price = model.predict(new_house)
print("\n Predicted Price for new house:", predicted_price[0], "Lakh ₹")

#View feature importance
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n Feature Coefficients:\n", coefficients)

#Visualization for better understanding the insights

# We'll fix Bedrooms=4, Age=3 and vary only Size to see the price change
size_range = np.linspace(1000, 6000, 50)  # from 1000 to 6000 sqft
bedrooms = np.full_like(size_range, 4)    # keep 4 bedrooms constant
age = np.full_like(size_range, 3)         # keep 3 years old constant

#Combining all into a single input for prediction
X_visual = np.column_stack((size_range, bedrooms, age))

# Predict price for each house size
y_visual = model.predict(X_visual)

#Ploted actual data points
plt.scatter(df['Size (sqft)'], df['Price (Lakh ₹)'], color='blue', label='Actual Prices')

plt.plot(size_range, y_visual, color='red', linewidth=2, label='Regression Line (Predicted)')
plt.title("House Price Prediction vs Size")
plt.xlabel("House Size (sqft)")
plt.ylabel("Price (Lakh ₹)")
plt.legend()
plt.grid(True)
plt.show()
