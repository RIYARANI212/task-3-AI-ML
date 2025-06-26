# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Step 2: Create or Load Dataset
# Example Dataset (you can replace this with pd.read_csv("your_data.csv"))
data = {
    'Hours_Studied': [2, 4, 6, 8, 10, 12, 14],
    'Sleep_Hours': [8, 7, 6, 6, 5, 5, 4],
    'Marks': [55, 60, 65, 70, 75, 80, 85]
}
df = pd.DataFrame(data)
print("Dataset:\n", df)

# Step 3: Split Dataset
X = df[['Hours_Studied', 'Sleep_Hours']]  # Independent Variables (Multiple Regression)
y = df['Marks']                          # Dependent Variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Predict and Evaluate
y_pred = model.predict(X_test)

# Evaluation Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Coefficients:", model.coef_)
print("Model Intercept:", model.intercept_)
print("MAE:", mae)
print("MSE:", mse)
print("R² Score:", r2)

# Step 6: Visualization (Simple Linear Regression Plot for 'Hours_Studied')
plt.scatter(df['Hours_Studied'], df['Marks'], color='blue', label='Actual Marks')
plt.plot(df['Hours_Studied'], model.predict(df[['Hours_Studied', 'Sleep_Hours']]), color='red', label='Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Marks')
plt.title('Linear Regression - Marks vs Hours Studied')
plt.legend()
plt.grid(True)
plt.show()
