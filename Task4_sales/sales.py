import pandas as pd
# Load the dataset
df = pd.read_csv('advertising.csv')
# Basic info
print(df.head())
print(df.info())
print(df.describe())
df = df.dropna()
# One-hot encode categorical columns
df = pd.get_dummies(df, drop_first=True)
X = df.drop('Sales', axis=1)
y = df['Sales']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
from sklearn.metrics import mean_squared_error, r2_score
y_pred = model.predict(X_test)
print("RMSE:", mean_squared_error(y_test, y_pred,)**0.5)
print("R² Score:", r2_score(y_test, y_pred))
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.show()
