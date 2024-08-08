import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('salary.csv')
X = data['YearsExperience'].values.reshape(-1, 1)
Y = data['Salary']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)

r2 = r2_score(Y_test, Y_pred)
slope = model.coef_[0]
intercept = model.intercept_
print(f"R-squared Score: {r2:.2f}")
print(f"Regression Equation: Salary = {slope:.2f} * YearsExperience + {intercept:.2f}")

plt.scatter(X_train, Y_train, color='blue', label='Training data')
plt.plot(X, model.predict(X), color='green', label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Training Data')
plt.legend()
plt.show()

plt.scatter(X_test, Y_test, color='red', label='Actual test data')
plt.scatter(X_test, Y_pred, color='purple', marker='x', s=100, label='Predicted test data')
for X, Y_true, Y_pred in zip(X_test, Y_test, Y_pred):
    plt.plot([X, X], [Y_true, Y_pred], color='0black')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Test Data')
plt.legend()
plt.show()