
# A library for programmatic plot generation.
import matplotlib.pyplot as plt

# A library for data manipulation and analysis.
import pandas as pd

# LinearRegression from sklearn.
from sklearn.linear_model import LinearRegression

# A library for numerical computing.
import numpy as np

# train_test_split from sklearn.
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

import w2_unittest

path = "data/tvmarketing.csv"

### START CODE HERE ### (~ 1 line of code)
adv = pd.read_csv(path)
### END CODE HERE ###

# Print some part of the dataset.
adv.head(5)

w2_unittest.test_load_data(adv)

adv.plot(x='TV', y='Sales', kind='scatter', c='black')

X = adv['TV'].values
Y = adv['Sales'].values # Target

m_numpy, b_numpy = np.polyfit(X.flatten(), Y, deg=1)

print(f"Linear regression with NumPy. Slope: {m_numpy}. Intercept: {b_numpy}")

# This is organised as a function only for grading purposes.
def pred_numpy(m, b, X):

    Y = m * X + b

    return Y

X_pred = np.array([50, 120, 280])
Y_pred_numpy = pred_numpy(m_numpy, b_numpy, X_pred)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using NumPy linear regression:\n{Y_pred_numpy}")

w2_unittest.test_pred_numpy(pred_numpy)

lr_sklearn = LinearRegression()

print(f"Shape of X array: {X.shape}")
print(f"Shape of Y array: {Y.shape}")

try:
    lr_sklearn.fit(X, Y)
except ValueError as err:
    print(err)


X_sklearn = X[:, np.newaxis]
Y_sklearn = Y[:, np.newaxis]

print(f"Shape of new X array: {X_sklearn.shape}")
print(f"Shape of new Y array: {Y_sklearn.shape}")

X_train, X_test, Y_train, Y_test = train_test_split(X_sklearn, Y_sklearn, test_size=0.2, random_state=42)

lr_sklearn.fit(X_train, Y_train) #Insert proper arguments fro training asper step 1


Y_pred = lr_sklearn.predict(X_test)#use test data from X from step 1 above)


#Insert your code here
rmse =  (mean_squared_error(Y_test,Y_pred))**0.5
print("Root Mean Square Error:", rmse)

rf_model = RandomForestRegressor(random_state=42)
dt_model = DecisionTreeRegressor(random_state=42)

# Step 3: Fit the models to the training data
rf_model.fit(X_train, Y_train.ravel())
dt_model.fit(X_train, Y_train.ravel())

# Step 4: Make predictions for each model
rf_pred = rf_model.predict(X_test)
dt_pred = dt_model.predict(X_test)

# Step 5: Calculate the RMSE for each model
rf_rmse = mean_squared_error(Y_test, rf_pred) ** 0.5
dt_rmse = mean_squared_error(Y_test, dt_pred) ** 0.5

# Step 6: Store the RMSE values in a dictionary
model_rank = {
    'Linear Regression': rmse,
    'Random Forest': rf_rmse,
    'Decision Trees': dt_rmse
}


# Print the ranked models
print(model_rank)

# Step 6: Sort the models by RMSE from best to worst
model_rank_sorted = dict(sorted(model_rank.items(), key=lambda item: item[1]))

# Step 7: Print the ranked models and their RMSEs
print("Model Rank and Associated RMSE:")
for model, rmse_value in model_rank_sorted.items():
    print(f"{model}: {rmse_value}")


### START CODE HERE ### (~ 1 line of code)
lr_sklearn.fit(X_sklearn, Y_sklearn)
### END CODE HERE ###

m_sklearn = lr_sklearn.coef_
b_sklearn = lr_sklearn.intercept_

print(f"Linear regression using Scikit-Learn. Slope: {m_sklearn}. Intercept: {b_sklearn}")

w2_unittest.test_sklearn_fit(lr_sklearn)

# This is organised as a function only for grading purposes.
def pred_sklearn(X, lr_sklearn):
    ### START CODE HERE ### (~ 2 lines of code)
    X_2D = X[:, np.newaxis]
    Y = lr_sklearn.predict(X_2D)
    ### END CODE HERE ###

    return Y


Y_pred_sklearn = pred_sklearn(X_pred, lr_sklearn)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")


w2_unittest.test_sklearn_predict(pred_sklearn, lr_sklearn)


fig, ax = plt.subplots(1,1,figsize=(8,5))
ax.plot(X, Y, 'o', color='black')
ax.set_xlabel('TV')
ax.set_ylabel('Sales')

ax.plot(X, m_sklearn[0][0]*X+b_sklearn[0], color='red')
ax.plot(X_pred, Y_pred_sklearn, 'o', color='blue')


X_norm = (X - np.mean(X))/np.std(X)
Y_norm = (Y - np.mean(Y))/np.std(Y)


def E(m, b, X, Y):
    N = len(X)
    errors = ((m * X + b) - Y)  # Errors between predicted and real values
    cost = np.sum(errors ** 2) / (2 * N)  # Sum of squares cost
    return cost


def dEdm(m, b, X, Y):
    errors = ((m * X + b) - Y) # Calculate the error between predicted and actual values
    res = np.mean(errors * X)  # Correctly scaled gradient with negative sign
    return res

def dEdb(m, b, X, Y):
    errors = ((m * X + b) - Y)  # Calculate the error between predicted and actual values
    res = np.mean(errors)  # Correctly scaled gradient with negative sign
    return res


print(dEdm(0, 0, X_norm, Y_norm))
print(dEdb(0, 0, X_norm, Y_norm))
print(dEdm(1, 5, X_norm, Y_norm))
print(dEdb(1, 5, X_norm, Y_norm))


w2_unittest.test_partial_derivatives(dEdm, dEdb, X_norm, Y_norm)

def gradient_descent(dEdm, dEdb, m, b, X, Y, learning_rate = 0.001, num_iterations = 1000, print_cost=False):
    for iteration in range(num_iterations):
        ### START CODE HERE ### (~ 2 lines of code)
        m_new = m - learning_rate * dEdm(m, b, X, Y)
        b_new = b - learning_rate * dEdb(m, b, X, Y)
        ### END CODE HERE ###
        m = m_new
        b = b_new
        if print_cost:
            print (f"Cost after iteration {iteration}: {E(m, b, X, Y)}")

    return m, b

print(gradient_descent(dEdm, dEdb, 0, 0, X_norm, Y_norm))
print(gradient_descent(dEdm, dEdb, 1, 5, X_norm, Y_norm, learning_rate = 0.01, num_iterations = 10))


w2_unittest.test_gradient_descent(gradient_descent, dEdm, dEdb, X_norm, Y_norm)

m_initial = 0; b_initial = 0; num_iterations = 30; learning_rate = 1.2
m_gd, b_gd = gradient_descent(dEdm, dEdb, m_initial, b_initial,
                              X_norm, Y_norm, learning_rate, num_iterations, print_cost=True)

print(f"Gradient descent result: m_min, b_min = {m_gd}, {b_gd}")


X_pred = np.array([50, 120, 280])
# Use the same mean and standard deviation of the original training array X
X_pred_norm = (X_pred - np.mean(X))/np.std(X)
Y_pred_gd_norm = m_gd * X_pred_norm + b_gd
# Use the same mean and standard deviation of the original training array Y
Y_pred_gd = Y_pred_gd_norm * np.std(Y) + np.mean(Y)

print(f"TV marketing expenses:\n{X_pred}")
print(f"Predictions of sales using Scikit_Learn linear regression:\n{Y_pred_sklearn.T}")
print(f"Predictions of sales using Gradient Descent:\n{Y_pred_gd}")


#What imports do we need for Fast api
import asyncio
import uvicorn
from typing import Annotated, List
from fastapi import FastAPI, Depends, HTTPException, status, Path, Query
from pydantic import BaseModel, Field

#insert fast api decorator

app = FastAPI()

class SalesPredictionInput(BaseModel):
    features: List[float]

# Create an endpoint for linear regression predictions
@app.post("/predict-sales")
def predict_sales(input_data: SalesPredictionInput):
    # Extract the feature data from the input
    features = input_data.features

    feature_array = np.array(features).reshape(-1, 1)

    predictions = lr_model.predict(feature_array)

    return {"predicted_sales": predictions.tolist()}

async def main():
    config = uvicorn.Config(app)
    server = uvicorn.Server(config)
    await server.serve()

if __name__ == "__main__":
    asyncio.run(main())
