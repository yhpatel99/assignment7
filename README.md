# Diabetes Regression Models

This assignment demonstrates the use of three different regression models to predict diabetes progression based on the diabetes dataset from Scikit-Learn. The models used are:

1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor

## Dataset

The dataset used is the built-in diabetes dataset from Scikit-Learn. It contains 10 baseline variables, age, sex, body mass index, average blood pressure, and six blood serum measurements, obtained for each of 442 diabetes patients, as well as the target variable, a quantitative measure of disease progression one year after baseline.


### Model Training

Three regression models are initialized and trained on the training data:

- `LinearRegression()`
- `DecisionTreeRegressor(random_state=42)`
- `RandomForestRegressor(random_state=42)`

### Predictions

Predictions are made on the test data using the trained models.

### Evaluation

The models are evaluated using the following metrics:

- Mean Squared Error (MSE)
- R-squared (R²)
- Mean Absolute Error (MAE)

The evaluation metrics for each model are printed to the console.

### Conclusion

The best model is determined based on the R² score. The model with the highest R² score is considered the best model.

