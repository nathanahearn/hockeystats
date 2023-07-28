import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

#TRAINTESTSPLIT

# Load the processed player data from the CSV file
player_data_df = pd.read_csv('player_data.csv')

# Step 4: Perform train-test split
X = player_data_df[['DY-1 PPG', 'DY-1 Shots per game', 'DY PPG', 'DY Shots per game']]
y = player_data_df['DY+1 PPG']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Handle missing values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Step 6: Reset the index of y_train
y_train.reset_index(drop=True, inplace=True)

# Step 7: Fit the Multiple Linear Regression Model on the training data with imputed values
model = sm.OLS(y_train, sm.add_constant(X_train_imputed)).fit()

# Step 8: Add a new column for "Projected DY+1 PPG" based on the regression model
player_data_df['Projected DY+1 PPG'] = model.predict(sm.add_constant(X_test_imputed))

# Step 9: Evaluate the model's performance on the test set
y_pred = model.predict(sm.add_constant(X_test_imputed))
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Step 10: Print the performance metrics
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
print("Mean Absolute Error (MAE):", mae)
print("R-squared (R2) Score:", r2)

# Step 11: Create a scatter plot to visualize actual vs predicted values for the test set
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual DY+1 PPG')
plt.ylabel('Predicted DY+1 PPG')
plt.title('Actual vs Predicted DY+1 PPG')
plt.show()
