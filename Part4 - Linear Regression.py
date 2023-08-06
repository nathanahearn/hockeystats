import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

players = pd.read_csv(players = pd.read_csv(r"YOURCSVLOCATION\YOURCSV.csv")

# Step 1: Add a new column for "Shots Per Game" (SHOTS/GP)
players['Shots Per Game'] = players['SHOTS'] / players['GP']

# Step 2: Prepare the Data and Drop Rows with Irrelevant DY/+ Values
valid_dy_values = ['DY-1', 'DY', 'DY+1']

# Step 3: Prepare the Data and Drop Rows with Irrelevant DY/+ Values
data = players.groupby('NAME').filter(lambda x: set(x['DY+/-']) >= set(valid_dy_values))

player_count = data.shape[0]  # Get the number of rows (players) in the DataFrame

print("Total number of players after Step 2:", player_count)

# Step 4: Create a new DataFrame containing relevant data for each player
player_data_df = pd.DataFrame(columns=[
    'NAME', 'DY-1 PPG', 'DY-1 Shots per game', 'DY PPG', 'DY Shots per game', 'DY+1 PPG', 'DY+1 Shots per game'
])

for player_name in data['NAME'].unique():
    player_data = data[data['NAME'] == player_name]
    dy_minus1_data = player_data[player_data['DY+/-'] == 'DY-1'].iloc[0]
    dy_data = player_data[player_data['DY+/-'] == 'DY'].iloc[0]
    dy_plus1_data = player_data[player_data['DY+/-'] == 'DY+1'].iloc[0]

    player_data_df = pd.concat([player_data_df, pd.DataFrame({
        'NAME': [player_name],
        'DY-1 PPG': [dy_minus1_data['TP/GP']],
        'DY-1 Shots per game': [dy_minus1_data['Shots Per Game']],
        'DY PPG': [dy_data['TP/GP']],
        'DY Shots per game': [dy_data['Shots Per Game']],
        'DY+1 PPG': [dy_plus1_data['TP/GP']],
        'DY+1 Shots per game': [dy_plus1_data['Shots Per Game']],
    })], ignore_index=True)

# Save the DataFrame to a new CSV file
player_data_df.to_csv('player_data.csv', index=False)

# Step 5: Prepare the Data for Regression
independent_vars = player_data_df[['DY-1 PPG', 'DY-1 Shots per game', 'DY PPG', 'DY Shots per game']]
dependent_var = player_data_df['DY+1 PPG']

# Add constant column for intercept
independent_vars = sm.add_constant(independent_vars)

# Step 6: Fit the Multiple Linear Regression Model
model = sm.OLS(dependent_var, independent_vars).fit()

# Convert all columns in independent_vars to numeric data type
independent_vars = independent_vars.apply(pd.to_numeric, errors='coerce')

# Convert 'DY+1 PPG' column to numeric
dependent_var = pd.to_numeric(dependent_var)

# Step 7: Add a new column for "Projected DY+1 PPG" based on the regression model
player_data_df['Projected DY+1 PPG'] = model.predict(independent_vars)

# Print the updated DataFrame
print(player_data_df)

# Save the DataFrame to a new CSV file
player_data_df.to_csv('player_data_with_projections.csv', index=False)

# Step 8: Print the Regression Results
print(model.summary())

# Step 9: Create a scatter plot to visualize actual vs projected values
plt.scatter(dependent_var, model.predict(), alpha=0.5)
plt.xlabel('Actual DY+1 PPG')
plt.ylabel('Predicted DY+1 PPG')
plt.title('Actual vs Predicted DY+1 PPG')
plt.show()

# Step 10: Create a Fitted vs. Residuals Plot
plt.scatter(model.predict(), model.resid, alpha=0.5)
plt.axhline(0, color='red', linestyle='dashed')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Fitted vs. Residuals Plot')
plt.show()

# Step 11: Print the statistics for evaluating the model's effectiveness
print("Mean Squared Error (MSE):", model.mse_model)
print("Root Mean Squared Error (RMSE):", model.mse_resid ** 0.5)

# Step 12: Calculate Variance Inflation Factor (VIF) for each predictor variable
vif_data = independent_vars.copy()
vif_data.drop(columns=['const'], inplace=True)  # Drop the constant column
vif = pd.DataFrame()
vif["Features"] = vif_data.columns
vif["VIF"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
