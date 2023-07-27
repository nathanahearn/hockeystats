import pandas as pd
from scipy import stats

# Load the dataset into a DataFrame called 'players'
players = pd.read_csv(r"YOURCSVLOCATION\YOURCSV.csv")

# Filter players who were drafted and not drafted
playersdrafted = players[players['DRAFT TEAM'] != '-']
notdrafted = players[players['DRAFT TEAM'] == '-']

# Set the significance level
significance_level = 0.05

# Perform the t-test
t_stat, p_value = stats.ttest_ind(playersdrafted['TP/GP'], notdrafted['TP/GP'], equal_var=False)

# Print the t-statistic and p-value
print("T-Statistic:", t_stat)
print("P-Value:", p_value)
