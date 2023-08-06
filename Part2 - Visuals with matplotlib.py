import pandas as pd
import matplotlib.pyplot as plt

# Conver the CSV into a DataFrame for analysis
players = pd.read_csv(r"YOURCSVLOCATION\YOURCSV.csv")

# Define a custom sorting order for 'DY/+'
custom_sort_order = ['DY-2', 'DY-1', 'DY', 'DY+1', 'DY+2', 'DY+3']

# Sort the DataFrame by 'DY/+' using the custom sorting order
players['DY+/-'] = pd.Categorical(players['DY+/-'], categories=custom_sort_order, ordered=True)
players = players.sort_values('DY+/-')

# Plot 'TP/GP' against 'DY/+'
plt.scatter(players['DY+/-'], players['TP/GP'], alpha=0.5)
plt.xlabel('Draft Year Plus-Minus (DY+/-)')
plt.ylabel('Points Per Game (TP/GP)')
plt.title('TP/GP vs. DY+/-')

# Set the x-axis labels in the custom sorting order
plt.xticks(players['DY+/-'].cat.codes, players['DY+/-'], rotation=45)

plt.grid(True)
plt.show()
