import pandas as pd

players = pd.read_csv(r"YOURCSVLOCATION\QMJHLData.csv")

# Filter the rows where 'TP/GP' and 'DY/+' are not null
filtered_players = players.dropna(subset=['TP/GP', 'DY+/-'])

# Define a custom sorting order for 'DY/+'
custom_sort_order = ['DY-2', 'DY-1', 'DY', 'DY+1', 'DY+2', 'DY+3']

# Sort the DataFrame by 'DY/+' using the custom sorting order
filtered_players['DY+/-'] = pd.Categorical(filtered_players['DY+/-'], categories=custom_sort_order, ordered=True)
filtered_players = filtered_players.sort_values('DY+/-')

# Plot 'TP/GP' against 'DY/+'
plt.scatter(filtered_players['DY+/-'], filtered_players['TP/GP'], alpha=0.5)
plt.xlabel('Draft Year Plus-Minus (DY+/-)')
plt.ylabel('Points Per Game (TP/GP)')
plt.title('TP/GP vs. DY+/-')

# Set the x-axis labels in the custom sorting order
plt.xticks(filtered_players['DY+/-'].cat.codes, filtered_players['DY+/-'], rotation=45)

plt.grid(True)
plt.show()

