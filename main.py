import pandas as pd

players = pd.read_csv(r"C:\Users\nahearn\Desktop\Hockey Project\QMJHLData.csv")

# Print shape of players
print(players.shape)

# Check for duplicates
print(players.dropduplicates().shape)

# Print player count by year
print(players["SEASON"].value_counts().sort_values(ascending=False))

# Filter the rows where 'DRAFT TEAM' is not a dash
players_no_dash_draft_team = players[players['DRAFT TEAM'] != '-']

# Count the occurrences of amount of drafted players in each season
season_counts = players_no_dash_draft_team["SEASON"].value_counts().sort_index(ascending=False)

print(season_counts)


