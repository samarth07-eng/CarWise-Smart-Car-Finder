import pandas as pd
from pathlib import Path

# Locate cars.csv in the project folder
file_path = Path(__file__).parent / "cars.csv"

# Load existing car data
df = pd.read_csv(file_path)

# Add a maintenance column if it doesn't exist
if "maintenance" not in df.columns:
    df["maintenance"] = 3

# Maintenance levels:
# 1 = Very Low
# 2 = Low
# 3 = Medium
# 4 = High
# 5 = Very High

# Save the updated dataset
df.to_csv(file_path, index=False)

print("Maintenance column added successfully!")
print("Updated cars.csv")
print(df.head())