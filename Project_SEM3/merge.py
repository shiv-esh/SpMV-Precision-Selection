import pandas as pd

# Read the CSV files
df1 = pd.read_csv('file1.csv')  # Replace with your first file name
df2 = pd.read_csv('file2.csv')  # Replace with your second file name

# Merge the DataFrames on the 'matrix' column
merged_df = pd.merge(df1, df2, on='matrix', how='inner')  # Use 'inner', 'outer', 'left', or 'right' for different types of joins

# Save the merged DataFrame to a new CSV
merged_df.to_csv('merged_output.csv', index=False)

print("CSV files merged successfully!")
