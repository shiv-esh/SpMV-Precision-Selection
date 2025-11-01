import pandas as pd

# Read the CSV files
df1 = pd.read_csv('file1.csv')  # Replace with your first file name
df2 = pd.read_csv('file2.csv')  # Replace with your second file name

# Check for unique values in the 'matrix' column to debug
print("Unique values in df1['matrix']:", df1['matrix'].unique())
print("Unique values in df2['matrix']:", df2['matrix'].unique())

# Clean up the 'matrix' column by stripping spaces and converting to lowercase
df1['matrix'] = df1['matrix'].str.strip().str.lower()
df2['matrix'] = df2['matrix'].str.strip().str.lower()

# Merge the DataFrames on the 'matrix' column
merged_df = pd.merge(df1, df2, on='matrix', how='inner')  # Change 'inner' to 'outer', 'left', or 'right' if needed

# Check the first few rows of the merged data
print(merged_df.head())

# Save the merged DataFrame to a new CSV
merged_df.to_csv('merged_output.csv', index=False)

print("CSV files merged successfully!")

