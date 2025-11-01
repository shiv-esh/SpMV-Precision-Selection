import pandas as pd

# Read the CSV file into a DataFrame
filename = "spmv test_3k.csv"  # Replace with the actual file name
df = pd.read_csv(filename)

# Filter rows where both "rows" and "cols" are less than or equal to 512
filtered_df = df[(df['rows'].between(500, 15000)) & (df['cols'].between(500, 15000))]

# Save the filtered data to a new CSV file
output_filename = "filtered_" + filename
filtered_df.to_csv(output_filename, index=False)

print("Filtered data saved to", output_filename)
