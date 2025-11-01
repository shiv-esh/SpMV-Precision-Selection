# Define the filename
filename = "names_2.txt"  # Replace with the actual file name

# Open the input file for reading
with open(filename, "r") as file:
    lines = file.readlines()

# Function to convert string representation of numbers with commas to integers
def convert_to_int_with_commas(value_str):
    return int(value_str.replace(",", ""))

# Filter rows where 'Rows' value is greater than 'Cols'
filtered_lines = []
for line in lines:
    cols = line.split("\t")
    if len(cols) >= 5:  # Ensure there are at least 5 columns
        rows_value = convert_to_int_with_commas(cols[3])
        cols_value = convert_to_int_with_commas(cols[4])
        if rows_value < cols_value:
            filtered_lines.append(line)

# Save the filtered data to a new file
output_filename = "filtered_" + filename
with open(output_filename, "w") as file:
    file.writelines(filtered_lines)

print("Filtered data saved to", output_filename)

