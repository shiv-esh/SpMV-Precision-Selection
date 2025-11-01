import csv

# Input CSV file path
input_file_path = 'filtered_spmv test_small.csv'

# Output CSV file path
output_file_path = 'output_small1.csv'

# Open input CSV file for reading
with open(input_file_path, 'r') as infile:
    reader = csv.DictReader(infile)
    
    # Open output CSV file for writing
    with open(output_file_path, 'w', newline='') as outfile:
        fieldnames = ['matrix', 'target']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        # Write header to output CSV file
        writer.writeheader()
        
        # Iterate over each row in the input CSV file
        for row in reader:
            # Convert column1 and column2 values to float for comparison
            value1 = float(row['ent-base-rt'])
            value2 = float(row['row-comp-rt'])
            
            # Determine the smaller value
            target_value = 0
            if(value1 > value2):
                target_value = 1
            
            
            # Write matrix and target values to output CSV file
            writer.writerow({'matrix': row['matrix'], 'target': target_value})

print(f"Processed data written to {output_file_path}")
