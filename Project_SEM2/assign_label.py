import csv

# Input CSV file path
input_file_path = 'spmv random_nc.csv'

# Output CSV file path
output_file_path = 'output_label_random_nc.csv'

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
            value2 = float(row['ent-split-rt'])
            value3 = float(row['row-split-rt'])
            value4 = float(row['row-comp-rt'])
            value5 = float(row['row-dual-rt'])
          	
            # Determine the smaller value
            min_val = min([value1,value2,value3,value4,value5])
            if(min_val == value1):
            	target_value = 0
            elif(min_val == value2):
            	target_value = 1
            elif(min_val == value3):
            	target_value = 2
            elif(min_val == value4):
            	target_value = 3
            else:
            	target_value = 4
            	
            
            
            
            # Write matrix and target values to output CSV file
            writer.writerow({'matrix': row['matrix'], 'target': target_value})

print(f"Processed data written to {output_file_path}")
