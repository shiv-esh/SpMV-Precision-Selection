import pandas as pd
import shutil
import os

# Path to the CSV file
csv_file = "/home/shivesh/Desktop/Project_SEM2/filtered_spmv test_3k.csv"  # Replace with the actual CSV file path

# Path to the source directory containing the files to be moved
source_dir = "/home/shivesh/Desktop/mtx_files"  # Replace with the actual source directory path

# Path to the destination directory where the files will be moved
destination_dir = "/home/shivesh/Desktop/Project_SEM2/Test_files_3k"  # Replace with the actual destination directory path

# Read the CSV file into a DataFrame
df = pd.read_csv(csv_file)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    # Extract the filename from the "matrix" column
    filename = row["matrix"]
    
    # Construct the full path to the source file
    source_file = os.path.join(source_dir, filename+'.mtx')
    
    # Construct the full path to the destination file
    destination_file = os.path.join(destination_dir, filename+'.mtx')
    
    # Move the file from the source directory to the destination directory
    shutil.copyfile(source_file, destination_file)
    
    print(f"Moved {filename} to {destination_dir}")

print("All files moved successfully.")

