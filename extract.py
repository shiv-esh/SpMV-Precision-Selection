import os
import tarfile

# Directory containing the downloaded matrix files
download_dir = '/home/shivesh/Desktop/Project/matrices_new'
extract_dir = '/home/shivesh/Desktop/Project/matrices_new/extract'
# Check if the download directory exists
if not os.path.exists(download_dir):
    print(f"Directory '{download_dir}' does not exist.")
    exit()

# List all downloaded files in the directory
files = [f for f in os.listdir(download_dir) if f.endswith('.tar.gz')]

# Unzip each downloaded matrix file
for file in files:
    file_path = os.path.join(download_dir, file)
    
    # Extract the matrix file
    with tarfile.open(file_path, 'r:gz') as tar:
        tar.extractall(extract_dir)
    
    print(f"Unzipped {file} successfully.")

print("All matrix files have been unzipped.")
