import requests
import os
import pandas as pd
# List of matrix names or IDs to download
# with open('C:/Users/lenovo/Desktop/Project/names.txt', 'r') as file:
#     matrix_names = [line.strip() for line in file.readlines()]
# Replace with your matrix names or IDs

# Directory to save the downloaded matrices
# Replace with your matrix names or IDs
file_path = '/home/shivesh/Desktop/Project/names_2.txt'  # Replace with your file path
df = pd.read_csv(file_path, delimiter='\t')  # Assuming tab-separated values

# Extract matrix names from 'Name' column
matrix_names = df['Name'].tolist()
group = df['Group'].tolist()
matrix_dict = df.set_index('Name')['Group'].to_dict()

#     print(name)
# Directory to save the downloaded matrices
download_dir = '/home/shivesh/Desktop/matrices_4'

# Create the directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Base URL for the SuiteSparse Matrix Collection
base_url = 'https://suitesparse-collection-website.herokuapp.com/MM/'
proxy_host = '172.31.2.4'
proxy_port = '8080'
proxy_username = 'mse2023003'  # Optional
proxy_password = 'Shiv@1100'  # Optional
proxy_url = f'http://{proxy_host}:{proxy_port}'
proxies = {
    'http': proxy_url,
    'https': proxy_url
}
# Loop through the matrix names and download each matrix
for matrix_name,group in matrix_dict.items():
    # Construct the download URL
    download_url = base_url + group +'/'+ matrix_name + '.tar.gz'
    
    # Send a GET request to download the matrix
    response = requests.get(download_url,proxies=proxies)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Save the downloaded file
        with open(os.path.join(download_dir, matrix_name + '.tar.gz'), 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {matrix_name}.tar.gz")
    else:
        print(f"Failed to download {matrix_name}.tar.gz")

print("All matrices downloaded successfully!")

