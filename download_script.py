import requests
import os
import pandas as pd
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# File path to your matrix names or IDs
file_path = 'names.txt'  # Replace with your file path
df = pd.read_csv(file_path, delimiter='\t')  # Assuming tab-separated values

# Create a dictionary with matrix names and groups
matrix_dict = df.set_index('Name')['Group'].to_dict()

# Directory to save the downloaded matrices
download_dir = 'matrices_new'

# Create the directory if it doesn't exist
if not os.path.exists(download_dir):
    os.makedirs(download_dir)

# Base URL for the SuiteSparse Matrix Collection
base_url = 'https://suitesparse-collection-website.herokuapp.com/MM/'

# Retry settings
retry_strategy = Retry(
    total=5,  # Retry up to 5 times
    backoff_factor=1,  # Wait 1, 2, 4, 8, 16 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504],  # Retry on specific status codes
)
adapter = HTTPAdapter(max_retries=retry_strategy)
http = requests.Session()
http.mount("http://", adapter)
http.mount("https://", adapter)

# Loop through the matrix names and download each matrix
for matrix_name, group in matrix_dict.items():
    # Construct the download URL
    download_url = base_url + group + '/' + matrix_name + '.tar.gz'
    
    try:
        # Send a GET request to download the matrix
        response = http.get(download_url, timeout=10)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Save the downloaded file
            with open(os.path.join(download_dir, matrix_name + '.tar.gz'), 'wb') as file:
                file.write(response.content)
            print(f"Downloaded {matrix_name}.tar.gz")
        else:
            print(f"Failed to download {matrix_name}.tar.gz: Status Code {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"Failed to download {matrix_name}.tar.gz: {e}")

print("All matrices downloaded successfully!")

