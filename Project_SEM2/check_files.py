import os
import pandas as pd

def read_file_list_from_csv(csv_file, column_name):
  """Reads a list of files from a specified column in a CSV file.

  Args:
    csv_file: The path to the CSV file.
    column_name: The name of the column containing the file list.

  Returns:
    A list of file names.
  """

  df = pd.read_csv(csv_file)
  file_list = df[column_name].tolist()
  return file_list

# Example usage
csv_file_path = "/home/shivesh/Desktop/Project_SEM2/filtered_spmv test_209.csv"
column_name = "matrix"

file_list = read_file_list_from_csv(csv_file_path, column_name)

def check_files_in_directory(directory_path, file_list):
  """Checks if a list of files are present in a given directory.

  Args:
    directory_path: The path to the directory to check.
    file_list: A list of file names to check for.

  Returns:
    A list of file names that are not present in the directory.
  """

  existing_files = set(os.listdir(directory_path))
  missing_files = []
  for file in file_list:
    filename = file+'.mtx'
    if filename not in existing_files:
      missing_files.append(filename)
  return missing_files

# Example usage
directory_path = "/home/shivesh/Desktop/Project_SEM2/Test_files_all"


missing_files = check_files_in_directory(directory_path, file_list)

if missing_files:
  print(len(missing_files))
else:
  print("All files are present in the directory.")
