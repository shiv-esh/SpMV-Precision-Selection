import os
import subprocess
import csv

# Define the folder containing the .mtx files
input_folder = "../all_random_mtx"
output_csv = "results_random.csv"
cuda_program = "./spmv"  # Replace with the path to your compiled CUDA program

# Function to run the CUDA program on an .mtx file and capture the output
def run_cuda_program(mtx_file):
    try:
        # Run the CUDA program
        result = subprocess.run([cuda_program, mtx_file], capture_output=True, text=True)

        # Check if the program ran successfully
        if result.returncode != 0:
            print(f"Error running {cuda_program} on {mtx_file}: {result.stderr}")
            return None

        # Parse the output to extract the CSV-friendly line
        output_lines = result.stdout.splitlines()
        for line in output_lines:
            if "CSV_OUTPUT:" in line:
                # Extract the time elapsed and relative error from the output
                csv_data = line.split("CSV_OUTPUT:")[1].strip()
                return csv_data.split(',')

    except Exception as e:
        print(f"Exception occurred while running {cuda_program} on {mtx_file}: {e}")
        return None

# Get all .mtx files in the input folder
mtx_files = [f for f in os.listdir(input_folder) if f.endswith(".mtx")]

# Open the CSV file for writing
with open(output_csv, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    # Write the header
    csv_writer.writerow(["Matrix File", "Time Elapsed (ms)", "Relative Error"])

    # Iterate over each .mtx file and run the CUDA program
    for mtx_file in mtx_files:
        print(f"Processing {mtx_file}...")
        full_path = os.path.join(input_folder, mtx_file)
        result = run_cuda_program(full_path)

        if result:
            # Write the results to the CSV file
            csv_writer.writerow([mtx_file] + result)

print(f"Results saved to {output_csv}")
