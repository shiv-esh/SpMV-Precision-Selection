#!/bin/bash

source_dir="/home/shivesh/Desktop/Project/matrices_new/extract"
destination_dir="/home/shivesh/Desktop/mtx_files"

# Loop through subdirectories in the source directory
for dir in "$source_dir"/*; do
    # Loop through files in each subdirectory
    for file in "$dir"/*; do
        # Move each file to the destination directory
        cp "$file" "$destination_dir"
    done
done

echo "Files moved successfully."

