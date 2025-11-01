#!/bin/bash

program_path="/home/shivesh/Desktop/MpSpMV/src/spmv"
files_dir="/home/shivesh/Desktop/Project/mtx_files"

for file in "$files_dir"/*; do
    "$program_path" "$file"
done

echo "All files processed."

