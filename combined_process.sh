#!/bin/bash

# Directory containing the JSON files
input_folder="anlp-project/run-5"

# Iterate through all JSON files in the input folder
for file in "$input_folder"/*.json; 
do
    # Run the Python script on each file
    python scoring.py "$file"
done