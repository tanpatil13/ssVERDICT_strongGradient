#!/bin/bash

dir_name=$1

# Generate timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Create a directory named after the timestamp
mkdir -p "$dir_name/model_output_directory/$timestamp"

# Run the Python script, passing the timestamp as an argument
# Redirect output to a log file inside the timestamped directory
PYTHONPATH=$(pwd) python $dir_name/main.py "$timestamp" "$dir_name" > "$dir_name/model_output_directory/$timestamp/model_fitting_output_$timestamp.log" 2>&1