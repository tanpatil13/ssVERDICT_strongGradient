#!/bin/bash

# Generate timestamp
timestamp=$(date +%Y%m%d_%H%M%S)

# Create a directory named after the timestamp
mkdir -p "model_output_directory/$timestamp"

# Run the Python script, passing the timestamp as an argument
# Redirect output to a log file inside the timestamped directory
python main.py "$timestamp" > "model_output_directory/$timestamp/model_fitting_output_$timestamp.log" 2>&1