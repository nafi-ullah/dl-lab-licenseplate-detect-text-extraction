#!/bin/bash

# Test script for automatic license plate recognition
echo "Activating virtual environment and running license plate detection..."

# Activate virtual environment
source venv/bin/activate

# Run the main script
python3 main.py

echo "License plate detection completed!"
echo "Check license_plates_results.csv for the results."
