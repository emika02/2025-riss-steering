#!/bin/bash
set -e

# 1. THE BRIDGE: Use the full system path to the conda program
CONDA_PATH="/opt/miniconda3/bin/conda"

# 2. THE PATHS: Use the absolute path for the YAML and Target
YML_FILE="/zfsauton2/home/ekaczmar/representations-in-tsfms-main/representations-in-tsfms-main/environment.yml"
TARGET_PATH="/zfsauton/scratch/ekaczmar/envs/reps-tsfm"

echo "Attempting to bridge zfsauton2 and zfsauton..."

# Create the environment using the absolute path to the executable
$CONDA_PATH env create -p "$TARGET_PATH" -f "$YML_FILE"

# Source the activator so we can use 'conda activate' inside this script
source "/opt/miniconda3/etc/profile.d/conda.sh"

# Activate the new environment on the other drive
conda activate "$TARGET_PATH"

# Install the final library
pip install momentfm==0.1.4

echo "Success! Environment created on the zfsauton scratch drive."