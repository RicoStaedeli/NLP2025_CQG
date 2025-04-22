#!/bin/bash

# Create folders
mkdir -p Datasets
mkdir -p Evaluation/Evaluation
mkdir -p Evaluation/Results
mkdir -p Models

echo "Project structure created successfully."

# Define model directory
MODEL_DIR="Models/Meta-Llama-3-8B-Instruct"

# Download model if not already present
if [ ! -d "$MODEL_DIR" ]; then
    echo "Downloading Meta-Llama-3-8B-Instruct model..."
    huggingface-cli download meta-llama/Meta-Llama-3-8B-Instruct \
        --exclude "original/*" \
        --local-dir "$MODEL_DIR"
else
    echo "Model already exists at $MODEL_DIR. Skipping download."
fi

echo "All models downloaded."

# Install Python requirements if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    # python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Dependencies installed."
else
    echo "requirements.txt not found. Skipping Python dependencies installation."
fi

echo "Project setup done."