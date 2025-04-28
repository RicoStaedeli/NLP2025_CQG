#!/bin/bash

# Create main Models folder
mkdir -p Models
echo "Project structure created successfully."

# Define models: "HuggingFace repo" "local directory"
models=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct ../Models/Meta-Llama-3.1-8B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct ../Models/Qwen2.5-7B-Instruct"
)

# Loop through models and download if not already present
for model_info in "${models[@]}"; do
    repo=$(echo "$model_info" | awk '{print $1}')
    local_dir=$(echo "$model_info" | awk '{print $2}')

    if [ ! -d "$local_dir" ]; then
        echo "Downloading model from $repo into $local_dir..."
        huggingface-cli download "$repo" \
            --exclude "original/*" \
            --local-dir "$local_dir"
    else
        echo "Model already exists at $local_dir. Skipping download."
    fi
done

echo "All models processed."

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