#!/bin/bash

# Make sure we exit on any error
set -e

# Check if virtual environment exists, if not create it
if [ ! -d "hailo_env" ]; then
    echo "Creating virtual environment..."
    python3 -m venv hailo_env
fi

# Activate virtual environment
source hailo_env/bin/activate

# Install required packages
echo "Installing required packages..."
pip install --upgrade pip
pip install hailo_sdk
pip install hailo-model-zoo
pip install onnx2pytorch
pip install numpy
pip install onnx

# Clone Hailo Model Zoo if not already present
if [ ! -d "hailo_model_zoo" ]; then
    echo "Cloning Hailo Model Zoo..."
    git clone https://github.com/hailo-ai/hailo_model_zoo.git
fi

# Create directory for compiled model
mkdir -p compiled_model

# Export YOLOv5 to ONNX format
echo "Exporting YOLOv5 model to ONNX..."
python3 - << EOF
import torch
from pathlib import Path

# Load your trained model
model = torch.load('yolov5/runs/train/exp9/weights/best.pt')
model.eval()

# Export to ONNX
input_names = ['images']
output_names = ['output']
torch.onnx.export(
    model,
    torch.randn(1, 3, 416, 416),  # Adjust size based on your model
    'compiled_model/model.onnx',
    verbose=True,
    input_names=input_names,
    output_names=output_names,
    opset_version=12
)
EOF

# Compile model using Hailo CLI tools
echo "Compiling model for Hailo..."
hailomz compile \
    --model compiled_model/model.onnx \
    --target hailo8 \
    --output-name compiled_model/detection_net.hef \
    --optimization performance \
    --input-shape "1,3,416,416"

# Optimize network configuration
echo "Optimizing network configuration..."
hailomz optimize \
    --model compiled_model/detection_net.hef \
    --output compiled_model/detection_net_optimized.hef \
    --target hailo8

echo "Compilation complete! The compiled model is at: compiled_model/detection_net_optimized.hef"