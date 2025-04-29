FROM ros:humble-perception

# Use NVIDIA runtime
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=all

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    libopencv-dev \
    python3-opencv \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Fix NumPy version to avoid compatibility issues with ROS/CV Bridge
# Must be installed before other packages that might pull in newer numpy
RUN pip3 install --no-cache-dir 'numpy==1.24.3'

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir torch==2.0.1+cu118 torchvision==0.15.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# Install Ultralytics YOLO
RUN pip3 install --no-cache-dir ultralytics>=8.0.0

# Install ONNX runtime with GPU support
RUN pip3 install --no-cache-dir onnx>=1.14.0 onnxruntime-gpu>=1.16.0 onnx-simplifier>=0.4.35

# Install ROS2 Python dependencies
RUN pip3 install --no-cache-dir opencv-python>=4.7.0 pyyaml>=6.0 matplotlib>=3.7.0 pillow>=9.4.0 tqdm>=4.65.0

# Verify numpy version and reinstall to force the correct version if needed
RUN pip3 uninstall -y numpy && pip3 install --no-cache-dir 'numpy==1.24.3'

# Create app directory
WORKDIR /app

# Copy only the required files
COPY pallet_inference_node.py /app/
COPY trained_models/ /app/trained_models/

# Make Python script executable
RUN chmod +x /app/pallet_inference_node.py

# Setup entrypoint
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Create a wrapper script to ask for model type
RUN echo '#!/bin/bash\n\
echo "Select model type to run:"\n\
echo "1) Unoptimized PyTorch models (default)"\n\
echo "2) Optimized ONNX models"\n\
read -p "Enter choice [1-2] (default: 1): " choice\n\
\n\
if [ "$choice" = "2" ]; then\n\
    echo "Starting with optimized ONNX models..."\n\
    python3 /app/pallet_inference_node.py --ros-args -p use_onnx:=true -p onnx_detection_model_path:=/app/trained_models/optimized/best_detect_fp16.onnx -p onnx_segmentation_model_path:=/app/trained_models/optimized/best_segment_fp16.onnx\n\
else\n\
    echo "Starting with unoptimized PyTorch models..."\n\
    python3 /app/pallet_inference_node.py\n\
fi\n\
' > /app/run_models.sh

RUN chmod +x /app/run_models.sh

# By default, use unoptimized PyTorch models
ENV USE_ONNX=false

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/app/run_models.sh"] 