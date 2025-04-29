#!/bin/bash
set -e

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==== Pallet Detection - Model Optimization and Docker Setup ====${NC}"

# Step 1: Create optimized_models directory if it doesn't exist
if [ ! -d "optimized_models" ]; then
  mkdir -p optimized_models
  echo -e "${YELLOW}Created optimized_models directory${NC}"
fi

# Step 2: Run the model optimization script if needed
if [ ! -f "optimized_models/best_detect_fp16.onnx" ] || [ ! -f "optimized_models/best_segment_fp16.onnx" ]; then
  echo -e "\n${GREEN}==== Step 1: Optimizing Models ====${NC}"
  echo -e "${YELLOW}Converting PyTorch models to ONNX format and applying optimizations...${NC}"
  python3 onnx_transform.py --quantize
  echo -e "${GREEN}Models optimized successfully!${NC}"
else
  echo -e "\n${GREEN}==== Step 1: Using Existing Optimized Models ====${NC}"
  echo -e "${YELLOW}Found existing optimized models in optimized_models/ directory.${NC}"
  echo -e "${YELLOW}Skipping model optimization step.${NC}"
fi

# Step 3: Build the Docker image
echo -e "\n${GREEN}==== Step 2: Building Docker Image ====${NC}"
echo -e "${YELLOW}Building Docker image with ROS2 and NVIDIA support...${NC}"
docker build -t pallet_detection:latest .
echo -e "${GREEN}Docker image built successfully!${NC}"

# Step 4: Run the Docker container
echo -e "\n${GREEN}==== Step 3: Running Docker Container ====${NC}"
echo -e "${YELLOW}Starting the ROS2 node in Docker container with GPU support...${NC}"
echo -e "${YELLOW}Note: This requires NVIDIA drivers to be installed on the host system${NC}"
echo -e "${YELLOW}Command to run the container:${NC}"
echo -e "docker run --gpus all --network host \\"
echo -e "  -v $(pwd)/optimized_models:/ros2_ws/src/pallet_detection/optimized_models \\"
echo -e "  pallet_detection:latest"

echo -e "\n${GREEN}==== Setup Complete ====${NC}"
echo -e "${YELLOW}To run the container, execute the above command.${NC}"
echo -e "${YELLOW}For more details, see MODEL_OPTIMIZATION.md${NC}" 