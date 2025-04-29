# Model Optimization & Docker Setup for ROS2 Pallet Detection

This repository contains a ROS2 module for pallet detection and ground segmentation using optimized YOLO models.

## Model Optimization

The project provides a script for converting PyTorch models to ONNX format and applying optimization techniques:

1. ONNX conversion - Converting PyTorch models to the ONNX format for cross-platform compatibility
2. Model Simplification - Using ONNX simplifier to reduce complexity
3. Quantization - INT8 quantization to reduce model size and improve inference speed

### Running the Optimization Script

```bash
# Basic optimization (converts to ONNX and applies simplification)
python3 onnx_transform.py

# With quantization enabled
python3 onnx_transform.py --quantize

# With custom model paths
python3 onnx_transform.py --detection_model path/to/detection.pt --segmentation_model path/to/segmentation.pt
```

Optimized models will be saved in the `optimized_models` directory.

## Docker Container

The project provides a Docker container that packages the ROS2 module with all necessary dependencies. The container includes:

1. ROS2 Humble distribution
2. NVIDIA GPU support (requires NVIDIA drivers on the host)
3. Pre-installed dependencies (PyTorch, ONNX, etc.)
4. ROS2 package setup for the pallet detection module

### Building the Docker Image

```bash
docker build -t pallet_detection:latest .
```

### Running the Docker Container

To run the container with GPU support:

```bash
docker run --gpus all --network host \
  -v /path/to/your/optimized_models:/ros2_ws/src/pallet_detection/optimized_models \
  pallet_detection:latest
```

This will start the ROS2 node with the optimized models.

### Using Different Models

You can override the default launch parameters by specifying them when running the container:

```bash
docker run --gpus all --network host \
  -v /path/to/your/models:/ros2_ws/src/pallet_detection/optimized_models \
  pallet_detection:latest \
  ros2 launch pallet_detection pallet_detection.launch.py \
  use_onnx:=true \
  onnx_detection_model_path:=/ros2_ws/src/pallet_detection/optimized_models/your_detection_model.onnx \
  onnx_segmentation_model_path:=/ros2_ws/src/pallet_detection/optimized_models/your_segmentation_model.onnx
```

## Performance Considerations

- The optimized ONNX models provide approximately 2-3x faster inference compared to the original PyTorch models
- Quantized models offer further speedup with minimal accuracy loss
- For best performance, ensure your NVIDIA drivers are up-to-date 