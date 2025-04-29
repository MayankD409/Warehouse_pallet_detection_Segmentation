# Warehouse Pallet Detection and Ground Segmentation

This project implements computer vision capabilities for warehouse automation using YOLO models for pallet detection and ground segmentation. It's designed as a ROS2 Python node to run inference on trained models.

## Features

- **Object Detection**: Identifies pallets and ground areas using YOLOv8
- **Semantic Segmentation**: Performs pixel-level segmentation of pallets and ground
- **ROS2 Integration**: Publishes results as image topics for visualization and downstream processing
- **Fallback Mechanism**: Uses detection model to augment segmentation when ground areas are missed

## Project Structure

- `pallet_inference_node.py`: Main ROS2 node for real-time detection and segmentation
- `test_yolo.py`: Script for testing the YOLO detection model
- `test_yolo_seg.py`: Script for testing the YOLO segmentation model
- `train_yolo.py`: Script for training the YOLO detection model
- `train_yolo_seg.py`: Script for training the YOLO segmentation model
- `ROS2Node_README.md`: Detailed instructions for running the ROS2 node

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MayankD409/Warehouse_pallet_detection_Segmentation.git
   cd Warehouse_pallet_detection_Segmentation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Source your ROS2 installation:
   ```bash
   source /opt/ros/humble/setup.bash  # Adjust based on your ROS2 version
   ```

## Usage

### Running the ROS2 Node

```bash
python3 pallet_inference_node.py
```

### Testing the Models

To test the detection model:
```bash
python3 test_yolo.py
```

To test the segmentation model:
```bash
python3 test_yolo_seg.py
```

## Parameters

The ROS2 node supports the following parameters:

- `detection_model_path`: Path to the YOLO detection model
- `segmentation_model_path`: Path to the YOLO segmentation model
- `confidence_threshold`: Confidence threshold for detection (default: 0.5)
- `segmentation_confidence_threshold`: Confidence threshold for segmentation (default: 0.35)
- `input_image_topic`: Input camera topic
- `output_detection_topic`: Output topic for detection results
- `output_segmentation_topic`: Output topic for segmentation results

## License

This project is licensed under the MIT License - see the LICENSE file for details. 