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

## Setup and Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/MayankD409/Warehouse_pallet_detection_Segmentation.git
   cd Warehouse_pallet_detection_Segmentation
   ```

2. Install the required dependencies:
   ```bash
   pip3 install -r requirements.txt
   ```

3. Source your ROS2 installation:
   ```bash
   source /opt/ros/humble/setup.bash  # Adjust based on your ROS2 version
   ```

## Usage

### Running the ROS2 Node

```bash
cd src  # If you organized in a ROS2 package structure
python3 pallet_inference_node.py
```

You can adjust the node parameters by modifying the script directly or by using ROS2 parameters when launching the node:

```bash
python3 pallet_inference_node.py --ros-args -p detection_model_path:=path/to/detection/model.pt -p segmentation_model_path:=path/to/segmentation/model.pt
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

## Testing with ROS2 Bag Files

If you have recorded ROS2 bag files with camera data, you can use them to test the pallet_inference_node.py. This allows you to replay real-world or simulated data without needing an actual camera.

![Sample Bag Image](bag_image.png)

### Playing a ROS2 Bag File

1. First, make sure your node is running in a separate terminal:

```bash
python3 pallet_inference_node.py
```

2. In a new terminal, play your bag file:

```bash
ros2 bag play /path/to/your/bagfile --loop  # --loop is optional for continuous replay
```

If your bag file has a different topic name than the one expected by the node (`/robot1/zed2i/left/image_rect_color`), you can remap it:

```bash
ros2 bag play /path/to/your/bagfile --remap /bag_camera_topic:=/robot1/zed2i/left/image_rect_color
```

### Visualizing the Results

You can visualize the detection and segmentation results using RViz or rqt_image_view:

#### Using rqt_image_view:

```bash
ros2 run rqt_image_view rqt_image_view
```

Then from the dropdown menu at the top, select either:
- `/pallet_detection` - To view the detection results
- `/ground_segmentation` - To view the segmentation results

#### Using RViz:

```bash
ros2 run rviz2 rviz2
```

In RViz:
1. Add an Image display (click "Add" → "By topic" → select either `/pallet_detection` or `/ground_segmentation`)
2. You can add multiple Image displays to view both detection and segmentation results simultaneously

### Recording Your Own Bag Files

If you want to record your own bag files for testing:

```bash
ros2 bag record -o my_camera_bag /camera/image_raw  # Replace with your camera topic
```

You can record multiple topics simultaneously:

```bash
ros2 bag record -o my_bag /camera/image_raw /camera/camera_info
```

## Training Results

### Object Detection Training

The object detection model was trained using YOLOv8 to detect pallets in warehouse environments. Below are the training results:

![Detection Training Results](runs/detect/train/results.png)

#### Detection Model Performance
- Precision-Recall Curve:
  ![PR Curve](runs/detect/train/PR_curve.png)
- Confusion Matrix:
  ![Confusion Matrix](runs/detect/train/confusion_matrix_normalized.png)

### Segmentation Training

The segmentation model was trained to identify pallets and ground at the pixel level, enabling precise navigation and interaction.

![Segmentation Training Results](runs/segment/train/results.png)

#### Segmentation Model Performance
- Mask Precision-Recall Curve:
  ![Mask PR Curve](runs/segment/train/MaskPR_curve.png)
- Segmentation Examples:
  ![Validation Example](runs/segment/train/val_batch0_pred.jpg)