# Simple ROS2 Pallet Detection Workspace

This is a minimal workspace for running the pallet inference node without building a full ROS2 package.

## Prerequisites

- ROS2 (tested with Humble)
- Python3 with required packages (see requirements.txt)

## Installation

Install the required Python packages:

```bash
pip3 install -r requirements.txt
```

## Running the Node

Source your ROS2 installation:

```bash
source /opt/ros/humble/setup.bash  # Adjust based on your ROS2 version
```

Run the node directly:

```bash
cd src
python3 pallet_inference_node.py
```

## Parameters

You can adjust the node parameters by modifying the script directly or by using ROS2 parameters when launching the node:

```bash
cd src
python3 pallet_inference_node.py --ros-args -p detection_model_path:=path/to/detection/model.pt -p segmentation_model_path:=path/to/segmentation/model.pt
```

## Testing with ROS2 Bag Files

If you have recorded ROS2 bag files with camera data, you can use them to test the pallet_inference_node.py. This allows you to replay real-world or simulated data without needing an actual camera.

### Playing a ROS2 Bag File

1. First, make sure your node is running in a separate terminal:

```bash
cd src
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