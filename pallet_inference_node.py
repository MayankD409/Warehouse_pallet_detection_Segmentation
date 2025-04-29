#!/usr/bin/env python3

import os
import cv2
import numpy as np
from pathlib import Path
import torch

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy, QoSDurabilityPolicy

from ultralytics import YOLO

# Define class mappings based on test_yolo.py and test_yolo_seg.py results
DETECTION_CLASS_MAP = {
    0: 'Ground',  # Class ID 0 is ground in detection model
    1: 'Pallet'   # Class ID 1 is pallet in detection model
}

SEGMENTATION_CLASS_MAP = {
    0: 'Ground',  # Class ID 0 is ground in segmentation model
    1: 'Pallet'   # Class ID 1 is pallet in segmentation model
}

# Class IDs for visualization
GROUND_CLASS_ID = 0
PALLET_CLASS_ID = 1


class PalletInferenceNode(Node):
    """
    ROS2 Node for pallet detection and ground segmentation using YOLO models.
    
    This node:
    - Subscribes to image topics from a camera
    - Runs inference on the input image for pallet detection and ground segmentation
    - Publishes the detection/segmentation results overlaid on the original image
    """
    
    def __init__(self):
        super().__init__('pallet_inference_node')
        
        # Parameters (can be made configurable later)
        self.declare_parameter('detection_model_path', 'trained_models/unoptimized/detect/best.pt')
        self.declare_parameter('segmentation_model_path', 'trained_models/unoptimized/segment/best.pt')
        self.declare_parameter('use_onnx', False)
        self.declare_parameter('onnx_detection_model_path', 'trained_models/optimized/best_detect_fp16.onnx')
        self.declare_parameter('onnx_segmentation_model_path', 'trained_models/optimized/best_segment_fp16.onnx')
        self.declare_parameter('confidence_threshold', 0.5)
        # Lower segmentation threshold for better ground detection
        self.declare_parameter('segmentation_confidence_threshold', 0.35)
        self.declare_parameter('input_image_topic', '/robot1/zed2i/left/image_rect_color')
        self.declare_parameter('output_detection_topic', '/pallet_detection')
        self.declare_parameter('output_segmentation_topic', '/ground_segmentation')
        
        # Get parameters
        detection_model_path = self.get_parameter('detection_model_path').value
        segmentation_model_path = self.get_parameter('segmentation_model_path').value
        use_onnx = self.get_parameter('use_onnx').value
        onnx_detection_model_path = self.get_parameter('onnx_detection_model_path').value
        onnx_segmentation_model_path = self.get_parameter('onnx_segmentation_model_path').value
        self.confidence_threshold = self.get_parameter('confidence_threshold').value
        self.segmentation_confidence_threshold = self.get_parameter('segmentation_confidence_threshold').value
        input_image_topic = self.get_parameter('input_image_topic').value
        output_detection_topic = self.get_parameter('output_detection_topic').value
        output_segmentation_topic = self.get_parameter('output_segmentation_topic').value
        
        # Initialize YOLO models
        if use_onnx:
            if os.path.exists(onnx_detection_model_path):
                self.get_logger().info(f"Loading optimized ONNX detection model from: {onnx_detection_model_path}")
                self.detection_model = YOLO(onnx_detection_model_path)
            else:
                self.get_logger().warn(f"ONNX detection model not found at {onnx_detection_model_path}, falling back to PyTorch model")
                self.get_logger().info(f"Loading detection model from: {detection_model_path}")
                self.detection_model = YOLO(detection_model_path)
            
            if os.path.exists(onnx_segmentation_model_path):
                self.get_logger().info(f"Loading optimized ONNX segmentation model from: {onnx_segmentation_model_path}")
                self.segmentation_model = YOLO(onnx_segmentation_model_path)
            else:
                self.get_logger().warn(f"ONNX segmentation model not found at {onnx_segmentation_model_path}, falling back to PyTorch model")
                self.get_logger().info(f"Loading segmentation model from: {segmentation_model_path}")
                self.segmentation_model = YOLO(segmentation_model_path)
        else:
            self.get_logger().info(f"Loading detection model from: {detection_model_path}")
            self.detection_model = YOLO(detection_model_path)
            
            self.get_logger().info(f"Loading segmentation model from: {segmentation_model_path}")
            self.segmentation_model = YOLO(segmentation_model_path)
        
        # Initialize the CV bridge
        self.bridge = CvBridge()
        
        # Create publishers for detection and segmentation results
        self.detection_publisher = self.create_publisher(
            Image, 
            output_detection_topic, 
            10
        )
        
        self.segmentation_publisher = self.create_publisher(
            Image, 
            output_segmentation_topic, 
            10
        )
        
        # Create QoS profile to match the bag file's publisher
        # BEST_EFFORT reliability is required to match the bag file publisher
        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5,
            durability=QoSDurabilityPolicy.VOLATILE
        )
        
        # Create subscriber for input image with matching QoS profile
        self.image_subscription = self.create_subscription(
            Image,
            input_image_topic,
            self.image_callback,
            sensor_qos
        )
        
        # Log initialization completion
        self.get_logger().info('Pallet inference node has been initialized')
        self.get_logger().info(f'Detection confidence threshold: {self.confidence_threshold}')
        self.get_logger().info(f'Segmentation confidence threshold: {self.segmentation_confidence_threshold}')

    def image_callback(self, msg):
        """Process incoming image messages and run models"""
        self.get_logger().debug('Received image message')
        
        try:
            # Convert ROS Image message to OpenCV image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # --- Create separate copies for detection and segmentation overlays ---
            detection_image = cv_image.copy()
            segmentation_image = cv_image.copy()

            # Run models
            detection_preds = self.detection_model(cv_image, verbose=False, conf=self.confidence_threshold)
            segmentation_preds = self.segmentation_model(cv_image, verbose=False, conf=self.segmentation_confidence_threshold)
            
            # Process results onto their respective images
            processed_detection_image = self.process_detection(detection_preds, detection_image)
            processed_segmentation_image = self.process_segmentation(segmentation_preds, segmentation_image, detection_preds)
            
            # Publish results
            self.publish_detection_results(processed_detection_image, msg.header)
            self.publish_segmentation_results(processed_segmentation_image, msg.header)
            
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_detection(self, predictions, image):
        """Process detection model predictions and overlay results on the image"""
        # Process predictions and draw bounding boxes
        self.get_logger().info("--- Detection --- ") # Debug: Start detection block
        for i, pred in enumerate(predictions):
            self.get_logger().info(f"Detection Pred {i+1}/{len(predictions)}:") # Debug
            if pred.boxes is None or len(pred.boxes) == 0:
                 self.get_logger().info("  No boxes detected.") # Debug
                 continue
            for j, bbox in enumerate(pred.boxes.data.tolist()):
                x1, y1, x2, y2, score, class_id = bbox
                class_id = int(class_id) # Convert class_id to integer
                
                # --- Debug: Print raw detection info BEFORE filtering ---
                self.get_logger().info(f"  Raw detection {j+1}: class_id={class_id}, score={score:.4f}")
                # --------------------------------------------------------

                # Check confidence threshold
                if score < self.confidence_threshold:
                    self.get_logger().info(f"    -> Skipped detection {j+1}: Low confidence ({score:.4f} < {self.confidence_threshold})")
                    continue
                
                # Convert coordinates to integers for OpenCV
                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                
                # Get the class name
                class_name = DETECTION_CLASS_MAP[class_id]
                
                # Draw bounding boxes with different colors based on class
                if class_id == PALLET_CLASS_ID:
                    # Green for Pallet
                    color = (0, 255, 0)
                elif class_id == GROUND_CLASS_ID:
                    # Blue for Ground
                    color = (255, 0, 0)
                else:
                    # Default color for any other classes
                    color = (0, 255, 255)
                
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f'{class_name}: {score:.2f}'
                cv2.putText(image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

    def process_segmentation(self, predictions, image, detection_preds=None):
        """Process segmentation model predictions and overlay results on the image"""
        # Process predictions and overlay segmentation masks
        self.get_logger().info("--- Segmentation --- ") # Debug: Start segmentation block
        
        # Counter for segmentation classes
        segment_count = {'ground': 0, 'pallet': 0}
        
        for i, pred in enumerate(predictions):
            self.get_logger().info(f"Segmentation Pred {i+1}/{len(predictions)}:") # Debug
            # If we have segmentation masks
            if hasattr(pred, 'masks') and pred.masks is not None and len(pred.masks.data) > 0:
                for j, (mask, box, cls) in enumerate(zip(pred.masks.data, pred.boxes.data, pred.boxes.cls)):
                    class_id = int(cls.item()) # Convert class_id to integer
                    score = box[4].item()
                    
                    # --- Debug: Print raw segmentation info BEFORE filtering ---
                    self.get_logger().info(f"  Raw segmentation {j+1}: class_id={class_id}, score={score:.4f}")
                    # --------------------------------------------------------
                    
                    # Check confidence threshold
                    if score < self.segmentation_confidence_threshold:
                        self.get_logger().info(f"    -> Skipped segmentation {j+1}: Low confidence ({score:.4f} < {self.segmentation_confidence_threshold})")
                        continue
                    
                    # Convert the mask to a format usable by OpenCV
                    mask_data = mask.cpu().numpy()
                    mask_data = cv2.resize(mask_data, (image.shape[1], image.shape[0]))
                    mask_bool = mask_data > 0.35  # Use lower threshold for mask
                    
                    # Apply a semi-transparent overlay with color based on class
                    if class_id == GROUND_CLASS_ID:
                        # Blue color for ground
                        color = (255, 0, 0)
                        segment_count['ground'] += 1
                    elif class_id == PALLET_CLASS_ID:
                        # Red color for pallet
                        color = (0, 0, 255)
                        segment_count['pallet'] += 1
                    else:
                        # Default color for any other classes
                        color = (255, 255, 0)
                    
                    overlay = image.copy()
                    overlay[mask_bool] = color
                    
                    # Apply the overlay with alpha blending
                    alpha = 0.4
                    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                    
                    # Add contour around the segmentation
                    contours, _ = cv2.findContours(
                        mask_bool.astype(np.uint8), 
                        cv2.RETR_EXTERNAL, 
                        cv2.CHAIN_APPROX_SIMPLE
                    )
                    cv2.drawContours(image, contours, -1, color, 2)
                    
                    # Label the segmentation
                    moments = cv2.moments(mask_bool.astype(np.uint8))
                    if moments["m00"] != 0:
                        cX = int(moments["m10"] / moments["m00"])
                        cY = int(moments["m01"] / moments["m00"])
                        
                        label = f'{SEGMENTATION_CLASS_MAP[class_id]}: {score:.2f}' # Use class map
                        cv2.putText(image, label, (cX, cY), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:
                 self.get_logger().info("  No masks detected or masks attribute missing.") # Debug

        # If ground segments are missing, use detection predictions to augment
        if segment_count['ground'] == 0 and detection_preds is not None:
            self.get_logger().info("No ground segments found. Using detection results to augment.")
            for det_pred in detection_preds:
                for bbox in det_pred.boxes.data.tolist():
                    x1, y1, x2, y2, score, class_id = bbox
                    class_id = int(class_id)
                    
                    if class_id == GROUND_CLASS_ID and score >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                        
                        # Create a rectangular mask for the ground
                        ground_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                        cv2.rectangle(ground_mask, (x1, y1), (x2, y2), 1, -1)  # Filled rectangle
                        
                        # Apply the mask with a different color to distinguish it
                        color = (200, 100, 0)  # Different blue for detection-based ground
                        overlay = image.copy()
                        overlay[ground_mask == 1] = color
                        
                        # Blend with lower opacity
                        alpha = 0.3
                        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
                        
                        # Draw dashed border to show it's an approximation
                        for i in range(0, x2-x1, 10):
                            cv2.line(image, (x1+i, y1), (min(x1+i+5, x2), y1), color, 2)
                            cv2.line(image, (x1+i, y2), (min(x1+i+5, x2), y2), color, 2)
                        for i in range(0, y2-y1, 10):
                            cv2.line(image, (x1, y1+i), (x1, min(y1+i+5, y2)), color, 2)
                            cv2.line(image, (x2, y1+i), (x2, min(y1+i+5, y2)), color, 2)
                        
                        # Add label
                        label = f'Ground (det): {score:.2f}'
                        cv2.putText(image, label, (x1, y1 - 5), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        
                        segment_count['ground'] += 1

        # Add a summary text
        summary = f"Segmented: {segment_count['ground']} ground, {segment_count['pallet']} pallets"
        cv2.putText(image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        self.get_logger().info(summary)

        return image

    # --- Renamed original methods to avoid confusion ---
    def run_detection(self, image):
        self.get_logger().warning("'run_detection' is deprecated, use 'process_detection' instead.")
        predictions = self.detection_model(image, verbose=False)
        return self.process_detection(predictions, image.copy())

    def run_segmentation(self, image):
        self.get_logger().warning("'run_segmentation' is deprecated, use 'process_segmentation' instead.")
        predictions = self.segmentation_model(image, verbose=False)
        return self.process_segmentation(predictions, image.copy())
    # --- End of deprecated methods ---
    
    def publish_detection_results(self, image, header):
        """Publish the detection results image"""
        try:
            # Convert OpenCV image to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            img_msg.header = header  # Preserve original header
            
            # Publish the message
            self.detection_publisher.publish(img_msg)
            self.get_logger().debug('Published detection results')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing detection results: {str(e)}')

    def publish_segmentation_results(self, image, header):
        """Publish the segmentation results image"""
        try:
            # Convert OpenCV image to ROS Image message
            img_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            img_msg.header = header  # Preserve original header
            
            # Publish the message
            self.segmentation_publisher.publish(img_msg)
            self.get_logger().debug('Published segmentation results')
            
        except Exception as e:
            self.get_logger().error(f'Error publishing segmentation results: {str(e)}')


def main(args=None):
    # Initialize the ROS client library
    rclpy.init(args=args)
    
    # Create the node
    node = PalletInferenceNode()
    
    # Spin the node to execute callbacks
    rclpy.spin(node)
    
    # Cleanup and shutdown
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main() 