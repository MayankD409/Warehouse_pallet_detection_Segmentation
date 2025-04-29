import os
from ultralytics import YOLO
import cv2
import numpy as np

# Use bag_image.png for testing
image_name = "bag_image.png"
print(f"Using test image: {image_name}")

# Path to the best trained weights
ptName = os.path.join("runs", "segment", "train", "weights", "best.pt")
if not os.path.exists(ptName):
    print(f"Warning: {ptName} not found. Please check the path to your trained weights.")
    possible_paths = []
    for root, dirs, files in os.walk("runs"):
        for file in files:
            if file == "best.pt":
                possible_path = os.path.join(root, file)
                possible_paths.append(possible_path)
                print(f"Possible weight file found at: {possible_path}")
    
    if possible_paths:
        ptName = possible_paths[0]
        print(f"Using weight file: {ptName}")
    else:
        print("No weight files found. Please check your training output.")

# Ensure seg_save directory exists
os.makedirs("seg_save", exist_ok=True)

print(f"Loading model from: {ptName}")
model = YOLO(ptName)
print(f"Running inference on: {image_name}")

# Custom implementation to visualize both classes
image = cv2.imread(image_name)
if image is None:
    print(f"Failed to load image: {image_name}")
    exit(1)

# Lower the confidence threshold to catch more ground areas and weak detections
confidence_threshold = 0.35
print(f"Using lower confidence threshold: {confidence_threshold}")

# Run detection model first to get ground regions
detection_model_path = os.path.join("runs", "detect", "train", "weights", "best.pt")
if not os.path.exists(detection_model_path):
    print("Detection model not found, searching for alternatives...")
    for root, dirs, files in os.walk("runs"):
        for file in files:
            if file == "best.pt" and "detect" in root:
                detection_model_path = os.path.join(root, file)
                print(f"Found detection model at: {detection_model_path}")
                break

if os.path.exists(detection_model_path):
    print(f"Using detection model to assist segmentation: {detection_model_path}")
    detection_model = YOLO(detection_model_path)
    detection_results = detection_model(image)
else:
    print("Detection model not found. Proceeding with only segmentation model.")
    detection_results = None

# Run segmentation model
results = model(image, conf=confidence_threshold)

# Create a copy of the image for custom visualization
output_image = image.copy()

# Process segmentation results
segment_count = {'ground': 0, 'pallet': 0}
for result in results:
    if hasattr(result, 'masks') and result.masks is not None:
        for i, (mask, box, cls) in enumerate(zip(result.masks.data, result.boxes.data, result.boxes.cls)):
            # Get class and confidence
            class_id = int(cls.item())
            conf = float(box[4].item())
            
            # Process mask
            if conf >= confidence_threshold:  # Apply confidence threshold
                # Convert mask to numpy array
                mask_array = mask.cpu().numpy()
                # Resize mask to image size
                mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]))
                # Convert to binary mask
                mask_binary = (mask_array > 0.35).astype(np.uint8)  # Lower mask threshold too
                
                # Use different colors for different classes
                if class_id == 0:  # Ground class
                    color = (255, 0, 0)  # Blue for Ground
                    segment_count['ground'] += 1
                else:  # Pallet class
                    color = (0, 0, 255)  # Red for Pallet
                    segment_count['pallet'] += 1
                
                # Create color overlay
                overlay = output_image.copy()
                overlay[mask_binary == 1] = color
                
                # Blend with original image
                alpha = 0.4
                cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
                
                # Draw contours
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(output_image, contours, -1, color, 2)
                
                # Add label
                x1, y1, x2, y2 = map(int, box[:4])
                class_name = "Ground" if class_id == 0 else "Pallet"
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(output_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# If ground segments are missing, use detection model to augment
if detection_results is not None and segment_count['ground'] == 0:
    print("No ground segments found. Using detection results to augment.")
    for det_result in detection_results:
        for box in det_result.boxes.data:
            class_id = int(box[5].item())
            conf = float(box[4].item())
            
            if class_id == 0 and conf >= confidence_threshold:  # Ground class from detection
                x1, y1, x2, y2 = map(int, box[:4])
                
                # Create a rectangular mask for the ground
                ground_mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
                cv2.rectangle(ground_mask, (x1, y1), (x2, y2), 1, -1)  # Filled rectangle
                
                # Apply the mask with a different color to distinguish augmented ground
                color = (200, 100, 0)  # Different blue for augmented ground
                overlay = output_image.copy()
                overlay[ground_mask == 1] = color
                
                # Blend with lower opacity
                alpha = 0.3
                cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)
                
                # Draw dashed border to show it's an approximation
                for i in range(0, x2-x1, 10):
                    cv2.line(output_image, (x1+i, y1), (min(x1+i+5, x2), y1), color, 2)
                    cv2.line(output_image, (x1+i, y2), (min(x1+i+5, x2), y2), color, 2)
                for i in range(0, y2-y1, 10):
                    cv2.line(output_image, (x1, y1+i), (x1, min(y1+i+5, y2)), color, 2)
                    cv2.line(output_image, (x2, y1+i), (x2, min(y1+i+5, y2)), color, 2)
                
                # Add label
                label = f"Ground (det): {conf:.2f}"
                cv2.putText(output_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                segment_count['ground'] += 1

# Add a summary text
summary = f"Segmented: {segment_count['ground']} ground, {segment_count['pallet']} pallets"
cv2.putText(output_image, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

# Save output image
output_path = './seg_save/segmentation_bag_image_improved.png'
cv2.imwrite(output_path, output_image)
print(f"Custom segmentation result saved to: {output_path}")
print(f"Segmentation summary: {summary}")

# Skip validation as we're only testing on one image
print("Segmentation test complete!")