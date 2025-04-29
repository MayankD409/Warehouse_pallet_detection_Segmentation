import os
from ultralytics import YOLO
import cv2

# Use bag_image.png for testing
image_name = "bag_image.png"
print(f"Using test image: {image_name}")

# Path to the best trained weights
ptName = os.path.join("runs", "detect", "train", "weights", "best.pt")
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

# Ensure res_img directory exists
os.makedirs("res_img", exist_ok=True)

# Load model and predict
model = YOLO(ptName)
image = cv2.imread(image_name)
if image is None:
    print(f"Failed to load image: {image_name}")
    exit(1)
    
pred = model(image)

confidence_threshold = 0.5
for p in pred:
    for bbox in p.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = bbox
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        class_id = int(class_id)
        
        # Different colors for different classes
        if class_id == 0:  # Ground
            color = (255, 0, 0)  # Blue
        else:  # Pallet
            color = (0, 255, 0)  # Green
            
        if score < confidence_threshold: continue
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 4)
        class_name = "Ground" if class_id == 0 else "Pallet"
        label = f'{class_name}: {score:.2f}'
        cv2.putText(image, label, (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    
    output_path = './res_img/detection_bag_image.png'
    cv2.imwrite(output_path, image)
    print(f"Result saved to: {output_path}")

# Skip validation step as we're only testing on one image
print("Detection test complete!")