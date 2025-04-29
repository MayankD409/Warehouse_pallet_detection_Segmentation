from ultralytics import YOLO

model = YOLO("yolo11l-seg.pt")
# model.train(data = "/project/Pallets_segment/data.yaml", epochs = 200, batch = 4, device = "cuda")
model.train(data = "Augment_annotated/data.yaml", epochs = 200, batch = 2, device = "cuda")
# model.train(data = "/home/mayank/JOBS/Peer Robotics/Pallet_detection/Augment_annotated/data.yaml", epochs = 200, batch = 4, device = "cuda")

