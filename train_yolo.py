from ultralytics import YOLO

model = YOLO("yolo11l.pt")
# model.train(data = "/project/Pallets_detection/data.yaml", epochs = 200, batch = 4, device = "cuda")
model.train(data = "Augment_annotated/data.yaml", epochs = 200, batch = 4, device = "cuda")
# model.train(data = "/home/mayank/JOBS/Peer Robotics/Pallet_detection/Augment_annotated/data.yaml", epochs = 200, batch = 4, device = "cuda")

