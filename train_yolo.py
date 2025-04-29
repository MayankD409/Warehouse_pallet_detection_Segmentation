from ultralytics import YOLO

model = YOLO("yolo11l.pt")
model.train(data = "Augment_annotated/data.yaml", epochs = 200, batch = 4, device = "cuda")

