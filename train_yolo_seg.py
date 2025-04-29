from ultralytics import YOLO

model = YOLO("yolo11l-seg.pt")
model.train(data = "Augment_annotated/data.yaml", epochs = 200, batch = 2, device = "cuda")

