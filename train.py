from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="dataset/archive/css-data/data.yaml",
    epochs=5,
    imgsz=640
)