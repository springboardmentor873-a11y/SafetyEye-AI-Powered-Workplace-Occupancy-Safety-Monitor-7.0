from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="dataset/archive/css-data/data.yaml",
    epochs=50,
    imgsz=768,
    batch=4
)