from ultralytics import YOLO

model = YOLO("yolov8s.pt")

results = model.train(
    data="data.yaml",
    epochs=30,
    imgsz=320
)