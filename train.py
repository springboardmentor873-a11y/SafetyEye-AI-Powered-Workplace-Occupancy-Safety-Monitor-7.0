from ultralytics import YOLO

model = YOLO(r"runs/detect/train-4/weights/best.pt")

model.train(
    data=r"C:\Users\hp\Downloads\construction site\data.yaml",
    epochs=30,
    imgsz=640
)