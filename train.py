from ultralytics import YOLO

model = YOLO(r"runs/detect/train-7/weights/best.pt")

model.train(
    data=r"C:\Users\hp\Downloads\construction site\data.yaml",
    epochs=70,
    imgsz=768
)