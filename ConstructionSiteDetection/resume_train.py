from ultralytics import YOLO

model = YOLO("runs/detect/train-17/weights/last.pt")

model.train(resume=True)