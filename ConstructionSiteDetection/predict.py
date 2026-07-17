from ultralytics import YOLO

print("Loading model...")

model = YOLO("runs/detect/train-2/weights/best.pt")

print("Running prediction...")

results = model.predict(
    source="css-data/helmet.jpg",
    save=True,
    conf=0.25
)

print("Prediction completed!")