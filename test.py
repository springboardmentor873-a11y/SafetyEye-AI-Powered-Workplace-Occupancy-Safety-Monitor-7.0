from ultralytics import YOLO

model = YOLO(r"runs/detect/train-6/weights/best.pt")

results = model(
    r"C:\Users\hp\Downloads\construction site\source_files\source_files\construction-safety.jpg",
    save=True
)

results[0].show()   # image display cheyyan
print(results[0])