from ultralytics import YOLO

# Load YOLOv8 Small model
model = YOLO("yolov8s.pt")

# Train model
results = model.train(
    data="ppe_data.yaml",
    epochs=95,
    imgsz=768,
    batch=4
)

# Validate model
metrics = model.val()

print("\n===== Evaluation Metrics =====")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)