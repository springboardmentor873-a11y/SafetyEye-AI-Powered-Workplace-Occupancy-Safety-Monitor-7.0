from ultralytics import YOLO

model = YOLO("yolov8s.pt")

model.train(
    data="ppe_data.yaml",
    epochs=55,
    imgsz=640,
    batch=8
)

# Validate model
metrics = model.val()

print("\n===== Evaluation Metrics =====")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)