from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train-11/weights/best.pt")
# Evaluate model
metrics = model.val(data="ppe_data.yaml")

print("\n===== Evaluation Metrics =====")
print(f"Accuracy (mAP50): {metrics.box.map50 * 100:.2f}%")
print(f"mAP50: {metrics.box.map50:.4f}")
print(f"mAP50-95: {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.mp:.4f}")
print(f"Recall: {metrics.box.mr:.4f}")