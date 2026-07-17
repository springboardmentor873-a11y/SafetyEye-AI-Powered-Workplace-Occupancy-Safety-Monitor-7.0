from ultralytics import YOLO

model = YOLO("runs/detect/train-11/weights/best.pt")

metrics = model.val()

print("\n===== Evaluation Metrics =====")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)